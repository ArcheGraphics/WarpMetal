#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import ctypes
import hashlib
import os

from warp import config


class ModuleBuilder:
    def __init__(self, module, options):
        self.functions = {}
        self.structs = {}
        self.options = options
        self.module = module

        # build all functions declared in the module
        for func in module.functions.values():
            for f in func.user_overloads.values():
                self.build_function(f)
                if f.custom_replay_func is not None:
                    self.build_function(f.custom_replay_func)

        # build all kernel entry points
        for kernel in module.kernels.values():
            if not kernel.is_generic:
                self.build_kernel(kernel)
            else:
                for k in kernel.overloads.values():
                    self.build_kernel(k)

    def build_struct_recursive(self, struct: warp.codegen.Struct):
        structs = []

        stack = [struct]
        while stack:
            s = stack.pop()

            structs.append(s)

            for var in s.vars.values():
                if isinstance(var.type, warp.codegen.Struct):
                    stack.append(var.type)
                elif isinstance(var.type, warp.types.array) and isinstance(var.type.dtype, warp.codegen.Struct):
                    stack.append(var.type.dtype)

        # Build them in reverse to generate a correct dependency order.
        for s in reversed(structs):
            self.build_struct(s)

    def build_struct(self, struct):
        self.structs[struct] = None

    def build_kernel(self, kernel):
        kernel.adj.build(self)

        if kernel.adj.return_var is not None:
            if kernel.adj.return_var.ctype() != "void":
                raise TypeError(f"Error, kernels can't have return values, got: {kernel.adj.return_var}")

    def build_function(self, func):
        if func in self.functions:
            return
        else:
            func.adj.build(self)

            # complete the function return type after we have analyzed it (inferred from return statement in ast)
            if not func.value_func:

                def wrap(adj):
                    def value_type(arg_types, kwds, templates):
                        if adj.return_var is None or len(adj.return_var) == 0:
                            return None
                        if len(adj.return_var) == 1:
                            return adj.return_var[0].type
                        else:
                            return [v.type for v in adj.return_var]

                    return value_type

                func.value_func = wrap(func.adj)

            # use dict to preserve import order
            self.functions[func] = None

    def codegen(self, device):
        source = ""

        # code-gen structs
        for struct in self.structs.keys():
            source += warp.codegen.codegen_struct(struct)

        # code-gen all imported functions
        for func in self.functions.keys():
            if func.native_snippet is None:
                source += warp.codegen.codegen_func(
                    func.adj, c_func_name=func.native_func, device=device, options=self.options
                )
            else:
                source += warp.codegen.codegen_snippet(
                    func.adj, name=func.key, snippet=func.native_snippet, adj_snippet=func.adj_native_snippet
                )

        for kernel in self.module.kernels.values():
            # each kernel gets an entry point in the module
            if not kernel.is_generic:
                source += warp.codegen.codegen_kernel(kernel, device=device, options=self.options)
                source += warp.codegen.codegen_module(kernel, device=device)
            else:
                for k in kernel.overloads.values():
                    source += warp.codegen.codegen_kernel(k, device=device, options=self.options)
                    source += warp.codegen.codegen_module(k, device=device)

        # add headers
        if device == "cpu":
            source = warp.codegen.cpu_module_header + source
        else:
            source = warp.codegen.cuda_module_header + source

        return source


# -----------------------------------------------------
# stores all functions and kernels for a Python module
# creates a hash of the function to use for checking
# build cache


class Module:
    def __init__(self, name, loader):
        self.name = name
        self.loader = loader

        self.kernels = {}
        self.functions = {}
        self.constants = []
        self.structs = {}

        self.cpu_module = None
        self.cuda_modules = {}  # module lookup by CUDA context

        self.cpu_build_failed = False
        self.cuda_build_failed = False

        self.options = {
            "max_unroll": 16,
            "enable_backward": config.enable_backward,
            "fast_math": False,
            "cuda_output": None,  # supported values: "ptx", "cubin", or None (automatic)
            "mode": config.mode,
        }

        # kernel hook lookup per device
        # hooks are stored with the module so they can be easily cleared when the module is reloaded.
        # -> See ``Module.get_kernel_hooks()``
        self.kernel_hooks = {}

        # Module dependencies are determined by scanning each function
        # and kernel for references to external functions and structs.
        #
        # When a referenced module is modified, all of its dependents need to be reloaded
        # on the next launch.  To detect this, a module's hash recursively includes
        # all of its references.
        # -> See ``Module.hash_module()``
        #
        # The dependency mechanism works for both static and dynamic (runtime) modifications.
        # When a module is reloaded at runtime, we recursively unload all of its
        # dependents, so that they will be re-hashed and reloaded on the next launch.
        # -> See ``get_module()``

        self.references = set()  # modules whose content we depend on
        self.dependents = set()  # modules that depend on our content

        # Since module hashing is recursive, we improve performance by caching the hash of the
        # module contents (kernel source, function source, and struct source).
        # After all kernels, functions, and structs are added to the module (usually at import time),
        # the content hash doesn't change.
        # -> See ``Module.hash_module_recursive()``

        self.content_hash = None

        # number of times module auto-generates kernel key for user
        # used to ensure unique kernel keys
        self.count = 0

    def register_struct(self, struct):
        self.structs[struct.key] = struct

        # for a reload of module on next launch
        self.unload()

    def register_kernel(self, kernel):
        self.kernels[kernel.key] = kernel

        self.find_references(kernel.adj)

        # for a reload of module on next launch
        self.unload()

    def register_function(self, func, skip_adding_overload=False):
        if func.key not in self.functions:
            self.functions[func.key] = func
        else:
            # Check whether the new function's signature match any that has
            # already been registered. If so, then we simply override it, as
            # Python would do it, otherwise we register it as a new overload.
            func_existing = self.functions[func.key]
            sig = warp.types.get_signature(
                func.input_types.values(),
                func_name=func.key,
                arg_names=list(func.input_types.keys()),
            )
            sig_existing = warp.types.get_signature(
                func_existing.input_types.values(),
                func_name=func_existing.key,
                arg_names=list(func_existing.input_types.keys()),
            )
            if sig == sig_existing:
                self.functions[func.key] = func
            elif not skip_adding_overload:
                func_existing.add_overload(func)

        self.find_references(func.adj)

        # for a reload of module on next launch
        self.unload()

    def generate_unique_kernel_key(self, key):
        unique_key = f"{key}_{self.count}"
        self.count += 1
        return unique_key

    # collect all referenced functions / structs
    # given the AST of a function or kernel
    def find_references(self, adj):
        def add_ref(ref):
            if ref is not self:
                self.references.add(ref)
                ref.dependents.add(self)

        # scan for function calls
        for node in ast.walk(adj.tree):
            if isinstance(node, ast.Call):
                try:
                    # try to resolve the function
                    func, _ = adj.resolve_static_expression(node.func, eval_types=False)

                    # if this is a user-defined function, add a module reference
                    if isinstance(func, warp.context.Function) and func.module is not None:
                        add_ref(func.module)

                except Exception:
                    # Lookups may fail for builtins, but that's ok.
                    # Lookups may also fail for functions in this module that haven't been imported yet,
                    # and that's ok too (not an external reference).
                    pass

        # scan for structs
        for arg in adj.args:
            if isinstance(arg.type, warp.codegen.Struct) and arg.type.module is not None:
                add_ref(arg.type.module)

    def hash_module(self):
        def get_annotations(obj: Any) -> Mapping[str, Any]:
            """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
            # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
            if isinstance(obj, type):
                return obj.__dict__.get("__annotations__", {})

            return getattr(obj, "__annotations__", {})

        def get_type_name(type_hint):
            if isinstance(type_hint, warp.codegen.Struct):
                return get_type_name(type_hint.cls)
            return type_hint

        def hash_recursive(module, visited):
            # Hash this module, including all referenced modules recursively.
            # The visited set tracks modules already visited to avoid circular references.

            # check if we need to update the content hash
            if not module.content_hash:
                # recompute content hash
                ch = hashlib.sha256()

                # struct source
                for struct in module.structs.values():
                    s = ",".join(
                        "{}: {}".format(name, get_type_name(type_hint))
                        for name, type_hint in get_annotations(struct.cls).items()
                    )
                    ch.update(bytes(s, "utf-8"))

                # functions source
                for func in module.functions.values():
                    s = func.adj.source
                    ch.update(bytes(s, "utf-8"))

                    if func.custom_grad_func:
                        s = func.custom_grad_func.adj.source
                        ch.update(bytes(s, "utf-8"))
                    if func.custom_replay_func:
                        s = func.custom_replay_func.adj.source

                    # cache func arg types
                    for arg, arg_type in func.adj.arg_types.items():
                        s = f"{arg}: {get_type_name(arg_type)}"
                        ch.update(bytes(s, "utf-8"))

                # kernel source
                for kernel in module.kernels.values():
                    ch.update(bytes(kernel.adj.source, "utf-8"))
                    # cache kernel arg types
                    for arg, arg_type in kernel.adj.arg_types.items():
                        s = f"{arg}: {get_type_name(arg_type)}"
                        ch.update(bytes(s, "utf-8"))
                    # for generic kernels the Python source is always the same,
                    # but we hash the type signatures of all the overloads
                    if kernel.is_generic:
                        for sig in sorted(kernel.overloads.keys()):
                            ch.update(bytes(sig, "utf-8"))

                module.content_hash = ch.digest()

            h = hashlib.sha256()

            # content hash
            h.update(module.content_hash)

            # configuration parameters
            for k in sorted(module.options.keys()):
                s = f"{k}={module.options[k]}"
                h.update(bytes(s, "utf-8"))

            # ensure to trigger recompilation if flags affecting kernel compilation are changed
            if config.verify_fp:
                h.update(bytes("verify_fp", "utf-8"))

            h.update(bytes(config.mode, "utf-8"))

            # compile-time constants (global)
            if warp.types._constant_hash:
                h.update(warp.types._constant_hash.digest())

            # recurse on references
            visited.add(module)

            sorted_deps = sorted(module.references, key=lambda m: m.name)
            for dep in sorted_deps:
                if dep not in visited:
                    dep_hash = hash_recursive(dep, visited)
                    h.update(dep_hash)

            return h.digest()

        return hash_recursive(self, visited=set())

    def load(self, device):
        from warp.utils.scoped_timer import ScopedTimer

        device = get_device(device)

        if device.is_cpu:
            # check if already loaded
            if self.cpu_module:
                return True
            # avoid repeated build attempts
            if self.cpu_build_failed:
                return False
            if not warp.is_cpu_available():
                raise RuntimeError("Failed to build CPU module because no CPU buildchain was found")
        else:
            # check if already loaded
            if device.context in self.cuda_modules:
                return True
            # avoid repeated build attempts
            if self.cuda_build_failed:
                return False
            if not warp.is_cuda_available():
                raise RuntimeError("Failed to build CUDA module because CUDA is not available")

        with ScopedTimer(f"Module {self.name} load on device '{device}'", active=not config.quiet):
            build_path = warp.build.kernel_bin_dir
            gen_path = warp.build.kernel_gen_dir

            if not os.path.exists(build_path):
                os.makedirs(build_path)
            if not os.path.exists(gen_path):
                os.makedirs(gen_path)

            module_name = "wp_" + self.name
            module_path = os.path.join(build_path, module_name)
            module_hash = self.hash_module()

            builder = ModuleBuilder(self, self.options)

            if device.is_cpu:
                obj_path = os.path.join(build_path, module_name)
                obj_path = obj_path + ".o"
                cpu_hash_path = module_path + ".cpu.hash"

                # check cache
                if config.cache_kernels and os.path.isfile(cpu_hash_path) and os.path.isfile(obj_path):
                    with open(cpu_hash_path, "rb") as f:
                        cache_hash = f.read()

                    if cache_hash == module_hash:
                        runtime.llvm.load_obj(obj_path.encode("utf-8"), module_name.encode("utf-8"))
                        self.cpu_module = module_name
                        return True

                # build
                try:
                    cpp_path = os.path.join(gen_path, module_name + ".cpp")

                    # write cpp sources
                    cpp_source = builder.codegen("cpu")

                    cpp_file = open(cpp_path, "w")
                    cpp_file.write(cpp_source)
                    cpp_file.close()

                    # build object code
                    with ScopedTimer("Compile x86", active=config.verbose):
                        warp.build.build_cpu(
                            obj_path,
                            cpp_path,
                            mode=self.options["mode"],
                            fast_math=self.options["fast_math"],
                            verify_fp=config.verify_fp,
                        )

                    # update cpu hash
                    with open(cpu_hash_path, "wb") as f:
                        f.write(module_hash)

                    # load the object code
                    runtime.llvm.load_obj(obj_path.encode("utf-8"), module_name.encode("utf-8"))
                    self.cpu_module = module_name

                except Exception as e:
                    self.cpu_build_failed = True
                    raise (e)

            elif device.is_cuda:
                # determine whether to use PTX or CUBIN
                if device.is_cubin_supported:
                    # get user preference specified either per module or globally
                    preferred_cuda_output = self.options.get("cuda_output") or config.cuda_output
                    if preferred_cuda_output is not None:
                        use_ptx = preferred_cuda_output == "ptx"
                    else:
                        # determine automatically: older drivers may not be able to handle PTX generated using newer
                        # CUDA Toolkits, in which case we fall back on generating CUBIN modules
                        use_ptx = runtime.driver_version >= runtime.toolkit_version
                else:
                    # CUBIN not an option, must use PTX (e.g. CUDA Toolkit too old)
                    use_ptx = True

                if use_ptx:
                    output_arch = min(device.arch, config.ptx_target_arch)
                    output_path = module_path + f".sm{output_arch}.ptx"
                else:
                    output_arch = device.arch
                    output_path = module_path + f".sm{output_arch}.cubin"

                cuda_hash_path = module_path + f".sm{output_arch}.hash"

                # check cache
                if config.cache_kernels and os.path.isfile(cuda_hash_path) and os.path.isfile(output_path):
                    with open(cuda_hash_path, "rb") as f:
                        cache_hash = f.read()

                    if cache_hash == module_hash:
                        cuda_module = warp.build.load_cuda(output_path, device)
                        if cuda_module is not None:
                            self.cuda_modules[device.context] = cuda_module
                            return True

                # build
                try:
                    cu_path = os.path.join(gen_path, module_name + ".cu")

                    # write cuda sources
                    cu_source = builder.codegen("cuda")

                    cu_file = open(cu_path, "w")
                    cu_file.write(cu_source)
                    cu_file.close()

                    # generate PTX or CUBIN
                    with ScopedTimer("Compile CUDA", active=config.verbose):
                        warp.build.build_cuda(
                            cu_path,
                            output_arch,
                            output_path,
                            config=self.options["mode"],
                            fast_math=self.options["fast_math"],
                            verify_fp=config.verify_fp,
                        )

                    # update cuda hash
                    with open(cuda_hash_path, "wb") as f:
                        f.write(module_hash)

                    # load the module
                    cuda_module = warp.build.load_cuda(output_path, device)
                    if cuda_module is not None:
                        self.cuda_modules[device.context] = cuda_module
                    else:
                        raise Exception("Failed to load CUDA module")

                except Exception as e:
                    self.cuda_build_failed = True
                    raise (e)

            return True

    def unload(self):
        if self.cpu_module:
            runtime.llvm.unload_obj(self.cpu_module.encode("utf-8"))
            self.cpu_module = None

        # need to unload the CUDA module from all CUDA contexts where it is loaded
        # note: we ensure that this doesn't change the current CUDA context
        if self.cuda_modules:
            saved_context = runtime.core.cuda_context_get_current()
            for context, module in self.cuda_modules.items():
                runtime.core.cuda_unload_module(context, module)
            runtime.core.cuda_context_set_current(saved_context)
            self.cuda_modules = {}

        # clear kernel hooks
        self.kernel_hooks = {}

        # clear content hash
        self.content_hash = None

    # lookup and cache kernel entry points based on name, called after compilation / module load
    def get_kernel_hooks(self, kernel, device):
        # get all hooks for this device
        device_hooks = self.kernel_hooks.get(device.context)
        if device_hooks is None:
            self.kernel_hooks[device.context] = device_hooks = {}

        # look up this kernel
        hooks = device_hooks.get(kernel)
        if hooks is not None:
            return hooks

        name = kernel.get_mangled_name()

        if device.is_cpu:
            func = ctypes.CFUNCTYPE(None)
            forward = func(
                runtime.llvm.lookup(self.cpu_module.encode("utf-8"), (name + "_cpu_forward").encode("utf-8"))
            )
            backward = func(
                runtime.llvm.lookup(self.cpu_module.encode("utf-8"), (name + "_cpu_backward").encode("utf-8"))
            )
        else:
            cu_module = self.cuda_modules[device.context]
            forward = runtime.core.cuda_get_kernel(
                device.context, cu_module, (name + "_cuda_kernel_forward").encode("utf-8")
            )
            backward = runtime.core.cuda_get_kernel(
                device.context, cu_module, (name + "_cuda_kernel_backward").encode("utf-8")
            )

        hooks = KernelHooks(forward, backward)
        device_hooks[kernel] = hooks
        return hooks
