#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.
import ast
import ctypes
import hashlib
import os
from typing import Mapping, Any

from warp2 import config, types
from warp2.codegen import codegen
from warp2.codegen.struct import Struct
from warp2.module.function import Function
from warp2.module.kernel_hooks import KernelHooks


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

    def build_struct_recursive(self, struct: Struct):
        structs = []

        stack = [struct]
        while stack:
            s = stack.pop()

            structs.append(s)

            for var in s.vars.values():
                if isinstance(var.type, Struct):
                    stack.append(var.type)
                elif isinstance(var.type, types.array) and isinstance(var.type.dtype, Struct):
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
            source += codegen.codegen_struct(struct)

        # code-gen all imported functions
        for func in self.functions.keys():
            if func.native_snippet is None:
                source += codegen.codegen_func(
                    func.adj, c_func_name=func.native_func, device=device, options=self.options
                )
            else:
                source += codegen.codegen_snippet(
                    func.adj, name=func.key, snippet=func.native_snippet, adj_snippet=func.adj_native_snippet
                )

        for kernel in self.module.kernels.values():
            # each kernel gets an entry point in the module
            if not kernel.is_generic:
                source += codegen.codegen_kernel(kernel, device=device, options=self.options)
                source += codegen.codegen_module(kernel, device=device)
            else:
                for k in kernel.overloads.values():
                    source += codegen.codegen_kernel(k, device=device, options=self.options)
                    source += codegen.codegen_module(k, device=device)

        # add headers
        if device == "cpu":
            source = codegen.cpu_module_header + source
        else:
            source = codegen.cuda_module_header + source

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
            sig = types.get_signature(
                func.input_types.values(),
                func_name=func.key,
                arg_names=list(func.input_types.keys()),
            )
            sig_existing = types.get_signature(
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
                    if isinstance(func, Function) and func.module is not None:
                        add_ref(func.module)

                except Exception:
                    # Lookups may fail for builtins, but that's ok.
                    # Lookups may also fail for functions in this module that haven't been imported yet,
                    # and that's ok too (not an external reference).
                    pass

        # scan for structs
        for arg in adj.args:
            if isinstance(arg.type, Struct) and arg.type.module is not None:
                add_ref(arg.type.module)

    def hash_module(self):
        def get_annotations(obj: Any) -> Mapping[str, Any]:
            """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
            # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
            if isinstance(obj, type):
                return obj.__dict__.get("__annotations__", {})

            return getattr(obj, "__annotations__", {})

        def get_type_name(type_hint):
            if isinstance(type_hint, Struct):
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
            if types._constant_hash:
                h.update(types._constant_hash.digest())

            # recurse on references
            visited.add(module)

            sorted_deps = sorted(module.references, key=lambda m: m.name)
            for dep in sorted_deps:
                if dep not in visited:
                    dep_hash = hash_recursive(dep, visited)
                    h.update(dep_hash)

            return h.digest()

        return hash_recursive(self, visited=set())

    def codegen(self):
        builder = ModuleBuilder(self, self.options)
        cpp_source = builder.codegen("cpu")
        print(cpp_source)
