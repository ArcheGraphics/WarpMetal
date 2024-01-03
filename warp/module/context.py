#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import ast
import inspect
import io
import os
import sys
from typing import Any, Callable, Tuple, List

from warp import types
from warp.codegen.adjoint import Adjoint
from warp.codegen.struct import Struct
from warp.module.function import Function
from warp.module.kernel import Kernel
from warp.module.module import Module

# global dictionary of modules
user_modules = {}


def get_module(name):
    # some modules might be manually imported using `importlib` without being
    # registered into `sys.modules`
    parent = sys.modules.get(name, None)
    parent_loader = None if parent is None else parent.__loader__

    if name in user_modules:
        # check if the Warp module was created using a different loader object
        # if so, we assume the file has changed and we recreate the module to
        # clear out old kernels / functions
        if user_modules[name].loader is not parent_loader:
            old_module = user_modules[name]

            # Unload the old module and recursively unload all of its dependents.
            # This ensures that dependent modules will be re-hashed and reloaded on next launch.
            # The visited set tracks modules already visited to avoid circular references.
            def unload_recursive(module, visited):
                module.unload()
                visited.add(module)
                for d in module.dependents:
                    if d not in visited:
                        unload_recursive(d, visited)

            unload_recursive(old_module, visited=set())

            # clear out old kernels, funcs, struct definitions
            old_module.kernels = {}
            old_module.functions = {}
            old_module.constants = []
            old_module.structs = {}
            old_module.loader = parent_loader

        return user_modules[name]

    else:
        # else Warp module didn't exist yet, so create a new one
        user_modules[name] = Module(name, parent_loader)
        return user_modules[name]


# decorator to register function, @func
def func(f):
    name = warp.codegen.make_full_qualified_name(f)

    m = get_module(f.__module__)
    Function(
        func=f, key=name, namespace="", module=m, value_func=None
    )  # value_type not known yet, will be inferred during Adjoint.build()

    # return the top of the list of overloads for this key
    return m.functions[name]


def func_native(snippet, adj_snippet=None):
    """
    Decorator to register native code snippet, @func_native
    """

    def snippet_func(f):
        name = warp.codegen.make_full_qualified_name(f)

        m = get_module(f.__module__)
        func = Function(
            func=f, key=name, namespace="", module=m, native_snippet=snippet, adj_native_snippet=adj_snippet
        )  # cuda snippets do not have a return value_type

        return m.functions[name]

    return snippet_func


def func_grad(forward_fn):
    """
    Decorator to register a custom gradient function for a given forward function.
    The function signature must correspond to one of the function overloads in the following way:
    the first part of the input arguments are the original input variables with the same types as their
    corresponding arguments in the original function, and the second part of the input arguments are the
    adjoint variables of the output variables (if available) of the original function with the same types as the
    output variables. The function must not return anything.
    """

    def wrapper(grad_fn):
        generic = any(types.type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        reverse_args = {}
        reverse_args.update(forward_fn.input_types)

        # create temporary Adjoint instance to analyze the function signature
        adj = Adjoint(
            grad_fn, skip_forward_codegen=True, skip_reverse_codegen=False, transformers=forward_fn.adj.transformers
        )

        grad_args = adj.args
        grad_sig = types.get_signature([arg.type for arg in grad_args], func_name=forward_fn.key)

        generic = any(types.type_is_generic(x.type) for x in grad_args)
        if generic:
            raise RuntimeError(
                f"Cannot define custom grad definition for {forward_fn.key} since the provided grad function has generic input arguments."
            )

        def match_function(f):
            # check whether the function overload f matches the signature of the provided gradient function
            if not hasattr(f.adj, "return_var"):
                f.adj.build(None)
            expected_args = list(f.input_types.items())
            if f.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in f.adj.return_var]
            if len(grad_args) != len(expected_args):
                return False
            if any(not types.types_equal(a.type, exp_type) for a, (_, exp_type) in zip(grad_args, expected_args)):
                return False
            return True

        def add_custom_grad(f: Function):
            # register custom gradient function
            f.custom_grad_func = Function(
                grad_fn,
                key=f.key,
                namespace=f.namespace,
                input_types=reverse_args,
                value_func=None,
                module=f.module,
                template_func=f.template_func,
                skip_forward_codegen=True,
                custom_reverse_mode=True,
                custom_reverse_num_input_args=len(f.input_types),
                skip_adding_overload=False,
                code_transformers=f.adj.transformers,
            )
            f.adj.skip_reverse_codegen = True

        if hasattr(forward_fn, "user_overloads") and len(forward_fn.user_overloads):
            # find matching overload for which this grad function is defined
            for sig, f in forward_fn.user_overloads.items():
                if not grad_sig.startswith(sig):
                    continue
                if match_function(f):
                    add_custom_grad(f)
                    return
            raise RuntimeError(
                f"No function overload found for gradient function {grad_fn.__qualname__} for function {forward_fn.key}"
            )
        else:
            # resolve return variables
            forward_fn.adj.build(None)

            expected_args = list(forward_fn.input_types.items())
            if forward_fn.adj.return_var is not None:
                expected_args += [(f"adj_ret_{var.label}", var.type) for var in forward_fn.adj.return_var]

            # check if the signature matches this function
            if match_function(forward_fn):
                add_custom_grad(forward_fn)
            else:
                raise RuntimeError(
                    f"Gradient function {grad_fn.__qualname__} for function {forward_fn.key} has an incorrect signature. The arguments must match the "
                    "forward function arguments plus the adjoint variables corresponding to the return variables:"
                    f"\n{', '.join(map(lambda nt: f'{nt[0]}: {nt[1].__name__}', expected_args))}"
                )

    return wrapper


def get_function_args(func):
    """Ensures that all function arguments are annotated and returns a dictionary mapping from argument name to its type."""
    import inspect

    argspec = inspect.getfullargspec(func)

    # use source-level argument annotations
    if len(argspec.annotations) < len(argspec.args):
        raise RuntimeError(f"Incomplete argument annotations on function {func.__qualname__}")
    return argspec.annotations


def func_replay(forward_fn):
    """
    Decorator to register a custom replay function for a given forward function.
    The replay function is the function version that is called in the forward phase of the backward pass (replay mode) and corresponds to the forward function by default.
    The provided function has to match the signature of one of the original forward function overloads.
    """

    def wrapper(replay_fn):
        generic = any(types.type_is_generic(x) for x in forward_fn.input_types.values())
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since functions with generic input arguments are not yet supported."
            )

        args = get_function_args(replay_fn)
        arg_types = list(args.values())
        generic = any(types.type_is_generic(x) for x in arg_types)
        if generic:
            raise RuntimeError(
                f"Cannot define custom replay definition for {forward_fn.key} since the provided replay function has generic input arguments."
            )

        f = forward_fn.get_overload(arg_types)
        if f is None:
            inputs_str = ", ".join([f"{k}: {v.__name__}" for k, v in args.items()])
            raise RuntimeError(
                f"Could not find forward definition of function {forward_fn.key} that matches custom replay definition with arguments:\n{inputs_str}"
            )
        f.custom_replay_func = Function(
            replay_fn,
            key=f"replay_{f.key}",
            namespace=f.namespace,
            input_types=f.input_types,
            value_func=f.value_func,
            module=f.module,
            template_func=f.template_func,
            skip_reverse_codegen=True,
            skip_adding_overload=True,
            code_transformers=f.adj.transformers,
        )

    return wrapper


# decorator to register kernel, @kernel, custom_name may be a string
# that creates a kernel with a different name from the actual function
def kernel(f=None, *, enable_backward=None):
    def wrapper(f, *args, **kwargs):
        options = {}

        if enable_backward is not None:
            options["enable_backward"] = enable_backward

        m = get_module(f.__module__)
        k = Kernel(
            func=f,
            key=warp.codegen.make_full_qualified_name(f),
            module=m,
            options=options,
        )
        return k

    if f is None:
        # Arguments were passed to the decorator.
        return wrapper

    return wrapper(f)


# decorator to register struct, @struct
def struct(c):
    m = get_module(c.__module__)
    s = Struct(cls=c, key=warp.codegen.make_full_qualified_name(c), module=m)

    return s


# overload a kernel with the given argument types
def overload(kernel, arg_types=None):
    if isinstance(kernel, Kernel):
        # handle cases where user calls us directly, e.g. wp.overload(kernel, [args...])

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        if isinstance(arg_types, list):
            arg_list = arg_types
        elif isinstance(arg_types, dict):
            # substitute named args
            arg_list = [a.type for a in kernel.adj.args]
            for arg_name, arg_type in arg_types.items():
                idx = kernel.arg_indices.get(arg_name)
                if idx is None:
                    raise RuntimeError(f"Invalid argument name '{arg_name}' in overload of kernel {kernel.key}")
                arg_list[idx] = arg_type
        elif arg_types is None:
            arg_list = []
        else:
            raise TypeError("Kernel overload types must be given in a list or dict")

        # return new kernel overload
        return kernel.add_overload(arg_list)

    elif isinstance(kernel, types.FunctionType):
        # handle cases where user calls us as a function decorator (@wp.overload)

        # ensure this function name corresponds to a kernel
        fn = kernel
        module = get_module(fn.__module__)
        kernel = module.kernels.get(fn.__name__)
        if kernel is None:
            raise RuntimeError(f"Failed to find a kernel named '{fn.__name__}' in module {fn.__module__}")

        if not kernel.is_generic:
            raise RuntimeError(f"Only generic kernels can be overloaded.  Kernel {kernel.key} is not generic")

        # ensure the function is defined without a body, only ellipsis (...), pass, or a string expression
        # TODO: show we allow defining a new body for kernel overloads?
        source = inspect.getsource(fn)
        tree = ast.parse(source)
        assert isinstance(tree, ast.Module)
        assert isinstance(tree.body[0], ast.FunctionDef)
        func_body = tree.body[0].body
        for node in func_body:
            if isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Str, ast.Ellipsis)):
                continue
            raise RuntimeError(
                "Illegal statement in kernel overload definition.  Only pass, ellipsis (...), comments, or docstrings are allowed"
            )

        # ensure all arguments are annotated
        argspec = inspect.getfullargspec(fn)
        if len(argspec.annotations) < len(argspec.args):
            raise RuntimeError(f"Incomplete argument annotations on kernel overload {fn.__name__}")

        # get type annotation list
        arg_list = []
        for arg_name, arg_type in argspec.annotations.items():
            if arg_name != "return":
                arg_list.append(arg_type)

        # add new overload, but we must return the original kernel from @wp.overload decorator!
        kernel.add_overload(arg_list)
        return kernel

    else:
        raise RuntimeError("wp.overload() called with invalid argument!")


builtin_functions = {}


def add_builtin(
        key,
        input_types={},
        value_type=None,
        value_func=None,
        template_func=None,
        doc="",
        namespace="wp::",
        variadic=False,
        initializer_list_func=None,
        export=True,
        group="Other",
        hidden=False,
        skip_replay=False,
        missing_grad=False,
        native_func=None,
        defaults=None,
):
    # wrap simple single-type functions with a value_func()
    if value_func is None:
        def value_func(args, kwds, templates):
            return value_type

    if initializer_list_func is None:
        def initializer_list_func(args, templates):
            return False

    if defaults is None:
        defaults = {}

    # Add specialized versions of this builtin if it's generic by matching arguments against
    # hard coded types. We do this so you can use hard coded warp types outside kernels:
    generic = any(types.type_is_generic(x) for x in input_types.values())
    if generic and export:
        # get a list of existing generic vector types (includes matrices and stuff)
        # so we can match arguments against them:
        generic_vtypes = [x for x in types.vector_types if hasattr(x, "_wp_generic_type_str_")]

        # deduplicate identical types:
        def typekey(t):
            return f"{t._wp_generic_type_str_}_{t._wp_type_params_}"

        typedict = {typekey(t): t for t in generic_vtypes}
        generic_vtypes = [typedict[k] for k in sorted(typedict.keys())]

        # collect the parent type names of all the generic arguments:
        def generic_names(l):
            for t in l:
                if hasattr(t, "_wp_generic_type_str_"):
                    yield t._wp_generic_type_str_
                elif types.type_is_generic_scalar(t):
                    yield t.__name__

        genericset = set(generic_names(input_types.values()))

        # for each of those type names, get a list of all hard coded types derived
        # from them:
        def derived(name):
            if name == "Float":
                return types.float_types
            elif name == "Scalar":
                return types.scalar_types
            elif name == "Int":
                return types.int_types
            return [x for x in generic_vtypes if x._wp_generic_type_str_ == name]

        gtypes = {k: derived(k) for k in genericset}

        # find the scalar data types supported by all the arguments by intersecting
        # sets:
        def scalar_type(t):
            if t in types.scalar_types:
                return t
            return [p for p in t._wp_type_params_ if p in types.scalar_types][0]

        scalartypes = [{scalar_type(x) for x in gtypes[k]} for k in gtypes.keys()]
        if scalartypes:
            scalartypes = scalartypes.pop().intersection(*scalartypes)

        scalartypes = list(scalartypes)
        scalartypes.sort(key=str)

        # generate function calls for each of these scalar types:
        for stype in scalartypes:
            # find concrete types for this scalar type (eg if the scalar type is float32
            # this dict will look something like this:
            # {"vec":[wp.vec2,wp.vec3,wp.vec4], "mat":[wp.mat22,wp.mat33,wp.mat44]})
            consistenttypes = {k: [x for x in v if scalar_type(x) == stype] for k, v in gtypes.items()}

            def typelist(param):
                if types.type_is_generic_scalar(param):
                    return [stype]
                if hasattr(param, "_wp_generic_type_str_"):
                    l = consistenttypes[param._wp_generic_type_str_]
                    return [x for x in l if types.types_equal(param, x, match_generic=True)]
                return [param]

            # gotta try generating function calls for all combinations of these argument types
            # now.
            import itertools

            typelists = [typelist(param) for param in input_types.values()]
            for argtypes in itertools.product(*typelists):
                # Some of these argument lists won't work, eg if the function is mul(), we won't be
                # able to do a matrix vector multiplication for a mat22 and a vec3, so we call value_func
                # on the generated argument list and skip generation if it fails.
                # This also gives us the return type, which we keep for later:
                try:
                    return_type = value_func(argtypes, {}, [])
                except Exception:
                    continue

                # The return_type might just be vector_t(length=3,dtype=wp.float32), so we've got to match that
                # in the list of hard coded types so it knows it's returning one of them:
                if hasattr(return_type, "_wp_generic_type_str_"):
                    return_type_match = [
                        x
                        for x in generic_vtypes
                        if x._wp_generic_type_str_ == return_type._wp_generic_type_str_
                           and x._wp_type_params_ == return_type._wp_type_params_
                    ]
                    if not return_type_match:
                        continue
                    return_type = return_type_match[0]

                # finally we can generate a function call for these concrete types:
                add_builtin(
                    key,
                    input_types=dict(zip(input_types.keys(), argtypes)),
                    value_type=return_type,
                    doc=doc,
                    namespace=namespace,
                    variadic=variadic,
                    initializer_list_func=initializer_list_func,
                    export=export,
                    group=group,
                    hidden=True,
                    skip_replay=skip_replay,
                    missing_grad=missing_grad,
                )

    func = Function(
        func=None,
        key=key,
        namespace=namespace,
        input_types=input_types,
        value_func=value_func,
        template_func=template_func,
        variadic=variadic,
        initializer_list_func=initializer_list_func,
        export=export,
        doc=doc,
        group=group,
        hidden=hidden,
        skip_replay=skip_replay,
        missing_grad=missing_grad,
        generic=generic,
        native_func=native_func,
        defaults=defaults,
    )

    if key in builtin_functions:
        builtin_functions[key].add_overload(func)
    else:
        builtin_functions[key] = func


def type_str(t):
    if t is None:
        return "None"
    elif t == Any:
        return "Any"
    elif t == Callable:
        return "Callable"
    elif t == Tuple[int, int]:
        return "Tuple[int, int]"
    elif isinstance(t, int):
        return str(t)
    elif isinstance(t, List):
        return "Tuple[" + ", ".join(map(type_str, t)) + "]"
    elif isinstance(t, types.array):
        return f"Array[{type_str(t.dtype)}]"
    elif isinstance(t, types.indexedarray):
        return f"IndexedArray[{type_str(t.dtype)}]"
    elif isinstance(t, types.fabricarray):
        return f"FabricArray[{type_str(t.dtype)}]"
    elif isinstance(t, types.indexedfabricarray):
        return f"IndexedFabricArray[{type_str(t.dtype)}]"
    elif hasattr(t, "_wp_generic_type_str_"):
        generic_type = t._wp_generic_type_str_

        # for concrete vec/mat types use the short name
        if t in types.vector_types:
            return t.__name__

        # for generic vector / matrix type use a Generic type hint
        if generic_type == "vec_t":
            # return f"Vector"
            return f"Vector[{type_str(t._wp_type_params_[0])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "quat_t":
            # return f"Quaternion"
            return f"Quaternion[{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "mat_t":
            # return f"Matrix"
            return f"Matrix[{type_str(t._wp_type_params_[0])},{type_str(t._wp_type_params_[1])},{type_str(t._wp_scalar_type_)}]"
        elif generic_type == "transform_t":
            # return f"Transformation"
            return f"Transformation[{type_str(t._wp_scalar_type_)}]"
        else:
            raise TypeError("Invalid vector or matrix dimensions")
    else:
        return t.__name__


def print_function(f, file, noentry=False):  # pragma: no cover
    """Writes a function definition to a file for use in reST documentation

    Args:
        f: The function being written
        file: The file object for output
        noentry: If True, then the :noindex: and :nocontentsentry: directive
          options will be added

    Returns:
        A bool indicating True if f was written to file
    """

    if f.hidden:
        return False

    args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

    return_type = ""

    try:
        # todo: construct a default value for each of the functions args
        # so we can generate the return type for overloaded functions
        return_type = " -> " + type_str(f.value_func(None, None, None))
    except Exception:
        pass

    print(f".. function:: {f.key}({args}){return_type}", file=file)
    if noentry:
        print("   :noindex:", file=file)
        print("   :nocontentsentry:", file=file)
    print("", file=file)

    if f.doc != "":
        if not f.missing_grad:
            print(f"   {f.doc}", file=file)
        else:
            print(f"   {f.doc} [1]_", file=file)
        print("", file=file)

    print(file=file)

    return True


def print_builtins(file):  # pragma: no cover
    header = (
        "..\n"
        "   Autogenerated File - Do not edit. Run build_docs.py to generate.\n"
        "\n"
        ".. functions:\n"
        ".. currentmodule:: warp\n"
        "\n"
        "Kernel Reference\n"
        "================"
    )

    print(header, file=file)

    # type definitions of all functions by group
    print("\nScalar Types", file=file)
    print("------------", file=file)

    for t in types.scalar_types:
        print(f".. class:: {t.__name__}", file=file)
    # Manually add wp.bool since it's inconvenient to add to wp.types.scalar_types:
    print(f".. class:: (types.bool.__name__)", file=file)

    print("\n\nVector Types", file=file)
    print("------------", file=file)

    for t in types.vector_types:
        print(f".. class:: {t.__name__}", file=file)

    print("\nGeneric Types", file=file)
    print("-------------", file=file)

    print(".. class:: Int", file=file)
    print(".. class:: Float", file=file)
    print(".. class:: Scalar", file=file)
    print(".. class:: Vector", file=file)
    print(".. class:: Matrix", file=file)
    print(".. class:: Quaternion", file=file)
    print(".. class:: Transformation", file=file)
    print(".. class:: Array", file=file)

    # build dictionary of all functions by group
    groups = {}

    for k, f in builtin_functions.items():
        # build dict of groups
        if f.group not in groups:
            groups[f.group] = []

        # append all overloads to the group
        for o in f.overloads:
            groups[f.group].append(o)

    # Keep track of what function names have been written
    written_functions = {}

    for k, g in groups.items():
        print("\n", file=file)
        print(k, file=file)
        print("---------------", file=file)

        for f in g:
            if f.key in written_functions:
                # Add :noindex: + :nocontentsentry: since Sphinx gets confused
                print_function(f, file=file, noentry=True)
            else:
                if print_function(f, file=file):
                    written_functions[f.key] = []

    # footnotes
    print(".. rubric:: Footnotes", file=file)
    print(".. [1] Note: function gradients not implemented for backpropagation.", file=file)


def export_stubs(file):  # pragma: no cover
    """Generates stub file for auto-complete of builtin functions"""

    import textwrap

    print(
        "# Autogenerated file, do not edit, this file provides stubs for builtins autocomplete in VSCode, PyCharm, etc",
        file=file,
    )
    print("", file=file)
    print("from typing import Any", file=file)
    print("from typing import Tuple", file=file)
    print("from typing import Callable", file=file)
    print("from typing import TypeVar", file=file)
    print("from typing import Generic", file=file)
    print("from typing import overload as over", file=file)
    print(file=file)

    # type hints, these need to be mirrored into the stubs file
    print('Length = TypeVar("Length", bound=int)', file=file)
    print('Rows = TypeVar("Rows", bound=int)', file=file)
    print('Cols = TypeVar("Cols", bound=int)', file=file)
    print('DType = TypeVar("DType")', file=file)

    print('Int = TypeVar("Int")', file=file)
    print('Float = TypeVar("Float")', file=file)
    print('Scalar = TypeVar("Scalar")', file=file)
    print("Vector = Generic[Length, Scalar]", file=file)
    print("Matrix = Generic[Rows, Cols, Scalar]", file=file)
    print("Quaternion = Generic[Float]", file=file)
    print("Transformation = Generic[Float]", file=file)
    print("Array = Generic[DType]", file=file)
    print("FabricArray = Generic[DType]", file=file)
    print("IndexedFabricArray = Generic[DType]", file=file)

    # prepend __init__.py
    with open(os.path.join(os.path.dirname(file.name), "__init__.py")) as header_file:
        # strip comment lines
        lines = [line for line in header_file if not line.startswith("#")]
        header = "".join(lines)

    print(header, file=file)
    print(file=file)

    for k, g in builtin_functions.items():
        for f in g.overloads:
            args = ", ".join(f"{k}: {type_str(v)}" for k, v in f.input_types.items())

            return_str = ""

            if f.export is False or f.hidden is True:  # or f.generic:
                continue

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = f.value_func(None, None, None)
                if return_type:
                    return_str = " -> " + type_str(return_type)

            except Exception:
                pass

            print("@over", file=file)
            print(f"def {f.key}({args}){return_str}:", file=file)
            print('    """', file=file)
            print(textwrap.indent(text=f.doc, prefix="    "), file=file)
            print('    """', file=file)
            print("    ...\n\n", file=file)


def export_builtins(file: io.TextIOBase):  # pragma: no cover
    def ctype_str(t):
        if isinstance(t, int):
            return "int"
        elif isinstance(t, float):
            return "float"
        else:
            return t.__name__

    file.write("namespace wp {\n\n")
    file.write('extern "C" {\n\n')

    for k, g in builtin_functions.items():
        for f in g.overloads:
            if f.export is False or f.generic:
                continue

            simple = True
            for k, v in f.input_types.items():
                if isinstance(v, types.array) or v == Any or v == Callable or v == Tuple:
                    simple = False
                    break

            # only export simple types that don't use arrays
            # or templated types
            if not simple or f.variadic:
                continue

            args = ", ".join(f"{ctype_str(v)} {k}" for k, v in f.input_types.items())
            params = ", ".join(f.input_types.keys())

            return_type = ""

            try:
                # todo: construct a default value for each of the functions args
                # so we can generate the return type for overloaded functions
                return_type = ctype_str(f.value_func(None, None, None))
            except Exception:
                continue

            if return_type.startswith("Tuple"):
                continue

            if args == "":
                file.write(f"WP_API void {f.mangled_name}({return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n")
            elif return_type == "None":
                file.write(f"WP_API void {f.mangled_name}({args}) {{ wp::{f.key}({params}); }}\n")
            else:
                file.write(
                    f"WP_API void {f.mangled_name}({args}, {return_type}* ret) {{ *ret = wp::{f.key}({params}); }}\n"
                )

    file.write('\n}  // extern "C"\n\n')
    file.write("}  // namespace wp\n")
