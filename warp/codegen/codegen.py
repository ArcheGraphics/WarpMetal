#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.
import ctypes
import re
from typing import Any, Mapping

from warp import float16, types
from warp.codegen.var import Var
from warp.module import builtins


def get_annotations(obj: Any) -> Mapping[str, Any]:
    """Alternative to `inspect.get_annotations()` for Python 3.9 and older."""
    # See https://docs.python.org/3/howto/annotations.html#accessing-the-annotations-dict-of-an-object-in-python-3-9-and-older
    if isinstance(obj, type):
        return obj.__dict__.get("__annotations__", {})

    return getattr(obj, "__annotations__", {})


def compute_type_str(base_name, template_params):
    if not template_params:
        return base_name

    def param2str(p):
        if isinstance(p, int):
            return str(p)
        elif hasattr(p, "_type_"):
            return f"wp::{p.__name__}"
        return p.__name__

    return f"{base_name}<{','.join(map(param2str, template_params))}>"


# ----------------
# code generation
cpu_module_header = """
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(wp::s_threadIdx)
#define builtin_tid2d(x, y) wp::tid(x, y, wp::s_threadIdx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, wp::s_threadIdx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, wp::s_threadIdx, dim)

"""

cuda_module_header = """
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

"""

struct_template = """
struct {name}
{{
{struct_body}

    CUDA_CALLABLE {name}({forward_args})
    {forward_initializers}
    {{
    }}

    CUDA_CALLABLE {name}& operator += (const {name}&) {{ return *this; }}

}};

static CUDA_CALLABLE void adj_{name}({reverse_args})
{{
{reverse_body}}}

CUDA_CALLABLE void adj_atomic_add({name}* p, {name} t)
{{
{atomic_add_body}}}


"""

cpu_forward_function_template = """
// {filename}:{lineno}
static {return_type} {name}(
    {forward_args})
{{
{forward_body}}}

"""

cpu_reverse_function_template = """
// {filename}:{lineno}
static void adj_{name}(
    {reverse_args})
{{
{reverse_body}}}

"""

cuda_forward_function_template = """
// {filename}:{lineno}
static CUDA_CALLABLE {return_type} {name}(
    {forward_args})
{{
{forward_body}}}

"""

cuda_reverse_function_template = """
// {filename}:{lineno}
static CUDA_CALLABLE void adj_{name}(
    {reverse_args})
{{
{reverse_body}}}

"""

cuda_kernel_template = """

extern "C" __global__ void {name}_cuda_kernel_forward(
    {forward_args})
{{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {{
{forward_body}}}}}

extern "C" __global__ void {name}_cuda_kernel_backward(
    {reverse_args})
{{
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x)) {{
{reverse_body}}}}}

"""

cpu_kernel_template = """

void {name}_cpu_kernel_forward(
    {forward_args})
{{
{forward_body}}}

void {name}_cpu_kernel_backward(
    {reverse_args})
{{
{reverse_body}}}

"""

cpu_module_template = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    {forward_args})
{{
    for (size_t i=0; i < dim.size; ++i)
    {{
        wp::s_threadIdx = i;

        {name}_cpu_kernel_forward(
            {forward_params});
    }}
}}

WP_API void {name}_cpu_backward(
    {reverse_args})
{{
    for (size_t i=0; i < dim.size; ++i)
    {{
        wp::s_threadIdx = i;

        {name}_cpu_kernel_backward(
            {reverse_params});
    }}
}}

}} // extern C

"""

cuda_module_header_template = """

extern "C" {{

// Python CUDA entry points
WP_API void {name}_cuda_forward(
    void* stream,
    {forward_args});

WP_API void {name}_cuda_backward(
    void* stream,
    {reverse_args});

}} // extern C
"""

cpu_module_header_template = """

extern "C" {{

// Python CPU entry points
WP_API void {name}_cpu_forward(
    {forward_args});

WP_API void {name}_cpu_backward(
    {reverse_args});

}} // extern C
"""


# converts a constant Python value to equivalent C-repr
def constant_str(value):
    value_type = type(value)

    if value_type == bool or value_type == builtins.bool:
        if value:
            return "true"
        else:
            return "false"

    elif value_type == str:
        # ensure constant strings are correctly escaped
        return '"' + str(value.encode("unicode-escape").decode()) + '"'

    elif isinstance(value, ctypes.Array):
        if value_type._wp_scalar_type_ == float16:
            # special case for float16, which is stored as uint16 in the ctypes.Array
            scalar_value = runtime.core.half_bits_to_float
        else:
            scalar_value = lambda x: x

        # list of scalar initializer values
        initlist = []
        for i in range(value._length_):
            x = ctypes.Array.__getitem__(value, i)
            initlist.append(str(scalar_value(x)))

        dtypestr = f"wp::initializer_array<{value._length_},wp::{value._wp_scalar_type_.__name__}>"

        # construct value from initializer array, e.g. wp::initializer_array<4,wp::float32>{1.0, 2.0, 3.0, 4.0}
        return f"{dtypestr}{{{', '.join(initlist)}}}"

    elif value_type in types.scalar_types:
        # make sure we emit the value of objects, e.g. uint32
        return str(value.value)

    else:
        # otherwise just convert constant to string
        return str(value)


def indent(args, stops=1):
    sep = ",\n"
    for i in range(stops):
        sep += "    "

    # return sep + args.replace(", ", "," + sep)
    return sep.join(args)


# generates a C function name based on the python function name
def make_full_qualified_name(func):
    if not isinstance(func, str):
        func = func.__qualname__
    return re.sub("[^0-9a-zA-Z_]+", "", func.replace(".", "__"))


def codegen_struct(struct, device="cpu", indent_size=4):
    name = make_full_qualified_name(struct.cls)

    body = []
    indent_block = " " * indent_size

    if len(struct.vars) > 0:
        for label, var in struct.vars.items():
            body.append(var.ctype() + " " + label + ";\n")
    else:
        # for empty structs, emit the dummy attribute to avoid any compiler-specific alignment issues
        body.append("char _dummy_;\n")

    forward_args = []
    reverse_args = []

    forward_initializers = []
    reverse_body = []
    atomic_add_body = []

    # forward args
    for label, var in struct.vars.items():
        var_ctype = var.ctype()
        forward_args.append(f"{var_ctype} const& {label} = {{}}")
        reverse_args.append(f"{var_ctype} const&")

        namespace = "wp::" if var_ctype.startswith("wp::") or var_ctype == "bool" else ""
        atomic_add_body.append(f"{indent_block}{namespace}adj_atomic_add(&p->{label}, t.{label});\n")

        prefix = f"{indent_block}," if forward_initializers else ":"
        forward_initializers.append(f"{indent_block}{prefix} {label}{{{label}}}\n")

    # reverse args
    for label, var in struct.vars.items():
        reverse_args.append(var.ctype() + " & adj_" + label)
        if types.is_array(var.type):
            reverse_body.append(f"{indent_block}adj_{label} = adj_ret.{label};\n")
        else:
            reverse_body.append(f"{indent_block}adj_{label} += adj_ret.{label};\n")

    reverse_args.append(name + " & adj_ret")

    return struct_template.format(
        name=name,
        struct_body="".join([indent_block + l for l in body]),
        forward_args=indent(forward_args),
        forward_initializers="".join(forward_initializers),
        reverse_args=indent(reverse_args),
        reverse_body="".join(reverse_body),
        atomic_add_body="".join(atomic_add_body),
    )


def codegen_func_forward_body(adj, device="cpu", indent=4):
    body = []
    indent_block = " " * indent

    for f in adj.blocks[0].body_forward:
        body += [f + "\n"]

    return "".join([indent_block + l for l in body])


def codegen_func_forward(adj, func_type="kernel", device="cpu"):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:
        if var.constant is None:
            s += f"    {var.ctype()} {var.emit()};\n"
        else:
            s += f"    const {var.ctype()} {var.emit()} = {constant_str(var.constant)};\n"

    # forward pass
    s += "    //---------\n"
    s += "    // forward\n"

    if device == "cpu":
        s += codegen_func_forward_body(adj, device=device, indent=4)

    elif device == "cuda":
        if func_type == "kernel":
            s += codegen_func_forward_body(adj, device=device, indent=8)
        else:
            s += codegen_func_forward_body(adj, device=device, indent=4)

    return s


def codegen_func_reverse_body(adj, device="cpu", indent=4, func_type="kernel"):
    body = []
    indent_block = " " * indent

    # forward pass
    body += ["//---------\n"]
    body += ["// forward\n"]

    for f in adj.blocks[0].body_replay:
        body += [f + "\n"]

    # reverse pass
    body += ["//---------\n"]
    body += ["// reverse\n"]

    for l in reversed(adj.blocks[0].body_reverse):
        body += [l + "\n"]

    # In grid-stride kernels the reverse body is in a for loop
    if device == "cuda" and func_type == "kernel":
        body += ["continue;\n"]
    else:
        body += ["return;\n"]

    return "".join([indent_block + l for l in body])


def codegen_func_reverse(adj, func_type="kernel", device="cpu"):
    s = ""

    # primal vars
    s += "    //---------\n"
    s += "    // primal vars\n"

    for var in adj.variables:
        if var.constant is None:
            s += f"    {var.ctype()} {var.emit()};\n"
        else:
            s += f"    const {var.ctype()} {var.emit()} = {constant_str(var.constant)};\n"

    # dual vars
    s += "    //---------\n"
    s += "    // dual vars\n"

    for var in adj.variables:
        s += f"    {var.ctype(value_type=True)} {var.emit_adj()} = {{}};\n"

    if device == "cpu":
        s += codegen_func_reverse_body(adj, device=device, indent=4)
    elif device == "cuda":
        if func_type == "kernel":
            s += codegen_func_reverse_body(adj, device=device, indent=8, func_type=func_type)
        else:
            s += codegen_func_reverse_body(adj, device=device, indent=4, func_type=func_type)
    else:
        raise ValueError(f"Device {device} not supported for codegen")

    return s


def codegen_func(adj, c_func_name: str, device="cpu", options={}):
    # forward header
    if adj.return_var is not None and len(adj.return_var) == 1:
        return_type = adj.return_var[0].ctype()
    else:
        return_type = "void"

    has_multiple_outputs = adj.return_var is not None and len(adj.return_var) != 1

    forward_args = []
    reverse_args = []

    # forward args
    for i, arg in enumerate(adj.args):
        s = f"{arg.ctype()} {arg.emit()}"
        forward_args.append(s)
        if not adj.custom_reverse_mode or i < adj.custom_reverse_num_input_args:
            reverse_args.append(s)
    if has_multiple_outputs:
        for i, arg in enumerate(adj.return_var):
            forward_args.append(arg.ctype() + " & ret_" + str(i))
            reverse_args.append(arg.ctype() + " & ret_" + str(i))

    # reverse args
    for i, arg in enumerate(adj.args):
        if adj.custom_reverse_mode and i >= adj.custom_reverse_num_input_args:
            break
        # indexed array gradients are regular arrays
        if isinstance(arg.type, types.indexedarray):
            _arg = Var(arg.label, types.array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)
    if has_multiple_outputs:
        for i, arg in enumerate(adj.return_var):
            reverse_args.append(arg.ctype() + " & adj_ret_" + str(i))
    elif return_type != "void":
        reverse_args.append(return_type + " & adj_ret")
    # custom output reverse args (user-declared)
    if adj.custom_reverse_mode:
        for arg in adj.args[adj.custom_reverse_num_input_args:]:
            reverse_args.append(f"{arg.ctype()} & {arg.emit()}")

    if device == "cpu":
        forward_template = cpu_forward_function_template
        reverse_template = cpu_reverse_function_template
    elif device == "cuda":
        forward_template = cuda_forward_function_template
        reverse_template = cuda_reverse_function_template
    else:
        raise ValueError(f"Device {device} is not supported")

    # codegen body
    forward_body = codegen_func_forward(adj, func_type="function", device=device)

    s = ""
    if not adj.skip_forward_codegen:
        s += forward_template.format(
            name=c_func_name,
            return_type=return_type,
            forward_args=indent(forward_args),
            forward_body=forward_body,
            filename=adj.filename,
            lineno=adj.fun_lineno,
        )

    if not adj.skip_reverse_codegen:
        if adj.custom_reverse_mode:
            reverse_body = "\t// user-defined adjoint code\n" + forward_body
        else:
            if options.get("enable_backward", True):
                reverse_body = codegen_func_reverse(adj, func_type="function", device=device)
            else:
                reverse_body = '\t// reverse mode disabled (module option "enable_backward" is False)\n'
        s += reverse_template.format(
            name=c_func_name,
            return_type=return_type,
            reverse_args=indent(reverse_args),
            forward_body=forward_body,
            reverse_body=reverse_body,
            filename=adj.filename,
            lineno=adj.fun_lineno,
        )

    return s


def codegen_snippet(adj, name, snippet, adj_snippet):
    forward_args = []
    reverse_args = []

    # forward args
    for i, arg in enumerate(adj.args):
        s = f"{arg.ctype()} {arg.emit().replace('var_', '')}"
        forward_args.append(s)
        reverse_args.append(s)

    # reverse args
    for i, arg in enumerate(adj.args):
        if isinstance(arg.type, types.indexedarray):
            _arg = Var(arg.label, types.array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " & adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " & adj_" + arg.label)

    forward_template = cuda_forward_function_template
    reverse_template = cuda_reverse_function_template

    s = ""
    s += forward_template.format(
        name=name,
        return_type="void",
        forward_args=indent(forward_args),
        forward_body=snippet,
        filename=adj.filename,
        lineno=adj.fun_lineno,
    )

    if adj_snippet:
        reverse_body = adj_snippet
    else:
        reverse_body = ""

    s += reverse_template.format(
        name=name,
        return_type="void",
        reverse_args=indent(reverse_args),
        forward_body=snippet,
        reverse_body=reverse_body,
        filename=adj.filename,
        lineno=adj.fun_lineno,
    )

    return s


def codegen_kernel(kernel, device, options):
    # Update the module's options with the ones defined on the kernel, if any.
    options = dict(options)
    options.update(kernel.options)

    adj = kernel.adj

    forward_args = ["wp::launch_bounds_t dim"]
    reverse_args = ["wp::launch_bounds_t dim"]

    # forward args
    for arg in adj.args:
        forward_args.append(arg.ctype() + " var_" + arg.label)
        reverse_args.append(arg.ctype() + " var_" + arg.label)

    # reverse args
    for arg in adj.args:
        # indexed array gradients are regular arrays
        if isinstance(arg.type, types.indexedarray):
            _arg = Var(arg.label, types.array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(_arg.ctype() + " adj_" + arg.label)
        else:
            reverse_args.append(arg.ctype() + " adj_" + arg.label)

    # codegen body
    forward_body = codegen_func_forward(adj, func_type="kernel", device=device)

    if options["enable_backward"]:
        reverse_body = codegen_func_reverse(adj, func_type="kernel", device=device)
    else:
        reverse_body = ""

    if device == "cpu":
        template = cpu_kernel_template
    elif device == "cuda":
        template = cuda_kernel_template
    else:
        raise ValueError(f"Device {device} is not supported")

    s = template.format(
        name=kernel.get_mangled_name(),
        forward_args=indent(forward_args),
        reverse_args=indent(reverse_args),
        forward_body=forward_body,
        reverse_body=reverse_body,
    )

    return s


def codegen_module(kernel, device="cpu"):
    if device != "cpu":
        return ""

    adj = kernel.adj

    # build forward signature
    forward_args = ["wp::launch_bounds_t dim"]
    forward_params = ["dim"]

    for arg in adj.args:
        if hasattr(arg.type, "_wp_generic_type_str_"):
            # vectors and matrices are passed from Python by pointer
            forward_args.append(f"const {arg.ctype()}* var_" + arg.label)
            forward_params.append(f"*var_{arg.label}")
        else:
            forward_args.append(f"{arg.ctype()} var_{arg.label}")
            forward_params.append("var_" + arg.label)

    # build reverse signature
    reverse_args = [*forward_args]
    reverse_params = [*forward_params]

    for arg in adj.args:
        if isinstance(arg.type, types.indexedarray):
            # indexed array gradients are regular arrays
            _arg = Var(arg.label, types.array(dtype=arg.type.dtype, ndim=arg.type.ndim))
            reverse_args.append(f"const {_arg.ctype()} adj_{arg.label}")
            reverse_params.append(f"adj_{_arg.label}")
        elif hasattr(arg.type, "_wp_generic_type_str_"):
            # vectors and matrices are passed from Python by pointer
            reverse_args.append(f"const {arg.ctype()}* adj_{arg.label}")
            reverse_params.append(f"*adj_{arg.label}")
        else:
            reverse_args.append(f"{arg.ctype()} adj_{arg.label}")
            reverse_params.append(f"adj_{arg.label}")

    s = cpu_module_template.format(
        name=kernel.get_mangled_name(),
        forward_args=indent(forward_args),
        reverse_args=indent(reverse_args),
        forward_params=indent(forward_params, 3),
        reverse_params=indent(reverse_params, 3),
    )
    return s
