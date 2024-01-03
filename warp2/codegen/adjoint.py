#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import ast
import inspect
import textwrap
import builtins
import sys
from typing import List, Any, Callable

from warp2 import types, config
from warp2.codegen.block import Block
from warp2.codegen.codegen import compute_type_str
from warp2.codegen.reference import is_reference, Reference, strip_reference
from warp2.codegen.struct import Struct
from warp2.codegen.var import Var
from warp2.module.context import builtin_functions
from warp2.module.function import Function
from warp2.module.module import ModuleBuilder
from warp2.types import types_equal, type_repr, type_is_value, type_scalar_type, type_length, type_is_vector, is_array, \
    type_is_matrix


class WarpCodegenError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenTypeError(TypeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenAttributeError(AttributeError):
    def __init__(self, message):
        super().__init__(message)


class WarpCodegenKeyError(KeyError):
    def __init__(self, message):
        super().__init__(message)


# map operator to function name
builtin_operators = {}

# see https://www.ics.uci.edu/~pattis/ICS-31/lectures/opexp.pdf for a
# nice overview of python operators

builtin_operators[ast.Add] = "add"
builtin_operators[ast.Sub] = "sub"
builtin_operators[ast.Mult] = "mul"
builtin_operators[ast.MatMult] = "mul"
builtin_operators[ast.Div] = "div"
builtin_operators[ast.FloorDiv] = "floordiv"
builtin_operators[ast.Pow] = "pow"
builtin_operators[ast.Mod] = "mod"
builtin_operators[ast.UAdd] = "pos"
builtin_operators[ast.USub] = "neg"
builtin_operators[ast.Not] = "unot"

builtin_operators[ast.Gt] = ">"
builtin_operators[ast.Lt] = "<"
builtin_operators[ast.GtE] = ">="
builtin_operators[ast.LtE] = "<="
builtin_operators[ast.Eq] = "=="
builtin_operators[ast.NotEq] = "!="

builtin_operators[ast.BitAnd] = "bit_and"
builtin_operators[ast.BitOr] = "bit_or"
builtin_operators[ast.BitXor] = "bit_xor"
builtin_operators[ast.Invert] = "invert"
builtin_operators[ast.LShift] = "lshift"
builtin_operators[ast.RShift] = "rshift"

comparison_chain_strings = [
    builtin_operators[ast.Gt],
    builtin_operators[ast.Lt],
    builtin_operators[ast.LtE],
    builtin_operators[ast.GtE],
    builtin_operators[ast.Eq],
    builtin_operators[ast.NotEq],
]


def op_str_is_chainable(op: str) -> builtins.bool:
    return op in comparison_chain_strings


def format_template(template, input_vars, output_var):
    # code generation methods

    # output var is always the 0th index
    args = [output_var] + input_vars
    s = template.format(*args)

    return s


def format_args(prefix: str, args):
    # generates a list of formatted args

    arg_strs = []

    for a in args:
        if isinstance(a, Function):
            # functions don't have a var_ prefix so strip it off here
            if prefix == "var":
                arg_strs.append(a.key)
            else:
                arg_strs.append(f"{prefix}_{a.key}")
        elif is_reference(a.type):
            arg_strs.append(f"{prefix}_{a}")
        elif isinstance(a, Var):
            arg_strs.append(a.emit(prefix))
        else:
            raise WarpCodegenTypeError(f"Arguments must be variables or functions, got {type(a)}")

    return arg_strs


def format_forward_call_args(args, use_initializer_list):
    # generates argument string for a forward function call

    arg_str = ", ".join(format_args("var", args))
    if use_initializer_list:
        return f"{{{arg_str}}}"
    return arg_str


def format_reverse_call_args(
        args_var,
        args,
        args_out,
        use_initializer_list,
        has_output_args=True,
):
    # generates argument string for a reverse function call

    formatted_var = format_args("var", args_var)
    formatted_out = []
    if has_output_args and len(args_out) > 1:
        formatted_out = format_args("var", args_out)
    formatted_var_adj = format_args(
        "&adj" if use_initializer_list else "adj",
        args,
    )
    formatted_out_adj = format_args("adj", args_out)

    if len(formatted_var_adj) == 0 and len(formatted_out_adj) == 0:
        # there are no adjoint arguments, so we don't need to call the reverse function
        return None

    if use_initializer_list:
        var_str = f"{{{', '.join(formatted_var)}}}"
        out_str = f"{{{', '.join(formatted_out)}}}"
        adj_str = f"{{{', '.join(formatted_var_adj)}}}"
        out_adj_str = ", ".join(formatted_out_adj)
        if len(args_out) > 1:
            arg_str = ", ".join([var_str, out_str, adj_str, out_adj_str])
        else:
            arg_str = ", ".join([var_str, adj_str, out_adj_str])
    else:
        arg_str = ", ".join(formatted_var + formatted_out + formatted_var_adj + formatted_out_adj)
    return arg_str


class Adjoint:
    # Source code transformer, this class takes a Python function and
    # generates forward and backward SSA forms of the function instructions

    args: List[Var]
    builder: ModuleBuilder

    blocks: List[Block]
    label_count: int

    def __init__(
            self,
            func,
            overload_annotations=None,
            is_user_function=False,
            skip_forward_codegen=False,
            skip_reverse_codegen=False,
            custom_reverse_mode=False,
            custom_reverse_num_input_args=-1,
            transformers=None,
    ):
        self.indentation = None
        self.loop_blocks = None
        self.loop_symbols = None
        self.return_var = None
        self.variables = None
        if transformers is None:
            transformers = []
        self.func = func

        self.is_user_function = is_user_function

        # whether the generation of the forward code is skipped for this function
        self.skip_forward_codegen = skip_forward_codegen
        # whether the generation of the adjoint code is skipped for this function
        self.skip_reverse_codegen = skip_reverse_codegen

        # build AST from function object
        self.source = inspect.getsource(func)

        # get source code lines and line number where function starts
        self.raw_source, self.fun_lineno = inspect.getsourcelines(func)

        # keep track of line number in function code
        self.lineno = None

        # ensures that indented class methods can be parsed as kernels
        self.source = textwrap.dedent(self.source)

        # extract name of source file
        self.filename = inspect.getsourcefile(func) or "unknown source file"

        # build AST and apply node transformers
        self.tree = ast.parse(self.source)
        self.transformers = transformers
        for transformer in transformers:
            self.tree = transformer.visit(self.tree)

        self.fun_name = self.tree.body[0].name

        # whether the forward code shall be used for the reverse pass and a custom
        # function signature is applied to the reverse version of the function
        self.custom_reverse_mode = custom_reverse_mode
        # the number of function arguments that pertain to the forward function
        # input arguments (i.e. the number of arguments that are not adjoint arguments)
        self.custom_reverse_num_input_args = custom_reverse_num_input_args

        # parse argument types
        argspec = inspect.getfullargspec(func)

        # ensure all arguments are annotated
        if overload_annotations is None:
            # use source-level argument annotations
            if len(argspec.annotations) < len(argspec.args):
                raise WarpCodegenError(f"Incomplete argument annotations on function {self.fun_name}")
            self.arg_types = argspec.annotations
        else:
            # use overload argument annotations
            for arg_name in argspec.args:
                if arg_name not in overload_annotations:
                    raise WarpCodegenError(f"Incomplete overload annotations for function {self.fun_name}")
            self.arg_types = overload_annotations.copy()

        self.args = []
        self.symbols = {}

        for name, type in self.arg_types.items():
            # skip return hint
            if name == "return":
                continue

            # add variable for argument
            arg = Var(name, type, False)
            self.args.append(arg)

            # pre-populate symbol dictionary with function argument names
            # this is to avoid registering false references to overshadowed modules
            self.symbols[name] = arg

        # There are cases where a same module might be rebuilt multiple times,
        # for example when kernels are nested inside of functions, or when
        # a kernel's launch raises an exception. Ideally we'd always want to
        # avoid rebuilding kernels but some corner cases seem to depend on it,
        # so we only avoid rebuilding kernels that errored out to give a chance
        # for unit testing errors being spit out from kernels.
        self.skip_build = False

    # generate function ssa form and adjoint
    def build(self, builder: ModuleBuilder):
        if self.skip_build:
            return

        self.builder = builder

        self.symbols = {}  # map from symbols to adjoint variables
        self.variables = []  # list of local variables (in order)

        self.return_var = None  # return type for function or kernel
        self.loop_symbols = []  # symbols at the start of each loop

        # blocks
        self.blocks = [Block()]
        self.loop_blocks = []

        # holds current indent level
        self.indentation = ""

        # used to generate new label indices
        self.label_count = 0

        # update symbol map for each argument
        for a in self.args:
            self.symbols[a.label] = a

        # recursively evaluate function body
        try:
            self.eval(self.tree.body[0])
        except Exception as e:
            try:
                if isinstance(e, KeyError) and getattr(e.args[0], "__module__", None) == "ast":
                    msg = f'Syntax error: unsupported construct "ast.{e.args[0].__name__}"'
                else:
                    msg = "Error"
                lineno = self.lineno + self.fun_lineno
                line = self.source.splitlines()[self.lineno]
                msg += f' while parsing function "{self.fun_name}" at {self.filename}:{lineno}:\n{line}\n'
                ex, data, traceback = sys.exc_info()
                e = ex(";".join([msg] + [str(a) for a in data.args])).with_traceback(traceback)
            finally:
                self.skip_build = True
                raise e

        if builder is not None:
            for a in self.args:
                if isinstance(a.type, Struct):
                    builder.build_struct_recursive(a.type)
                elif isinstance(a.type, types.array) and isinstance(a.type.dtype, Struct):
                    builder.build_struct_recursive(a.type.dtype)

    def indent(self):
        self.indentation = self.indentation + "    "

    def dedent(self):
        self.indentation = self.indentation[:-4]

    def begin_block(self):
        b = Block()

        # give block a unique id
        b.label = self.label_count
        self.label_count += 1

        self.blocks.append(b)
        return b

    def end_block(self):
        return self.blocks.pop()

    def add_var(self, type=None, constant=None):
        index = len(self.variables)
        name = str(index)

        # allocate new variable
        v = Var(name, type=type, constant=constant)

        self.variables.append(v)

        self.blocks[-1].vars.append(v)

        return v

    # append a statement to the forward pass
    def add_forward(adj, statement, replay=None, skip_replay=False):
        adj.blocks[-1].body_forward.append(adj.indentation + statement)

        if not skip_replay:
            if replay:
                # if custom replay specified then output it
                adj.blocks[-1].body_replay.append(adj.indentation + replay)
            else:
                # by default just replay the original statement
                adj.blocks[-1].body_replay.append(adj.indentation + statement)

    # append a statement to the reverse pass
    def add_reverse(adj, statement):
        adj.blocks[-1].body_reverse.append(adj.indentation + statement)

    def add_constant(adj, n):
        output = adj.add_var(type=type(n), constant=n)
        return output

    def load(adj, var):
        if is_reference(var.type):
            var = adj.add_builtin_call("load", [var])
        return var

    def add_comp(adj, op_strings, left, comps):
        output = adj.add_var(builtins.bool)

        left = adj.load(left)
        s = output.emit() + " = " + ("(" * len(comps)) + left.emit() + " "

        prev_comp = None

        for op, comp in zip(op_strings, comps):
            comp_chainable = op_str_is_chainable(op)
            if comp_chainable and prev_comp:
                # We  restrict chaining to operands of the same type
                if prev_comp.type is comp.type:
                    prev_comp = adj.load(prev_comp)
                    comp = adj.load(comp)
                    s += "&& (" + prev_comp.emit() + " " + op + " " + comp.emit() + ")) "
                else:
                    raise WarpCodegenTypeError(
                        f"Cannot chain comparisons of unequal types: {prev_comp.type} {op} {comp.type}."
                    )
            else:
                comp = adj.load(comp)
                s += op + " " + comp.emit() + ") "

            prev_comp = comp

        s = s.rstrip() + ";"

        adj.add_forward(s)

        return output

    def add_bool_op(adj, op_string, exprs):
        exprs = [adj.load(expr) for expr in exprs]
        output = adj.add_var(builtins.bool)
        command = output.emit() + " = " + (" " + op_string + " ").join([expr.emit() for expr in exprs]) + ";"
        adj.add_forward(command)

        return output

    def resolve_func(adj, func, args, min_outputs, templates, kwds):
        arg_types = [strip_reference(a.type) for a in args if not isinstance(a, Function)]

        if not func.is_builtin():
            # user-defined function
            overload = func.get_overload(arg_types)
            if overload is not None:
                return overload
        else:
            # if func is overloaded then perform overload resolution here
            # we validate argument types before they go to generated native code
            for f in func.overloads:
                # skip type checking for variadic functions
                if not f.variadic:
                    # check argument counts match are compatible (may be some default args)
                    if len(f.input_types) < len(args):
                        continue

                    def match_args(args, f):
                        # check argument types equal
                        for i, (arg_name, arg_type) in enumerate(f.input_types.items()):
                            # if arg type registered as Any, treat as
                            # template allowing any type to match
                            if arg_type == Any:
                                continue

                            # handle function refs as a special case
                            if arg_type == Callable and type(args[i]) is Function:
                                continue

                            if arg_type == Reference and is_reference(args[i].type):
                                continue

                            # look for default values for missing args
                            if i >= len(args):
                                if arg_name not in f.defaults:
                                    return False
                            else:
                                # otherwise check arg type matches input variable type
                                if not types_equal(arg_type, strip_reference(args[i].type), match_generic=True):
                                    return False

                        return True

                    if not match_args(args, f):
                        continue

                # check output dimensions match expectations
                if min_outputs:
                    try:
                        value_type = f.value_func(args, kwds, templates)
                        if not hasattr(value_type, "__len__") or len(value_type) != min_outputs:
                            continue
                    except Exception:
                        # value func may fail if the user has given
                        # incorrect args, so we need to catch this
                        continue

                # found a match, use it
                return f

        # unresolved function, report error
        arg_types = []

        for x in args:
            if isinstance(x, Var):
                # shorten Warp primitive type names
                if isinstance(x.type, list):
                    if len(x.type) != 1:
                        raise WarpCodegenError("Argument must not be the result from a multi-valued function")
                    arg_type = x.type[0]
                else:
                    arg_type = x.type

                arg_types.append(type_repr(arg_type))

            if isinstance(x, Function):
                arg_types.append("function")

        raise WarpCodegenError(
            f"Couldn't find function overload for '{func.key}' that matched inputs with types: [{', '.join(arg_types)}]"
        )

    def add_call(adj, func, args, min_outputs=None, templates=[], kwds=None):
        func = adj.resolve_func(func, args, min_outputs, templates, kwds)

        # push any default values onto args
        for i, (arg_name, arg_type) in enumerate(func.input_types.items()):
            if i >= len(args):
                if arg_name in func.defaults:
                    const = adj.add_constant(func.defaults[arg_name])
                    args.append(const)
                else:
                    break

        # if it is a user-function then build it recursively
        if not func.is_builtin():
            adj.builder.build_function(func)

        # evaluate the function type based on inputs
        arg_types = [strip_reference(a.type) for a in args if not isinstance(a, Function)]
        return_type = func.value_func(arg_types, kwds, templates)

        func_name = compute_type_str(func.native_func, templates)
        param_types = list(func.input_types.values())

        use_initializer_list = func.initializer_list_func(args, templates)

        args_var = [
            adj.load(a)
            if not ((param_types[i] == Reference or param_types[i] == Callable) if i < len(param_types) else False)
            else a
            for i, a in enumerate(args)
        ]

        if return_type is None:
            # handles expression (zero output) functions, e.g.: void do_something();

            output = None
            output_list = []

            forward_call = (
                f"{func.namespace}{func_name}({adj.format_forward_call_args(args_var, use_initializer_list)});"
            )
            replay_call = forward_call
            if func.custom_replay_func is not None:
                replay_call = f"{func.namespace}replay_{func_name}({adj.format_forward_call_args(args_var, use_initializer_list)});"

        elif not isinstance(return_type, list) or len(return_type) == 1:
            # handle simple function (one output)

            if isinstance(return_type, list):
                return_type = return_type[0]
            output = adj.add_var(return_type)
            output_list = [output]

            forward_call = f"var_{output} = {func.namespace}{func_name}({adj.format_forward_call_args(args_var, use_initializer_list)});"
            replay_call = forward_call
            if func.custom_replay_func is not None:
                replay_call = f"var_{output} = {func.namespace}replay_{func_name}({adj.format_forward_call_args(args_var, use_initializer_list)});"

        else:
            # handle multiple value functions

            output = [adj.add_var(v) for v in return_type]
            output_list = output

            forward_call = (
                f"{func.namespace}{func_name}({adj.format_forward_call_args(args_var + output, use_initializer_list)});"
            )
            replay_call = forward_call

        if func.skip_replay:
            adj.add_forward(forward_call, replay="// " + replay_call)
        else:
            adj.add_forward(forward_call, replay=replay_call)

        if not func.missing_grad and len(args):
            reverse_has_output_args = len(output_list) > 1 and func.custom_grad_func is None
            arg_str = adj.format_reverse_call_args(
                args_var,
                args,
                output_list,
                use_initializer_list,
                has_output_args=reverse_has_output_args,
            )
            if arg_str is not None:
                reverse_call = f"{func.namespace}adj_{func.native_func}({arg_str});"
                adj.add_reverse(reverse_call)

        return output

    def add_builtin_call(adj, func_name, args, min_outputs=None, templates=None, kwds=None):
        if templates is None:
            templates = []
        func = builtin_functions[func_name]
        return adj.add_call(func, args, min_outputs, templates, kwds)

    def add_return(adj, var):
        if var is None or len(var) == 0:
            adj.add_forward("return;", f"goto label{adj.label_count};")
        elif len(var) == 1:
            adj.add_forward(f"return {var[0].emit()};", f"goto label{adj.label_count};")
            adj.add_reverse("adj_" + str(var[0]) + " += adj_ret;")
        else:
            for i, v in enumerate(var):
                adj.add_forward(f"ret_{i} = {v.emit()};")
                adj.add_reverse(f"adj_{v} += adj_ret_{i};")
            adj.add_forward("return;", f"goto label{adj.label_count};")

        adj.add_reverse(f"label{adj.label_count}:;")

        adj.label_count += 1

    # define an if statement
    def begin_if(adj, cond):
        cond = adj.load(cond)
        adj.add_forward(f"if ({cond.emit()}) {{")
        adj.add_reverse("}")

        adj.indent()

    def end_if(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        cond = adj.load(cond)
        adj.add_reverse(f"if ({cond.emit()}) {{")

    def begin_else(adj, cond):
        cond = adj.load(cond)
        adj.add_forward(f"if (!{cond.emit()}) {{")
        adj.add_reverse("}")

        adj.indent()

    def end_else(adj, cond):
        adj.dedent()

        adj.add_forward("}")
        cond = adj.load(cond)
        adj.add_reverse(f"if (!{cond.emit()}) {{")

    # define a for-loop
    def begin_for(adj, iter):
        cond_block = adj.begin_block()
        adj.loop_blocks.append(cond_block)
        adj.add_forward(f"for_start_{cond_block.label}:;")
        adj.indent()

        # evaluate cond
        adj.add_forward(f"if (iter_cmp({iter.emit()}) == 0) goto for_end_{cond_block.label};")

        # evaluate iter
        val = adj.add_builtin_call("iter_next", [iter])

        adj.begin_block()

        return val

    def end_for(adj, iter):
        body_block = adj.end_block()
        cond_block = adj.end_block()
        adj.loop_blocks.pop()

        ####################
        # forward pass

        for i in cond_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        for i in body_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        adj.add_forward(f"goto for_start_{cond_block.label};", skip_replay=True)

        adj.dedent()
        adj.add_forward(f"for_end_{cond_block.label}:;", skip_replay=True)

        ####################
        # reverse pass

        reverse = []

        # reverse iterator
        reverse.append(adj.indentation + f"{iter.emit()} = wp::iter_reverse({iter.emit()});")

        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints
        for i in body_block.vars:
            reverse.append(adj.indentation + f"\t{i.emit_adj()} = {{}};")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(adj.indentation + f"\tgoto for_start_{cond_block.label};")
        reverse.append(adj.indentation + f"for_end_{cond_block.label}:;")

        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    # define a while loop
    def begin_while(adj, cond):
        # evaluate condition in its own block
        # so we can control replay
        cond_block = adj.begin_block()
        adj.loop_blocks.append(cond_block)
        cond_block.body_forward.append(f"while_start_{cond_block.label}:;")

        c = adj.eval(cond)

        cond_block.body_forward.append(f"if (({c.emit()}) == false) goto while_end_{cond_block.label};")

        # being block around loop
        adj.begin_block()
        adj.indent()

    def end_while(adj):
        adj.dedent()
        body_block = adj.end_block()
        cond_block = adj.end_block()
        adj.loop_blocks.pop()

        ####################
        # forward pass

        for i in cond_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        for i in body_block.body_forward:
            adj.blocks[-1].body_forward.append(i)

        adj.blocks[-1].body_forward.append(f"goto while_start_{cond_block.label};")
        adj.blocks[-1].body_forward.append(f"while_end_{cond_block.label}:;")

        ####################
        # reverse pass
        reverse = []

        # cond
        for i in cond_block.body_forward:
            reverse.append(i)

        # zero adjoints of local vars
        for i in body_block.vars:
            reverse.append(f"{i.emit_adj()} = {{}};")

        # replay
        for i in body_block.body_replay:
            reverse.append(i)

        # reverse
        for i in reversed(body_block.body_reverse):
            reverse.append(i)

        reverse.append(f"goto while_start_{cond_block.label};")
        reverse.append(f"while_end_{cond_block.label}:;")

        # output
        adj.blocks[-1].body_reverse.extend(reversed(reverse))

    def emit_FunctionDef(adj, node):
        for f in node.body:
            adj.eval(f)

        if adj.return_var is not None and len(adj.return_var) == 1:
            if not isinstance(node.body[-1], ast.Return):
                adj.add_forward("return {};", skip_replay=True)

    def emit_If(adj, node):
        if len(node.body) == 0:
            return None

        # eval condition
        cond = adj.eval(node.test)

        # save symbol map
        symbols_prev = adj.symbols.copy()

        # eval body
        adj.begin_if(cond)

        for stmt in node.body:
            adj.eval(stmt)

        adj.end_if(cond)

        # detect existing symbols with conflicting definitions (variables assigned inside the branch)
        # and resolve with a phi (select) function
        for items in symbols_prev.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                # insert a phi function that selects var1, var2 based on cond
                out = adj.add_builtin_call("select", [cond, var1, var2])
                adj.symbols[sym] = out

        symbols_prev = adj.symbols.copy()

        # evaluate 'else' statement as if (!cond)
        if len(node.orelse) > 0:
            adj.begin_else(cond)

            for stmt in node.orelse:
                adj.eval(stmt)

            adj.end_else(cond)

        # detect existing symbols with conflicting definitions (variables assigned inside the else)
        # and resolve with a phi (select) function
        for items in symbols_prev.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                # insert a phi function that selects var1, var2 based on cond
                # note the reversed order of vars since we want to use !cond as our select
                out = adj.add_builtin_call("select", [cond, var2, var1])
                adj.symbols[sym] = out

    def emit_Compare(adj, node):
        # node.left, node.ops (list of ops), node.comparators (things to compare to)
        # e.g. (left ops[0] node.comparators[0]) ops[1] node.comparators[1]

        left = adj.eval(node.left)
        comps = [adj.eval(comp) for comp in node.comparators]
        op_strings = [builtin_operators[type(op)] for op in node.ops]

        return adj.add_comp(op_strings, left, comps)

    def emit_BoolOp(adj, node):
        # op, expr list values

        op = node.op
        if isinstance(op, ast.And):
            func = "&&"
        elif isinstance(op, ast.Or):
            func = "||"
        else:
            raise WarpCodegenKeyError(f"Op {op} is not supported")

        return adj.add_bool_op(func, [adj.eval(expr) for expr in node.values])

    def emit_Name(adj, node):
        # lookup symbol, if it has already been assigned to a variable then return the existing mapping
        if node.id in adj.symbols:
            return adj.symbols[node.id]

        # try and resolve the name using the function's globals context (used to lookup constants + functions)
        obj = adj.func.__globals__.get(node.id)

        if obj is None:
            # Lookup constant in captured contents
            capturedvars = dict(
                zip(adj.func.__code__.co_freevars, [c.cell_contents for c in (adj.func.__closure__ or [])])
            )
            obj = capturedvars.get(str(node.id), None)

        if obj is None:
            raise WarpCodegenKeyError("Referencing undefined symbol: " + str(node.id))

        if types.is_value(obj):
            # evaluate constant
            out = adj.add_constant(obj)
            adj.symbols[node.id] = out
            return out

        # the named object is either a function, class name, or module
        # pass it back to the caller for processing
        return obj

    @staticmethod
    def resolve_type_attribute(var_type: type, attr: str):
        if isinstance(var_type, type) and type_is_value(var_type):
            if attr == "dtype":
                return type_scalar_type(var_type)
            elif attr == "length":
                return type_length(var_type)

        return getattr(var_type, attr, None)

    def vector_component_index(adj, component, vector_type):
        if len(component) != 1:
            raise WarpCodegenAttributeError(f"Vector swizzle must be single character, got .{component}")

        dim = vector_type._shape_[0]
        swizzles = "xyzw"[0:dim]
        if component not in swizzles:
            raise WarpCodegenAttributeError(
                f"Vector swizzle for {vector_type} must be one of {swizzles}, got {component}"
            )

        index = swizzles.index(component)
        index = adj.add_constant(index)
        return index

    def emit_Attribute(adj, node):
        if hasattr(node, "is_adjoint"):
            node.value.is_adjoint = True

        aggregate = adj.eval(node.value)

        try:
            if isinstance(aggregate, types.ModuleType) or isinstance(aggregate, type):
                out = getattr(aggregate, node.attr)

                if types.is_value(out):
                    return adj.add_constant(out)

                return out

            if hasattr(node, "is_adjoint"):
                # create a Var that points to the struct attribute, i.e.: directly generates `struct.attr` when used
                attr_name = aggregate.label + "." + node.attr
                attr_type = aggregate.type.vars[node.attr].type

                return Var(attr_name, attr_type)

            aggregate_type = strip_reference(aggregate.type)

            # reading a vector component
            if type_is_vector(aggregate_type):
                index = adj.vector_component_index(node.attr, aggregate_type)

                return adj.add_builtin_call("extract", [aggregate, index])

            else:
                attr_type = Reference(aggregate_type.vars[node.attr].type)
                attr = adj.add_var(attr_type)

                if is_reference(aggregate.type):
                    adj.add_forward(f"{attr.emit()} = &({aggregate.emit()}->{node.attr});")
                    adj.add_reverse(f"{aggregate.emit_adj()}.{node.attr} = {attr.emit_adj()};")
                else:
                    adj.add_forward(f"{attr.emit()} = &({aggregate.emit()}.{node.attr});")
                    adj.add_reverse(f"{aggregate.emit_adj()}.{node.attr} = {attr.emit_adj()};")

                return attr

        except (KeyError, AttributeError):
            # Try resolving as type attribute
            aggregate_type = strip_reference(aggregate.type) if isinstance(aggregate, Var) else aggregate

            type_attribute = adj.resolve_type_attribute(aggregate_type, node.attr)
            if type_attribute is not None:
                return type_attribute

            if isinstance(aggregate, Var):
                raise WarpCodegenAttributeError(
                    f"Error, `{node.attr}` is not an attribute of '{aggregate.label}' ({type_repr(aggregate.type)})"
                )
            raise WarpCodegenAttributeError(f"Error, `{node.attr}` is not an attribute of '{aggregate}'")

    def emit_String(adj, node):
        # string constant
        return adj.add_constant(node.s)

    def emit_Num(adj, node):
        # lookup constant, if it has already been assigned then return existing var
        key = (node.n, type(node.n))

        if key in adj.symbols:
            return adj.symbols[key]
        else:
            out = adj.add_constant(node.n)
            adj.symbols[key] = out
            return out

    def emit_Ellipsis(adj, node):
        # stubbed @wp.native_func
        return

    def emit_NameConstant(adj, node):
        if node.value is True:
            return adj.add_constant(True)
        elif node.value is False:
            return adj.add_constant(False)
        elif node.value is None:
            raise WarpCodegenTypeError("None type unsupported")

    def emit_Constant(adj, node):
        if isinstance(node, ast.Str):
            return adj.emit_String(node)
        elif isinstance(node, ast.Num):
            return adj.emit_Num(node)
        elif isinstance(node, ast.Ellipsis):
            return adj.emit_Ellipsis(node)
        else:
            assert isinstance(node, ast.NameConstant)
            return adj.emit_NameConstant(node)

    def emit_BinOp(adj, node):
        # evaluate binary operator arguments
        left = adj.eval(node.left)
        right = adj.eval(node.right)

        name = builtin_operators[type(node.op)]

        return adj.add_builtin_call(name, [left, right])

    def emit_UnaryOp(adj, node):
        # evaluate unary op arguments
        arg = adj.eval(node.operand)

        name = builtin_operators[type(node.op)]

        return adj.add_builtin_call(name, [arg])

    def materialize_redefinitions(adj, symbols):
        # detect symbols with conflicting definitions (assigned inside the for loop)
        for items in symbols.items():
            sym = items[0]
            var1 = items[1]
            var2 = adj.symbols[sym]

            if var1 != var2:
                if config.verbose and not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source.splitlines()[adj.lineno]
                    msg = f'Warning: detected mutated variable {sym} during a dynamic for-loop in function "{adj.fun_name}" at {adj.filename}:{lineno}: this may not be a differentiable operation.\n{line}\n'
                    print(msg)

                if var1.constant is not None:
                    raise WarpCodegenError(
                        f"Error mutating a constant {sym} inside a dynamic loop, use the following syntax: pi = float(3.141) to declare a dynamic variable"
                    )

                # overwrite the old variable value (violates SSA)
                adj.add_builtin_call("assign", [var1, var2])

                # reset the symbol to point to the original variable
                adj.symbols[sym] = var1

    def emit_While(adj, node):
        adj.begin_while(node.test)

        adj.loop_symbols.append(adj.symbols.copy())

        # eval body
        for s in node.body:
            adj.eval(s)

        adj.materialize_redefinitions(adj.loop_symbols[-1])
        adj.loop_symbols.pop()

        adj.end_while()

    def eval_num(adj, a):
        if isinstance(a, ast.Num):
            return True, a.n
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub) and isinstance(a.operand, ast.Num):
            return True, -a.operand.n

        # try and resolve the expression to an object
        # e.g.: wp.constant in the globals scope
        obj, path = adj.resolve_static_expression(a)
        return types.is_int(obj), obj

    # detects whether a loop contains a break (or continue) statement
    def contains_break(adj, body):
        for s in body:
            if isinstance(s, ast.Break):
                return True
            elif isinstance(s, ast.Continue):
                return True
            elif isinstance(s, ast.If):
                if adj.contains_break(s.body):
                    return True
                if adj.contains_break(s.orelse):
                    return True
            else:
                # note that nested for or while loops containing a break statement
                # do not affect the current loop
                pass

        return False

    # returns a constant range() if unrollable, otherwise None
    def get_unroll_range(adj, loop):
        if (
                not isinstance(loop.iter, ast.Call)
                or not isinstance(loop.iter.func, ast.Name)
                or loop.iter.func.id != "range"
                or len(loop.iter.args) == 0
                or len(loop.iter.args) > 3
        ):
            return None

        # if all range() arguments are numeric constants we will unroll
        # note that this only handles trivial constants, it will not unroll
        # constant compile-time expressions e.g.: range(0, 3*2)

        # Evaluate the arguments and check that they are numeric constants
        # It is important to do that in one pass, so that if evaluating these arguments have side effects
        # the code does not get generated more than once
        range_args = [adj.eval_num(arg) for arg in loop.iter.args]
        arg_is_numeric, arg_values = zip(*range_args)

        if all(arg_is_numeric):
            # All argument are numeric constants

            # range(end)
            if len(loop.iter.args) == 1:
                start = 0
                end = arg_values[0]
                step = 1

            # range(start, end)
            elif len(loop.iter.args) == 2:
                start = arg_values[0]
                end = arg_values[1]
                step = 1

            # range(start, end, step)
            elif len(loop.iter.args) == 3:
                start = arg_values[0]
                end = arg_values[1]
                step = arg_values[2]

            # test if we're above max unroll count
            max_iters = abs(end - start) // abs(step)
            max_unroll = adj.builder.options["max_unroll"]

            ok_to_unroll = True

            if max_iters > max_unroll:
                if config.verbose:
                    print(
                        f"Warning: fixed-size loop count of {max_iters} is larger than the module 'max_unroll' limit of {max_unroll}, will generate dynamic loop."
                    )
                ok_to_unroll = False

            elif adj.contains_break(loop.body):
                if config.verbose:
                    print("Warning: 'break' or 'continue' found in loop body, will generate dynamic loop.")
                ok_to_unroll = False

            if ok_to_unroll:
                return range(start, end, step)

        # Unroll is not possible, range needs to be valuated dynamically
        range_call = adj.add_builtin_call(
            "range",
            [adj.add_constant(val) if is_numeric else val for is_numeric, val in range_args],
        )
        return range_call

    def emit_For(adj, node):
        # try and unroll simple range() statements that use constant args
        unroll_range = adj.get_unroll_range(node)

        if isinstance(unroll_range, range):
            for i in unroll_range:
                const_iter = adj.add_constant(i)
                var_iter = adj.add_builtin_call("int", [const_iter])
                adj.symbols[node.target.id] = var_iter

                # eval body
                for s in node.body:
                    adj.eval(s)

        # otherwise generate a dynamic loop
        else:
            # evaluate the Iterable -- only if not previously evaluated when trying to unroll
            if unroll_range is not None:
                # Range has already been evaluated when trying to unroll, do not re-evaluate
                iter = unroll_range
            else:
                iter = adj.eval(node.iter)

            adj.symbols[node.target.id] = adj.begin_for(iter)

            # for loops should be side-effect free, here we store a copy
            adj.loop_symbols.append(adj.symbols.copy())

            # eval body
            for s in node.body:
                adj.eval(s)

            adj.materialize_redefinitions(adj.loop_symbols[-1])
            adj.loop_symbols.pop()

            adj.end_for(iter)

    def emit_Break(adj, node):
        adj.materialize_redefinitions(adj.loop_symbols[-1])

        adj.add_forward(f"goto for_end_{adj.loop_blocks[-1].label};")

    def emit_Continue(adj, node):
        adj.materialize_redefinitions(adj.loop_symbols[-1])

        adj.add_forward(f"goto for_start_{adj.loop_blocks[-1].label};")

    def emit_Expr(adj, node):
        return adj.eval(node.value)

    def check_tid_in_func_error(adj, node):
        if adj.is_user_function:
            if hasattr(node.func, "attr") and node.func.attr == "tid":
                lineno = adj.lineno + adj.fun_lineno
                line = adj.source.splitlines()[adj.lineno]
                raise WarpCodegenError(
                    "tid() may only be called from a Warp kernel, not a Warp function. "
                    "Instead, obtain the indices from a @wp.kernel and pass them as "
                    f"arguments to the function {adj.fun_name}, {adj.filename}:{lineno}:\n{line}\n"
                )

    def emit_Call(adj, node):
        adj.check_tid_in_func_error(node)

        # try and lookup function in globals by
        # resolving path (e.g.: module.submodule.attr)
        func, path = adj.resolve_static_expression(node.func)
        templates = []

        if not isinstance(func, Function):
            if len(path) == 0:
                raise WarpCodegenError(f"Unrecognized syntax for function call, path not valid: '{node.func}'")

            attr = path[-1]
            caller = func
            func = None

            # try and lookup function name in builtins (e.g.: using `dot` directly without wp prefix)
            if attr in builtin_functions:
                func = builtin_functions[attr]

            # vector class type e.g.: wp.vec3f constructor
            if func is None and hasattr(caller, "_wp_generic_type_str_"):
                templates = caller._wp_type_params_
                func = builtin_functions.get(caller._wp_constructor_)

            # scalar class type e.g.: wp.int8 constructor
            if func is None and hasattr(caller, "__name__") and caller.__name__ in builtin_functions:
                func = builtin_functions.get(caller.__name__)

            # struct constructor
            if func is None and isinstance(caller, Struct):
                adj.builder.build_struct_recursive(caller)
                func = caller.initializer()

            if func is None:
                raise WarpCodegenError(
                    f"Could not find function {'.'.join(path)} as a built-in or user-defined function. Note that user functions must be annotated with a @wp.func decorator to be called from a kernel."
                )

        args = []

        # eval all arguments
        for arg in node.args:
            var = adj.eval(arg)
            args.append(var)

        # eval all keyword ags
        def kwval(kw):
            if isinstance(kw.value, ast.Num):
                return kw.value.n
            elif isinstance(kw.value, ast.Tuple):
                arg_is_numeric, arg_values = zip(*(adj.eval_num(e) for e in kw.value.elts))
                if not all(arg_is_numeric):
                    raise WarpCodegenError(
                        f"All elements of the tuple keyword argument '{kw.name}' must be numeric constants, got '{arg_values}'"
                    )
                return arg_values
            else:
                return adj.resolve_static_expression(kw.value)[0]

        kwds = {kw.arg: kwval(kw) for kw in node.keywords}

        # get expected return count, e.g.: for multi-assignment
        min_outputs = None
        if hasattr(node, "expects"):
            min_outputs = node.expects

        # add var with value type from the function
        out = adj.add_call(func=func, args=args, kwds=kwds, templates=templates, min_outputs=min_outputs)
        return out

    def emit_Index(adj, node):
        # the ast.Index node appears in 3.7 versions
        # when performing array slices, e.g.: x = arr[i]
        # but in version 3.8 and higher it does not appear

        if hasattr(node, "is_adjoint"):
            node.value.is_adjoint = True

        return adj.eval(node.value)

    def emit_Subscript(adj, node):
        if hasattr(node.value, "attr") and node.value.attr == "adjoint":
            # handle adjoint of a variable, i.e. wp.adjoint[var]
            node.slice.is_adjoint = True
            var = adj.eval(node.slice)
            var_name = var.label
            var = Var(f"adj_{var_name}", type=var.type, constant=None, prefix=False)
            return var

        target = adj.eval(node.value)

        indices = []

        if isinstance(node.slice, ast.Tuple):
            # handles the x[i,j] case (Python 3.8.x upward)
            for arg in node.slice.elts:
                var = adj.eval(arg)
                indices.append(var)

        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple):
            # handles the x[i,j] case (Python 3.7.x)
            for arg in node.slice.value.elts:
                var = adj.eval(arg)
                indices.append(var)
        else:
            # simple expression, e.g.: x[i]
            var = adj.eval(node.slice)
            indices.append(var)

        target_type = strip_reference(target.type)
        if is_array(target_type):
            if len(indices) == target_type.ndim:
                # handles array loads (where each dimension has an index specified)
                out = adj.add_builtin_call("address", [target, *indices])
            else:
                # handles array views (fewer indices than dimensions)
                out = adj.add_builtin_call("view", [target, *indices])

        else:
            # handles non-array type indexing, e.g: vec3, mat33, etc
            out = adj.add_builtin_call("extract", [target, *indices])

        return out

    def emit_Assign(adj, node):
        if len(node.targets) != 1:
            raise WarpCodegenError("Assigning the same value to multiple variables is not supported")

        lhs = node.targets[0]

        # handle the case where we are assigning multiple output variables
        if isinstance(lhs, ast.Tuple):
            # record the expected number of outputs on the node
            # we do this so we can decide which function to
            # call based on the number of expected outputs
            if isinstance(node.value, ast.Call):
                node.value.expects = len(lhs.elts)

            # evaluate values
            if isinstance(node.value, ast.Tuple):
                out = [adj.eval(v) for v in node.value.elts]
            else:
                out = adj.eval(node.value)

            names = []
            for v in lhs.elts:
                if isinstance(v, ast.Name):
                    names.append(v.id)
                else:
                    raise WarpCodegenError(
                        "Multiple return functions can only assign to simple variables, e.g.: x, y = func()"
                    )

            if len(names) != len(out):
                raise WarpCodegenError(
                    f"Multiple return functions need to receive all their output values, incorrect number of values to unpack (expected {len(out)}, got {len(names)})"
                )

            for name, rhs in zip(names, out):
                if name in adj.symbols:
                    if not types_equal(rhs.type, adj.symbols[name].type):
                        raise WarpCodegenTypeError(
                            f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                        )

                adj.symbols[name] = rhs

        # handles the case where we are assigning to an array index (e.g.: arr[i] = 2.0)
        elif isinstance(lhs, ast.Subscript):
            if hasattr(lhs.value, "attr") and lhs.value.attr == "adjoint":
                # handle adjoint of a variable, i.e. wp.adjoint[var]
                lhs.slice.is_adjoint = True
                src_var = adj.eval(lhs.slice)
                var = Var(f"adj_{src_var.label}", type=src_var.type, constant=None, prefix=False)
                value = adj.eval(node.value)
                adj.add_forward(f"{var.emit()} = {value.emit()};")
                return

            target = adj.eval(lhs.value)
            value = adj.eval(node.value)

            slice = lhs.slice
            indices = []

            if isinstance(slice, ast.Tuple):
                # handles the x[i, j] case (Python 3.8.x upward)
                for arg in slice.elts:
                    var = adj.eval(arg)
                    indices.append(var)
            elif isinstance(slice, ast.Index) and isinstance(slice.value, ast.Tuple):
                # handles the x[i, j] case (Python 3.7.x)
                for arg in slice.value.elts:
                    var = adj.eval(arg)
                    indices.append(var)
            else:
                # simple expression, e.g.: x[i]
                var = adj.eval(slice)
                indices.append(var)

            target_type = strip_reference(target.type)

            if is_array(target_type):
                adj.add_builtin_call("array_store", [target, *indices, value])

            elif type_is_vector(target_type) or type_is_matrix(target_type):
                if is_reference(target.type):
                    attr = adj.add_builtin_call("indexref", [target, *indices])
                else:
                    attr = adj.add_builtin_call("index", [target, *indices])

                adj.add_builtin_call("store", [attr, value])

                if config.verbose and not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source.splitlines()[adj.lineno]
                    node_source = adj.get_node_source(lhs.value)
                    print(
                        f"Warning: mutating {node_source} in function {adj.fun_name} at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}\n"
                    )

            else:
                raise WarpCodegenError("Can only subscript assign array, vector, and matrix types")

        elif isinstance(lhs, ast.Name):
            # symbol name
            name = lhs.id

            # evaluate rhs
            rhs = adj.eval(node.value)

            # check type matches if symbol already defined
            if name in adj.symbols:
                if not types_equal(strip_reference(rhs.type), adj.symbols[name].type):
                    raise WarpCodegenTypeError(
                        f"Error, assigning to existing symbol {name} ({adj.symbols[name].type}) with different type ({rhs.type})"
                    )

            # handle simple assignment case (a = b), where we generate a value copy rather than reference
            if isinstance(node.value, ast.Name) or is_reference(rhs.type):
                out = adj.add_builtin_call("copy", [rhs])
            else:
                out = rhs

            # update symbol map (assumes lhs is a Name node)
            adj.symbols[name] = out

        elif isinstance(lhs, ast.Attribute):
            rhs = adj.eval(node.value)
            aggregate = adj.eval(lhs.value)
            aggregate_type = strip_reference(aggregate.type)

            # assigning to a vector component
            if type_is_vector(aggregate_type):
                index = adj.vector_component_index(lhs.attr, aggregate_type)

                if is_reference(aggregate.type):
                    attr = adj.add_builtin_call("indexref", [aggregate, index])
                else:
                    attr = adj.add_builtin_call("index", [aggregate, index])

                adj.add_builtin_call("store", [attr, rhs])

            else:
                attr = adj.emit_Attribute(lhs)
                if is_reference(attr.type):
                    adj.add_builtin_call("store", [attr, rhs])
                else:
                    adj.add_builtin_call("assign", [attr, rhs])

                if config.verbose and not adj.custom_reverse_mode:
                    lineno = adj.lineno + adj.fun_lineno
                    line = adj.source.splitlines()[adj.lineno]
                    msg = f'Warning: detected mutated struct {attr.label} during function "{adj.fun_name}" at {adj.filename}:{lineno}: this is a non-differentiable operation.\n{line}\n'
                    print(msg)

        else:
            raise WarpCodegenError("Error, unsupported assignment statement.")

    def emit_Return(adj, node):
        if node.value is None:
            var = None
        elif isinstance(node.value, ast.Tuple):
            var = tuple(adj.eval(arg) for arg in node.value.elts)
        else:
            var = (adj.eval(node.value),)

        if adj.return_var is not None:
            old_ctypes = tuple(v.ctype(value_type=True) for v in adj.return_var)
            new_ctypes = tuple(v.ctype(value_type=True) for v in var)
            if old_ctypes != new_ctypes:
                raise WarpCodegenTypeError(
                    f"Error, function returned different types, previous: [{', '.join(old_ctypes)}], new [{', '.join(new_ctypes)}]"
                )

        if var is not None:
            adj.return_var = tuple()
            for ret in var:
                ret = adj.load(ret)
                adj.return_var += (ret,)

        adj.add_return(adj.return_var)

    def emit_AugAssign(adj, node):
        # replace augmented assignment with assignment statement + binary op
        new_node = ast.Assign(targets=[node.target], value=ast.BinOp(node.target, node.op, node.value))
        adj.eval(new_node)

    def emit_Tuple(adj, node):
        # LHS for expressions, such as i, j, k = 1, 2, 3
        for elem in node.elts:
            adj.eval(elem)

    def emit_Pass(adj, node):
        pass

    node_visitors = {
        ast.FunctionDef: emit_FunctionDef,
        ast.If: emit_If,
        ast.Compare: emit_Compare,
        ast.BoolOp: emit_BoolOp,
        ast.Name: emit_Name,
        ast.Attribute: emit_Attribute,
        ast.Str: emit_String,  # Deprecated in 3.8; use Constant
        ast.Num: emit_Num,  # Deprecated in 3.8; use Constant
        ast.NameConstant: emit_NameConstant,  # Deprecated in 3.8; use Constant
        ast.Constant: emit_Constant,
        ast.BinOp: emit_BinOp,
        ast.UnaryOp: emit_UnaryOp,
        ast.While: emit_While,
        ast.For: emit_For,
        ast.Break: emit_Break,
        ast.Continue: emit_Continue,
        ast.Expr: emit_Expr,
        ast.Call: emit_Call,
        ast.Index: emit_Index,  # Deprecated in 3.8; Use the index value directly instead.
        ast.Subscript: emit_Subscript,
        ast.Assign: emit_Assign,
        ast.Return: emit_Return,
        ast.AugAssign: emit_AugAssign,
        ast.Tuple: emit_Tuple,
        ast.Pass: emit_Pass,
        ast.Ellipsis: emit_Ellipsis
    }

    def eval(adj, node):
        if hasattr(node, "lineno"):
            adj.set_lineno(node.lineno - 1)

        emit_node = adj.node_visitors[type(node)]

        return emit_node(adj, node)

    # helper to evaluate expressions of the form
    # obj1.obj2.obj3.attr in the function's global scope
    def resolve_path(adj, path):
        if len(path) == 0:
            return None

        # if root is overshadowed by local symbols, bail out
        if path[0] in adj.symbols:
            return None

        if path[0] in __builtins__:
            return __builtins__[path[0]]

        # Look up the closure info and append it to adj.func.__globals__
        # in case you want to define a kernel inside a function and refer
        # to variables you've declared inside that function:
        extract_contents = (
            lambda contents: contents
            if isinstance(contents, Function) or not callable(contents)
            else contents
        )
        capturedvars = dict(
            zip(
                adj.func.__code__.co_freevars,
                [extract_contents(c.cell_contents) for c in (adj.func.__closure__ or [])],
            )
        )
        vars_dict = {**adj.func.__globals__, **capturedvars}

        if path[0] in vars_dict:
            func = vars_dict[path[0]]

        # Support Warp types in kernels without the module suffix (e.g. v = vec3(0.0,0.2,0.4)):
        else:
            func = getattr(warp, path[0], None)

        if func:
            for i in range(1, len(path)):
                if hasattr(func, path[i]):
                    func = getattr(func, path[i])

        return func

    # Evaluates a static expression that does not depend on runtime values
    # if eval_types is True, try resolving the path using evaluated type information as well
    def resolve_static_expression(adj, root_node, eval_types=True):
        attributes = []

        node = root_node
        while isinstance(node, ast.Attribute):
            attributes.append(node.attr)
            node = node.value

        if eval_types and isinstance(node, ast.Call):
            # support for operators returning modules
            # i.e. operator_name(*operator_args).x.y.z
            operator_args = node.args
            operator_name = getattr(node.func, "id", None)

            if operator_name is None:
                raise WarpCodegenError(
                    f"Invalid operator call syntax, expected a plain name, got {ast.dump(node.func, annotate_fields=False)}"
                )

            if operator_name == "type":
                if len(operator_args) != 1:
                    raise WarpCodegenError(f"type() operator expects exactly one argument, got {len(operator_args)}")

                # type() operator
                var = adj.eval(operator_args[0])

                if isinstance(var, Var):
                    var_type = strip_reference(var.type)
                    # Allow accessing type attributes, for instance array.dtype
                    while attributes:
                        attr_name = attributes.pop()
                        var_type, prev_type = adj.resolve_type_attribute(var_type, attr_name), var_type

                        if var_type is None:
                            raise WarpCodegenAttributeError(
                                f"{attr_name} is not an attribute of {type_repr(prev_type)}"
                            )

                    return var_type, [type_repr(var_type)]
                else:
                    raise WarpCodegenError(f"Cannot deduce the type of {var}")

            raise WarpCodegenError(f"Unknown operator '{operator_name}'")

        # reverse list since ast presents it backward order
        path = [*reversed(attributes)]
        if isinstance(node, ast.Name):
            path.insert(0, node.id)

        # Try resolving path from captured context
        captured_obj = adj.resolve_path(path)
        if captured_obj is not None:
            return captured_obj, path

        # Still nothing found, maybe this is a predefined type attribute like `dtype`
        if eval_types:
            try:
                val = adj.eval(root_node)
                if val:
                    return [val, type_repr(val)]

            except Exception:
                pass

        return None, path

    # annotate generated code with the original source code line
    def set_lineno(adj, lineno):
        if adj.lineno is None or adj.lineno != lineno:
            line = lineno + adj.fun_lineno
            source = adj.raw_source[lineno].strip().ljust(80 - len(adj.indentation), " ")
            adj.add_forward(f"// {source}       <L {line}>")
            adj.add_reverse(f"// adj: {source}  <L {line}>")
        adj.lineno = lineno

    def get_node_source(adj, node):
        # return the Python code corresponding to the given AST node
        return ast.get_source_segment("".join(adj.raw_source), node)
