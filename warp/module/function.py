#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.
from typing import Any, Callable, Tuple
from copy import copy as shallowcopy

from warp import types
from warp.codegen.adjoint import Adjoint
from warp.module.context import type_str


def create_value_func(type):
    def value_func(args, kwds, templates):
        return type

    return value_func


class Function:
    def __init__(
            self,
            func,
            key,
            namespace,
            input_types=None,
            value_func=None,
            template_func=None,
            module=None,
            variadic=False,
            initializer_list_func=None,
            export=False,
            doc="",
            group="",
            hidden=False,
            skip_replay=False,
            missing_grad=False,
            generic=False,
            native_func=None,
            defaults=None,
            custom_replay_func=None,
            native_snippet=None,
            adj_native_snippet=None,
            skip_forward_codegen=False,
            skip_reverse_codegen=False,
            custom_reverse_num_input_args=-1,
            custom_reverse_mode=False,
            overloaded_annotations=None,
            code_transformers=None,
            skip_adding_overload=False,
    ):
        if code_transformers is None:
            code_transformers = []
        self.func = func  # points to Python function decorated with @wp.func, may be None for builtins
        self.key = key
        self.namespace = namespace
        self.value_func = value_func  # a function that takes a list of args and a list of templates and returns the value type, e.g.: load(array, index) returns the type of value being loaded
        self.template_func = template_func
        self.input_types = {}
        self.export = export
        self.doc = doc
        self.group = group
        self.module = module
        self.variadic = variadic  # function can take arbitrary number of inputs, e.g.: printf()
        self.defaults = defaults
        # Function instance for a custom implementation of the replay pass
        self.custom_replay_func = custom_replay_func
        self.native_snippet = native_snippet
        self.adj_native_snippet = adj_native_snippet
        self.custom_grad_func = None

        if initializer_list_func is None:
            self.initializer_list_func = lambda x, y: False
        else:
            self.initializer_list_func = (
                initializer_list_func  # True if the arguments should be emitted as an initializer list in the c++ code
            )
        self.hidden = hidden  # function will not be listed in docs
        self.skip_replay = (
            skip_replay  # whether or not operation will be performed during the forward replay in the backward pass
        )
        self.missing_grad = missing_grad  # whether or not builtin is missing a corresponding adjoint
        self.generic = generic

        # allow registering builtin functions with a different name in Python from the native code
        if native_func is None:
            self.native_func = key
        else:
            self.native_func = native_func

        if func:
            # user-defined function

            # generic and concrete overload lookups by type signature
            self.user_templates = {}
            self.user_overloads = {}

            # user defined (Python) function
            self.adj = Adjoint(
                func,
                is_user_function=True,
                skip_forward_codegen=skip_forward_codegen,
                skip_reverse_codegen=skip_reverse_codegen,
                custom_reverse_num_input_args=custom_reverse_num_input_args,
                custom_reverse_mode=custom_reverse_mode,
                overload_annotations=overloaded_annotations,
                transformers=code_transformers,
            )

            # record input types
            for name, type in self.adj.arg_types.items():
                if name == "return":
                    self.value_func = create_value_func(type)

                else:
                    self.input_types[name] = type

        else:
            # builtin function

            # embedded linked list of all overloads
            # the builtin_functions dictionary holds
            # the list head for a given key (func name)
            self.overloads = []

            # builtin (native) function, canonicalize argument types
            for k, v in input_types.items():
                self.input_types[k] = types.type_to_warp(v)

            # cache mangled name
            if self.is_simple():
                self.mangled_name = self.mangle()
            else:
                self.mangled_name = None

        if not skip_adding_overload:
            self.add_overload(self)

        # add to current module
        if module:
            module.register_function(self, skip_adding_overload)

    def is_builtin(self):
        return self.func is None

    def is_simple(self):
        if self.variadic:
            return False

        # only export simple types that don't use arrays
        for k, v in self.input_types.items():
            if isinstance(v, types.array) or v == Any or v == Callable or v == Tuple:
                return False

        return_type = ""

        try:
            # todo: construct a default value for each of the functions args
            # so we can generate the return type for overloaded functions
            return_type = type_str(self.value_func(None, None, None))
        except Exception:
            return False

        if return_type.startswith("Tuple"):
            return False

        return True

    def mangle(self):
        # builds a mangled name for the C-exported
        # function, e.g.: builtin_normalize_vec3()

        name = "builtin_" + self.key

        types = []
        for t in self.input_types.values():
            types.append(t.__name__)

        return "_".join([name, *types])

    def add_overload(self, f: 'Function'):
        if self.is_builtin():
            # todo: note that it is an error to add two functions
            # with the exact same signature as this would cause compile
            # errors during compile time. We should check here if there
            # is a previously created function with the same signature
            self.overloads.append(f)

            # make sure variadic overloads appear last so non variadic
            # ones are matched first:
            self.overloads.sort(key=lambda f: f.variadic)

        else:
            # get function signature based on the input types
            sig = types.get_signature(
                f.input_types.values(), func_name=f.key, arg_names=list(f.input_types.keys())
            )

            # check if generic
            if types.is_generic_signature(sig):
                if sig in self.user_templates:
                    raise RuntimeError(
                        f"Duplicate generic function overload {self.key} with arguments {f.input_types.values()}"
                    )
                self.user_templates[sig] = f
            else:
                if sig in self.user_overloads:
                    raise RuntimeError(
                        f"Duplicate function overload {self.key} with arguments {f.input_types.values()}"
                    )
                self.user_overloads[sig] = f

    def get_overload(self, arg_types):
        assert not self.is_builtin()

        sig = types.get_signature(arg_types, func_name=self.key)

        f = self.user_overloads.get(sig)
        if f is not None:
            return f
        else:
            for f in self.user_templates.values():
                if len(f.input_types) != len(arg_types):
                    continue

                # try to match the given types to the function template types
                template_types = list(f.input_types.values())
                args_matched = True

                for i in range(len(arg_types)):
                    if not types.type_matches_template(arg_types[i], template_types[i]):
                        args_matched = False
                        break

                if args_matched:
                    # instantiate this function with the specified argument types

                    arg_names = f.input_types.keys()
                    overload_annotations = dict(zip(arg_names, arg_types))

                    ovl = shallowcopy(f)
                    ovl.adj = Adjoint(f.func, overload_annotations)
                    ovl.input_types = overload_annotations
                    ovl.value_func = None

                    self.user_overloads[sig] = ovl

                    return ovl

            # failed  to find overload
            return None

    def __repr__(self):
        inputs_str = ", ".join([f"{k}: {types.type_repr(v)}" for k, v in self.input_types.items()])
        return f"<Function {self.key}({inputs_str})>"
