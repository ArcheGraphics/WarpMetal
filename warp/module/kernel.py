#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

from copy import copy as shallowcopy

from warp import types
from warp.codegen.adjoint import Adjoint
from warp.module.context import get_module


# caches source and compiled entry points for a kernel (will be populated after module loads)
class Kernel:
    def __init__(self, func, key=None, module=None, options=None, code_transformers=None):
        if code_transformers is None:
            code_transformers = []
        self.func = func

        if module is None:
            self.module = get_module(func.__module__)
        else:
            self.module = module

        if key is None:
            unique_key = self.module.generate_unique_kernel_key(func.__name__)
            self.key = unique_key
        else:
            self.key = key

        self.options = {} if options is None else options

        self.adj = Adjoint(func, transformers=code_transformers)

        # check if generic
        self.is_generic = False
        for arg_type in self.adj.arg_types.values():
            if types.type_is_generic(arg_type):
                self.is_generic = True
                break

        # unique signature (used to differentiate instances of generic kernels during codegen)
        self.sig = ""

        # known overloads for generic kernels, indexed by type signature
        self.overloads = {}

        # argument indices by name
        self.arg_indices = dict((a.label, i) for i, a in enumerate(self.adj.args))

        if self.module:
            self.module.register_kernel(self)

    def infer_argument_types(self, args):
        template_types = list(self.adj.arg_types.values())

        if len(args) != len(template_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        arg_names = list(self.adj.arg_types.keys())

        return types.infer_argument_types(args, template_types, arg_names)

    def add_overload(self, arg_types):
        if len(arg_types) != len(self.adj.arg_types):
            raise RuntimeError(f"Invalid number of arguments for kernel {self.key}")

        arg_names = list(self.adj.arg_types.keys())
        template_types = list(self.adj.arg_types.values())

        # make sure all argument types are concrete and match the kernel parameters
        for i in range(len(arg_types)):
            if not types.type_matches_template(arg_types[i], template_types[i]):
                if types.type_is_generic(arg_types[i]):
                    raise TypeError(
                        f"Kernel {self.key} argument '{arg_names[i]}' cannot be generic, got {arg_types[i]}"
                    )
                else:
                    raise TypeError(
                        f"Kernel {self.key} argument '{arg_names[i]}' type mismatch: expected {template_types[i]}, got {arg_types[i]}"
                    )

        # get a type signature from the given argument types
        sig = types.get_signature(arg_types, func_name=self.key)
        if sig in self.overloads:
            raise RuntimeError(
                f"Duplicate overload for kernel {self.key}, an overload with the given arguments already exists"
            )

        overload_annotations = dict(zip(arg_names, arg_types))

        # instantiate this kernel with the given argument types
        ovl = shallowcopy(self)
        ovl.adj = Adjoint(self.func, overload_annotations)
        ovl.is_generic = False
        ovl.overloads = {}
        ovl.sig = sig

        self.overloads[sig] = ovl

        self.module.unload()

        return ovl

    def get_overload(self, arg_types):
        sig = types.get_signature(arg_types, func_name=self.key)

        ovl = self.overloads.get(sig)
        if ovl is not None:
            return ovl
        else:
            return self.add_overload(arg_types)

    def get_mangled_name(self):
        if self.sig:
            return f"{self.key}_{self.sig}"
        else:
            return self.key
