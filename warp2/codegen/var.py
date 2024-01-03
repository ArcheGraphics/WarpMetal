#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.
from warp2 import float32, int32
from warp2.codegen.codegen import compute_type_str, make_full_qualified_name
from warp2.codegen.reference import is_reference
from warp2.codegen.struct import Struct
from warp2.types import is_array


class Var:
    def __init__(self, label: str, type, requires_grad=False, constant=None, prefix=True):
        # convert built-in types to wp types
        if type == float:
            type = float32
        elif type == int:
            type = int32

        self.label = label
        self.type = type
        self.requires_grad = requires_grad
        self.constant = constant
        self.prefix = prefix

    def __str__(self):
        return self.label

    @staticmethod
    def type_to_ctype(t, value_type=False):
        if is_array(t):
            if hasattr(t.dtype, "_wp_generic_type_str_"):
                dtypestr = compute_type_str(f"wp::{t.dtype._wp_generic_type_str_}", t.dtype._wp_type_params_)
            elif isinstance(t.dtype, Struct):
                dtypestr = make_full_qualified_name(t.dtype.cls)
            elif t.dtype.__name__ in ("bool", "int", "float"):
                dtypestr = t.dtype.__name__
            else:
                dtypestr = f"wp::{t.dtype.__name__}"
            classstr = f"wp::{type(t).__name__}"
            return f"{classstr}_t<{dtypestr}>"
        elif isinstance(t, Struct):
            return make_full_qualified_name(t.cls)
        elif is_reference(t):
            if not value_type:
                return Var.type_to_ctype(t.value_type) + "*"
            else:
                return Var.type_to_ctype(t.value_type)
        elif hasattr(t, "_wp_generic_type_str_"):
            return compute_type_str(f"wp::{t._wp_generic_type_str_}", t._wp_type_params_)
        elif t.__name__ in ("bool", "int", "float"):
            return t.__name__
        else:
            return f"wp::{t.__name__}"

    def ctype(self, value_type=False):
        return Var.type_to_ctype(self.type, value_type)

    def emit(self, prefix: str = "var"):
        if self.prefix:
            return f"{prefix}_{self.label}"
        else:
            return self.label

    def emit_adj(self):
        return self.emit("adj")
