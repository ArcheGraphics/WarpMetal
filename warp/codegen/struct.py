#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

import ctypes

from warp import types, float16
from warp.codegen.codegen import make_full_qualified_name, get_annotations
from warp.codegen.var import Var
from warp.module.function import Function
from warp.types import types_equal


class Struct:
    def __init__(self, cls, key, module):
        self.cls = cls
        self.module = module
        self.key = key

        self.vars = {}
        annotations = get_annotations(self.cls)
        for label, type in annotations.items():
            self.vars[label] = Var(label, type)

        fields = []
        for label, var in self.vars.items():
            if isinstance(var.type, types.array):
                fields.append((label, types.array_t))
            elif isinstance(var.type, Struct):
                fields.append((label, var.type.ctype))
            elif issubclass(var.type, ctypes.Array):
                fields.append((label, var.type))
            else:
                fields.append((label, var.type._type_))

        class StructType(ctypes.Structure):
            # if struct is empty, add a dummy field to avoid launch errors on CPU device ("ffi_prep_cif failed")
            _fields_ = fields or [("_dummy_", ctypes.c_byte)]

        self.ctype = StructType

        # create default constructor (zero-initialize)
        self.default_constructor = Function(
            func=None,
            key=self.key,
            namespace="",
            value_func=lambda *_: self,
            input_types={},
            initializer_list_func=lambda *_: False,
            native_func=make_full_qualified_name(self.cls),
        )

        # build a constructor that takes each param as a value
        input_types = {label: var.type for label, var in self.vars.items()}

        self.value_constructor = Function(
            func=None,
            key=self.key,
            namespace="",
            value_func=lambda *_: self,
            input_types=input_types,
            initializer_list_func=lambda *_: False,
            native_func=make_full_qualified_name(self.cls),
        )

        self.default_constructor.add_overload(self.value_constructor)

        if module:
            module.register_struct(self)

    def __call__(self):
        """
        This function returns s = StructInstance(self)
        s uses self.cls as template.
        To enable autocomplete on s, we inherit from self.cls.
        For example,

        @wp.struct
        class A:
            # annotations
            ...

        The type annotations are inherited in A(), allowing autocomplete in kernels
        """

        # return StructInstance(self)

        class NewStructInstance(self.cls, StructInstance):
            def __init__(inst):
                StructInstance.__init__(inst, self, None)

        return NewStructInstance()

    def initializer(self):
        return self.default_constructor

    # return structured NumPy dtype, including field names, formats, and offsets
    def numpy_dtype(self):
        names = []
        formats = []
        offsets = []
        for name, var in self.vars.items():
            names.append(name)
            offsets.append(getattr(self.ctype, name).offset)
            if isinstance(var.type, types.array):
                # array_t
                formats.append(types.array_t.numpy_dtype())
            elif isinstance(var.type, Struct):
                # nested struct
                formats.append(var.type.numpy_dtype())
            elif issubclass(var.type, ctypes.Array):
                scalar_typestr = types.type_typestr(var.type._wp_scalar_type_)
                if len(var.type._shape_) == 1:
                    # vector
                    formats.append(f"{var.type._length_}{scalar_typestr}")
                else:
                    # matrix
                    formats.append(f"{var.type._shape_}{scalar_typestr}")
            else:
                # scalar
                formats.append(types.type_typestr(var.type))

        return {"names": names, "formats": formats, "offsets": offsets, "itemsize": ctypes.sizeof(self.ctype)}

    # constructs a Warp struct instance from a pointer to the ctype
    def from_ptr(self, ptr):
        if not ptr:
            raise RuntimeError("NULL pointer exception")

        # create a new struct instance
        instance = self()

        for name, var in self.vars.items():
            offset = getattr(self.ctype, name).offset
            if isinstance(var.type, types.array):
                # We could reconstruct wp.array from array_t, but it's problematic.
                # There's no guarantee that the original wp.array is still allocated and
                # no easy way to make a backref.
                # Instead, we just create a stub annotation, which is not a fully usable array object.
                setattr(instance, name, types.array(dtype=var.type.dtype, ndim=var.type.ndim))
            elif isinstance(var.type, Struct):
                # nested struct
                value = var.type.from_ptr(ptr + offset)
                setattr(instance, name, value)
            elif issubclass(var.type, ctypes.Array):
                # vector/matrix
                value = var.type.from_ptr(ptr + offset)
                setattr(instance, name, value)
            else:
                # scalar
                cvalue = ctypes.cast(ptr + offset, ctypes.POINTER(var.type._type_)).contents
                if var.type == float16:
                    setattr(instance, name, types.half_bits_to_float(cvalue))
                else:
                    setattr(instance, name, cvalue.value)

        return instance


class StructInstance:
    def __init__(self, cls: Struct, ctype):
        super().__setattr__("_cls", cls)

        # maintain a c-types object for the top-level instance the struct
        if not ctype:
            super().__setattr__("_ctype", cls.ctype())
        else:
            super().__setattr__("_ctype", ctype)

        # create Python attributes for each of the struct's variables
        for field, var in cls.vars.items():
            if isinstance(var.type, Struct):
                self.__dict__[field] = StructInstance(var.type, getattr(self._ctype, field))
            elif isinstance(var.type, types.array):
                self.__dict__[field] = None
            else:
                self.__dict__[field] = var.type()

    def __setattr__(self, name, value):
        if name not in self._cls.vars:
            raise RuntimeError(f"Trying to set Warp struct attribute that does not exist {name}")

        var = self._cls.vars[name]

        # update our ctype flat copy
        if isinstance(var.type, types.array):
            if value is None:
                # create array with null pointer
                setattr(self._ctype, name, types.array_t())
            else:
                # wp.array
                assert isinstance(value, types.array)
                assert types_equal(
                    value.dtype, var.type.dtype
                ), f"assign to struct member variable {name} failed, expected type {types.type_repr(var.type.dtype)}, got type {types.type_repr(value.dtype)}"
                setattr(self._ctype, name, value.__ctype__())

        elif isinstance(var.type, Struct):
            # assign structs by-value, otherwise we would have problematic cases transferring ownership
            # of the underlying ctypes data between shared Python struct instances

            if not isinstance(value, StructInstance):
                raise RuntimeError(
                    f"Trying to assign a non-structure value to a struct attribute with type: {self._cls.key}"
                )

            # destination attribution on self
            dest = getattr(self, name)

            if dest._cls.key is not value._cls.key:
                raise RuntimeError(
                    f"Trying to assign a structure of type {value._cls.key} to an attribute of {self._cls.key}"
                )

            # update all nested ctype vars by deep copy
            for n in dest._cls.vars:
                setattr(dest, n, getattr(value, n))

            # early return to avoid updating our Python StructInstance
            return

        elif issubclass(var.type, ctypes.Array):
            # vector/matrix type, e.g. vec3
            if value is None:
                setattr(self._ctype, name, var.type())
            elif types_equal(type(value), var.type):
                setattr(self._ctype, name, value)
            else:
                # conversion from list/tuple, ndarray, etc.
                setattr(self._ctype, name, var.type(value))

        else:
            # primitive type
            if value is None:
                # zero initialize
                setattr(self._ctype, name, var.type._type_())
            else:
                if hasattr(value, "_type_"):
                    # assigning warp type value (e.g.: wp.float32)
                    value = value.value
                # float16 needs conversion to uint16 bits
                if var.type == float16:
                    setattr(self._ctype, name, types.float_to_half_bits(value))
                else:
                    setattr(self._ctype, name, value)

        # update Python instance
        super().__setattr__(name, value)

    def __ctype__(self):
        return self._ctype

    def __repr__(self):
        return struct_instance_repr_recursive(self, 0)

    # type description used in numpy structured arrays
    def numpy_dtype(self):
        return self._cls.numpy_dtype()

    # value usable in numpy structured arrays of .numpy_dtype(), e.g. (42, 13.37, [1.0, 2.0, 3.0])
    def numpy_value(self):
        npvalue = []
        for name, var in self._cls.vars.items():
            # get the attribute value
            value = getattr(self._ctype, name)

            if isinstance(var.type, types.array):
                # array_t
                npvalue.append(value.numpy_value())
            elif isinstance(var.type, Struct):
                # nested struct
                npvalue.append(value.numpy_value())
            elif issubclass(var.type, ctypes.Array):
                if len(var.type._shape_) == 1:
                    # vector
                    npvalue.append(list(value))
                else:
                    # matrix
                    npvalue.append([list(row) for row in value])
            else:
                # scalar
                if var.type == float16:
                    npvalue.append(types.half_bits_to_float(value))
                else:
                    npvalue.append(value)

        return tuple(npvalue)


def struct_instance_repr_recursive(inst: StructInstance, depth: int) -> str:
    indent = "\t"

    # handle empty structs
    if len(inst._cls.vars) == 0:
        return f"{inst._cls.key}()"

    lines = []
    lines.append(f"{inst._cls.key}(")

    for field_name, _ in inst._cls.ctype._fields_:
        field_value = getattr(inst, field_name, None)

        if isinstance(field_value, StructInstance):
            field_value = struct_instance_repr_recursive(field_value, depth + 1)

        lines.append(f"{indent * (depth + 1)}{field_name}={field_value},")

    lines.append(f"{indent * depth})")
    return "\n".join(lines)
