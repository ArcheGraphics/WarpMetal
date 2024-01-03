#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

from __future__ import annotations

import builtins
import ctypes
import hashlib
import inspect
import struct
import zlib
from typing import Any, Callable, Generic, List, Tuple, TypeVar, Union

import numpy as np

import warp2

# type hints
Length = TypeVar("Length", bound=int)
Rows = TypeVar("Rows")
Cols = TypeVar("Cols")

Int = TypeVar("Int")
Float = TypeVar("Float")
Scalar = TypeVar("Scalar")
Vector = Generic[Length, Scalar]
Matrix = Generic[Rows, Cols, Scalar]
Quaternion = Generic[Float]
Transformation = Generic[Float]

DType = TypeVar("DType")
Array = Generic[DType]

T = TypeVar("T")

# shared hash for all constants
_constant_hash = hashlib.sha256()


def constant(x):
    """Function to declare compile-time constants accessible from Warp kernels

    Args:
        x: Compile-time constant value, can be any of the built-in math types.
    """

    global _constant_hash

    # hash the constant value
    if isinstance(x, builtins.bool):
        # This needs to come before the check for `int` since all boolean
        # values are also instances of `int`.
        _constant_hash.update(struct.pack("?", x))
    elif isinstance(x, int):
        _constant_hash.update(struct.pack("<q", x))
    elif isinstance(x, float):
        _constant_hash.update(struct.pack("<d", x))
    elif isinstance(x, float16):
        # float16 is a special case
        p = ctypes.pointer(ctypes.c_float(x.value))
        _constant_hash.update(p.contents)
    elif isinstance(x, tuple(scalar_types)):
        p = ctypes.pointer(x._type_(x.value))
        _constant_hash.update(p.contents)
    elif isinstance(x, ctypes.Array):
        _constant_hash.update(bytes(x))
    else:
        raise RuntimeError(f"Invalid constant type: {type(x)}")

    return x


def float_to_half_bits(value):
    return warp2.context.runtime.core.float_to_half_bits(value)


def half_bits_to_float(value):
    return warp2.context.runtime.core.half_bits_to_float(value)


# ----------------------
# built-in types


def vector(length, dtype):
    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32

    class vec_t(ctypes.Array):
        # ctypes.Array data for length, shape and c type:
        _length_ = 0 if length is Any else length
        _shape_ = (_length_,)
        _type_ = ctypes.c_float if dtype in [Scalar, Float] else dtype._type_

        # warp2 scalar type:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [length, dtype]
        _wp_generic_type_str_ = "vec_t"
        _wp_constructor_ = "vector"

        # special handling for float16 type: in this case, data is stored
        # as uint16 but it's actually half precision floating point
        # data. This means we need to convert each of the arguments
        # to uint16s containing half float bits before storing them in
        # the array:
        scalar_import = float_to_half_bits if _wp_scalar_type_ == float16 else lambda x: x
        scalar_export = half_bits_to_float if _wp_scalar_type_ == float16 else lambda x: x

        def __init__(self, *args):
            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. (1, 2, 3)
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = vec_t.scalar_import(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, vec_t.scalar_import(args[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in vector constructor, expected {self._length_} elements, got {num_args}"
                )

        def __getitem__(self, key):
            if isinstance(key, int):
                return vec_t.scalar_export(super().__getitem__(key))
            elif isinstance(key, slice):
                if self._wp_scalar_type_ == float16:
                    return [vec_t.scalar_export(x) for x in super().__getitem__(key)]
                else:
                    return super().__getitem__(key)
            else:
                raise KeyError(f"Invalid key {key}, expected int or slice")

        def __setitem__(self, key, value):
            if isinstance(key, int):
                super().__setitem__(key, vec_t.scalar_import(value))
                return value
            elif isinstance(key, slice):
                if self._wp_scalar_type_ == float16:
                    super().__setitem__(key, [vec_t.scalar_import(x) for x in value])
                    return value
                else:
                    return super().__setitem__(key, value)
            else:
                raise KeyError(f"Invalid key {key}, expected int or slice")

        def __getattr__(self, name):
            idx = "xyzw".find(name)
            if idx != -1:
                return self.__getitem__(idx)

            return self.__getattribute__(name)

        def __setattr__(self, name, value):
            idx = "xyzw".find(name)
            if idx != -1:
                return self.__setitem__(idx, value)

            return super().__setattr__(name, value)

        def __add__(self, y):
            return warp2.add(self, y)

        def __radd__(self, y):
            return warp2.add(self, y)

        def __sub__(self, y):
            return warp2.sub(self, y)

        def __rsub__(self, x):
            return warp2.sub(x, self)

        def __mul__(self, y):
            return warp2.mul(self, y)

        def __rmul__(self, x):
            return warp2.mul(x, self)

        def __truediv__(self, y):
            return warp2.div(self, y)

        def __rdiv__(self, x):
            return warp2.div(x, self)

        def __pos__(self):
            return warp2.pos(self)

        def __neg__(self):
            return warp2.neg(self)

        def __str__(self):
            return f"[{', '.join(map(str, self))}]"

        def __eq__(self, other):
            for i in range(self._length_):
                if self[i] != other[i]:
                    return False
            return True

        @classmethod
        def from_ptr(cls, ptr):
            if ptr:
                # create a new vector instance and initialize the contents from the binary data
                # this skips float16 conversions, assuming that float16 data is already encoded as uint16
                value = cls()
                ctypes.memmove(ctypes.byref(value), ptr, ctypes.sizeof(cls._type_) * cls._length_)
                return value
            else:
                raise RuntimeError("NULL pointer exception")

    return vec_t


def matrix(shape, dtype):
    assert len(shape) == 2

    # canonicalize dtype
    if dtype == int:
        dtype = int32
    elif dtype == float:
        dtype = float32

    class mat_t(ctypes.Array):
        _length_ = 0 if shape[0] == Any or shape[1] == Any else shape[0] * shape[1]
        _shape_ = (0, 0) if _length_ == 0 else shape
        _type_ = ctypes.c_float if dtype in [Scalar, Float] else dtype._type_

        # warp2 scalar type:
        # used in type checking and when writing out c++ code for constructors:
        _wp_scalar_type_ = dtype
        _wp_type_params_ = [shape[0], shape[1], dtype]
        _wp_generic_type_str_ = "mat_t"
        _wp_constructor_ = "matrix"

        _wp_row_type_ = vector(0 if shape[1] == Any else shape[1], dtype)

        # special handling for float16 type: in this case, data is stored
        # as uint16 but it's actually half precision floating point
        # data. This means we need to convert each of the arguments
        # to uint16s containing half float bits before storing them in
        # the array:
        scalar_import = float_to_half_bits if _wp_scalar_type_ == float16 else lambda x: x
        scalar_export = half_bits_to_float if _wp_scalar_type_ == float16 else lambda x: x

        def __init__(self, *args):
            num_args = len(args)
            if num_args == 0:
                super().__init__()
            elif num_args == 1:
                if hasattr(args[0], "__len__"):
                    # try to copy from expanded sequence, e.g. [[1, 0], [0, 1]]
                    self.__init__(*args[0])
                else:
                    # set all elements to the same value
                    value = mat_t.scalar_import(args[0])
                    for i in range(self._length_):
                        super().__setitem__(i, value)
            elif num_args == self._length_:
                # set all scalar elements
                for i in range(self._length_):
                    super().__setitem__(i, mat_t.scalar_import(args[i]))
            elif num_args == self._shape_[0]:
                # row vectors
                for i, row in enumerate(args):
                    if not hasattr(row, "__len__") or len(row) != self._shape_[1]:
                        raise TypeError(
                            f"Invalid argument in matrix constructor, expected row of length {self._shape_[1]}, got {row}"
                        )
                    offset = i * self._shape_[1]
                    for i in range(self._shape_[1]):
                        super().__setitem__(offset + i, mat_t.scalar_import(row[i]))
            else:
                raise ValueError(
                    f"Invalid number of arguments in matrix constructor, expected {self._length_} elements, got {num_args}"
                )

        def __add__(self, y):
            return warp2.add(self, y)

        def __radd__(self, y):
            return warp2.add(self, y)

        def __sub__(self, y):
            return warp2.sub(self, y)

        def __rsub__(self, x):
            return warp2.sub(x, self)

        def __mul__(self, y):
            return warp2.mul(self, y)

        def __rmul__(self, x):
            return warp2.mul(x, self)

        def __matmul__(self, y):
            return warp2.mul(self, y)

        def __rmatmul__(self, x):
            return warp2.mul(x, self)

        def __truediv__(self, y):
            return warp2.div(self, y)

        def __rdiv__(self, x):
            return warp2.div(x, self)

        def __pos__(self):
            return warp2.pos(self)

        def __neg__(self):
            return warp2.neg(self)

        def __str__(self):
            row_str = []
            for r in range(self._shape_[0]):
                row_val = self.get_row(r)
                row_str.append(f"[{', '.join(map(str, row_val))}]")

            return "[" + ",\n ".join(row_str) + "]"

        def __eq__(self, other):
            for i in range(self._shape_[0]):
                for j in range(self._shape_[1]):
                    if self[i][j] != other[i][j]:
                        return False
            return True

        def get_row(self, r):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            row_data = super().__getitem__(slice(row_start, row_end))
            if self._wp_scalar_type_ == float16:
                return self._wp_row_type_(*[mat_t.scalar_export(x) for x in row_data])
            else:
                return self._wp_row_type_(row_data)

        def set_row(self, r, v):
            if r < 0 or r >= self._shape_[0]:
                raise IndexError("Invalid row index")
            row_start = r * self._shape_[1]
            row_end = row_start + self._shape_[1]
            if self._wp_scalar_type_ == float16:
                v = [mat_t.scalar_import(x) for x in v]
            super().__setitem__(slice(row_start, row_end), v)

        def __getitem__(self, key):
            if isinstance(key, Tuple):
                # element indexing m[i,j]
                if len(key) != 2:
                    raise KeyError(f"Invalid key, expected one or two indices, got {len(key)}")
                return mat_t.scalar_export(super().__getitem__(key[0] * self._shape_[1] + key[1]))
            elif isinstance(key, int):
                # row vector indexing m[r]
                return self.get_row(key)
            else:
                raise KeyError(f"Invalid key {key}, expected int or pair of ints")

        def __setitem__(self, key, value):
            if isinstance(key, Tuple):
                # element indexing m[i,j] = x
                if len(key) != 2:
                    raise KeyError(f"Invalid key, expected one or two indices, got {len(key)}")
                super().__setitem__(key[0] * self._shape_[1] + key[1], mat_t.scalar_import(value))
                return value
            elif isinstance(key, int):
                # row vector indexing m[r] = v
                self.set_row(key, value)
                return value
            else:
                raise KeyError(f"Invalid key {key}, expected int or pair of ints")

        @classmethod
        def from_ptr(cls, ptr):
            if ptr:
                # create a new matrix instance and initialize the contents from the binary data
                # this skips float16 conversions, assuming that float16 data is already encoded as uint16
                value = cls()
                ctypes.memmove(ctypes.byref(value), ptr, ctypes.sizeof(cls._type_) * cls._length_)
                return value
            else:
                raise RuntimeError("NULL pointer exception")

    return mat_t


class void:
    def __init__(self):
        pass


class bool:
    _length_ = 1
    _type_ = ctypes.c_bool

    def __init__(self, x=False):
        self.value = x


class float16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0.0):
        self.value = x


class float32:
    _length_ = 1
    _type_ = ctypes.c_float

    def __init__(self, x=0.0):
        self.value = x


class float64:
    _length_ = 1
    _type_ = ctypes.c_double

    def __init__(self, x=0.0):
        self.value = x


class int8:
    _length_ = 1
    _type_ = ctypes.c_int8

    def __init__(self, x=0):
        self.value = x


class uint8:
    _length_ = 1
    _type_ = ctypes.c_uint8

    def __init__(self, x=0):
        self.value = x


class int16:
    _length_ = 1
    _type_ = ctypes.c_int16

    def __init__(self, x=0):
        self.value = x


class uint16:
    _length_ = 1
    _type_ = ctypes.c_uint16

    def __init__(self, x=0):
        self.value = x


class int32:
    _length_ = 1
    _type_ = ctypes.c_int32

    def __init__(self, x=0):
        self.value = x


class uint32:
    _length_ = 1
    _type_ = ctypes.c_uint32

    def __init__(self, x=0):
        self.value = x


class int64:
    _length_ = 1
    _type_ = ctypes.c_int64

    def __init__(self, x=0):
        self.value = x


class uint64:
    _length_ = 1
    _type_ = ctypes.c_uint64

    def __init__(self, x=0):
        self.value = x


def quaternion(dtype=Any):
    class quat_t(vector(length=4, dtype=dtype)):
        pass
        # def __init__(self, *args):
        #     super().__init__(args)

    ret = quat_t
    ret._wp_type_params_ = [dtype]
    ret._wp_generic_type_str_ = "quat_t"
    ret._wp_constructor_ = "quaternion"

    return ret


class quath(quaternion(dtype=float16)):
    pass


class quatf(quaternion(dtype=float32)):
    pass


class quatd(quaternion(dtype=float64)):
    pass


def transformation(dtype=Any):
    class transform_t(vector(length=7, dtype=dtype)):
        _wp_init_from_components_sig_ = inspect.Signature(
            (
                inspect.Parameter(
                    "p",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=(0.0, 0.0, 0.0),
                ),
                inspect.Parameter(
                    "q",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        )
        _wp_type_params_ = [dtype]
        _wp_generic_type_str_ = "transform_t"
        _wp_constructor_ = "transformation"

        def __init__(self, *args, **kwargs):
            if len(args) == 1 and len(kwargs) == 0:
                if getattr(args[0], "_wp_generic_type_str_") == self._wp_generic_type_str_:
                    # Copy constructor.
                    super().__init__(*args[0])
                    return

            try:
                # For backward compatibility, try to check if the arguments
                # match the original signature that'd allow initializing
                # the `p` and `q` components separately.
                bound_args = self._wp_init_from_components_sig_.bind(*args, **kwargs)
                bound_args.apply_defaults()
                p, q = bound_args.args
            except (TypeError, ValueError):
                # Fallback to the vector's constructor.
                super().__init__(*args)
                return

            # Even if the arguments match the original “from components”
            # signature, we still need to make sure that they represent
            # sequences that can be unpacked.
            if hasattr(p, "__len__") and hasattr(q, "__len__"):
                # Initialize from the `p` and `q` components.
                super().__init__()
                self[0:3] = vector(length=3, dtype=dtype)(*p)
                self[3:7] = quaternion(dtype=dtype)(*q)
                return

            # Fallback to the vector's constructor.
            super().__init__(*args)

        @property
        def p(self):
            return self[0:3]

        @property
        def q(self):
            return self[3:7]

    return transform_t


class transformh(transformation(dtype=float16)):
    pass


class transformf(transformation(dtype=float32)):
    pass


class transformd(transformation(dtype=float64)):
    pass


class vec2h(vector(length=2, dtype=float16)):
    pass


class vec3h(vector(length=3, dtype=float16)):
    pass


class vec4h(vector(length=4, dtype=float16)):
    pass


class vec2f(vector(length=2, dtype=float32)):
    pass


class vec3f(vector(length=3, dtype=float32)):
    pass


class vec4f(vector(length=4, dtype=float32)):
    pass


class vec2d(vector(length=2, dtype=float64)):
    pass


class vec3d(vector(length=3, dtype=float64)):
    pass


class vec4d(vector(length=4, dtype=float64)):
    pass


class vec2b(vector(length=2, dtype=int8)):
    pass


class vec3b(vector(length=3, dtype=int8)):
    pass


class vec4b(vector(length=4, dtype=int8)):
    pass


class vec2ub(vector(length=2, dtype=uint8)):
    pass


class vec3ub(vector(length=3, dtype=uint8)):
    pass


class vec4ub(vector(length=4, dtype=uint8)):
    pass


class vec2s(vector(length=2, dtype=int16)):
    pass


class vec3s(vector(length=3, dtype=int16)):
    pass


class vec4s(vector(length=4, dtype=int16)):
    pass


class vec2us(vector(length=2, dtype=uint16)):
    pass


class vec3us(vector(length=3, dtype=uint16)):
    pass


class vec4us(vector(length=4, dtype=uint16)):
    pass


class vec2i(vector(length=2, dtype=int32)):
    pass


class vec3i(vector(length=3, dtype=int32)):
    pass


class vec4i(vector(length=4, dtype=int32)):
    pass


class vec2ui(vector(length=2, dtype=uint32)):
    pass


class vec3ui(vector(length=3, dtype=uint32)):
    pass


class vec4ui(vector(length=4, dtype=uint32)):
    pass


class vec2l(vector(length=2, dtype=int64)):
    pass


class vec3l(vector(length=3, dtype=int64)):
    pass


class vec4l(vector(length=4, dtype=int64)):
    pass


class vec2ul(vector(length=2, dtype=uint64)):
    pass


class vec3ul(vector(length=3, dtype=uint64)):
    pass


class vec4ul(vector(length=4, dtype=uint64)):
    pass


class mat22h(matrix(shape=(2, 2), dtype=float16)):
    pass


class mat33h(matrix(shape=(3, 3), dtype=float16)):
    pass


class mat44h(matrix(shape=(4, 4), dtype=float16)):
    pass


class mat22f(matrix(shape=(2, 2), dtype=float32)):
    pass


class mat33f(matrix(shape=(3, 3), dtype=float32)):
    pass


class mat44f(matrix(shape=(4, 4), dtype=float32)):
    pass


class mat22d(matrix(shape=(2, 2), dtype=float64)):
    pass


class mat33d(matrix(shape=(3, 3), dtype=float64)):
    pass


class mat44d(matrix(shape=(4, 4), dtype=float64)):
    pass


class spatial_vectorh(vector(length=6, dtype=float16)):
    pass


class spatial_vectorf(vector(length=6, dtype=float32)):
    pass


class spatial_vectord(vector(length=6, dtype=float64)):
    pass


class spatial_matrixh(matrix(shape=(6, 6), dtype=float16)):
    pass


class spatial_matrixf(matrix(shape=(6, 6), dtype=float32)):
    pass


class spatial_matrixd(matrix(shape=(6, 6), dtype=float64)):
    pass


# built-in type aliases that default to 32bit precision
vec2 = vec2f
vec3 = vec3f
vec4 = vec4f
mat22 = mat22f
mat33 = mat33f
mat44 = mat44f
quat = quatf
transform = transformf
spatial_vector = spatial_vectorf
spatial_matrix = spatial_matrixf

int_types = [int8, uint8, int16, uint16, int32, uint32, int64, uint64]
float_types = [float16, float32, float64]
scalar_types = int_types + float_types

vector_types = [
    vec2b,
    vec2ub,
    vec2s,
    vec2us,
    vec2i,
    vec2ui,
    vec2l,
    vec2ul,
    vec2h,
    vec2f,
    vec2d,
    vec3b,
    vec3ub,
    vec3s,
    vec3us,
    vec3i,
    vec3ui,
    vec3l,
    vec3ul,
    vec3h,
    vec3f,
    vec3d,
    vec4b,
    vec4ub,
    vec4s,
    vec4us,
    vec4i,
    vec4ui,
    vec4l,
    vec4ul,
    vec4h,
    vec4f,
    vec4d,
    mat22h,
    mat22f,
    mat22d,
    mat33h,
    mat33f,
    mat33d,
    mat44h,
    mat44f,
    mat44d,
    quath,
    quatf,
    quatd,
    transformh,
    transformf,
    transformd,
    spatial_vectorh,
    spatial_vectorf,
    spatial_vectord,
    spatial_matrixh,
    spatial_matrixf,
    spatial_matrixd,
]

np_dtype_to_warp_type = {
    np.dtype(np.bool_): bool,
    np.dtype(np.int8): int8,
    np.dtype(np.uint8): uint8,
    np.dtype(np.int16): int16,
    np.dtype(np.uint16): uint16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.byte): int8,
    np.dtype(np.ubyte): uint8,
    np.dtype(np.float16): float16,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
}

warp_type_to_np_dtype = {
    bool: np.bool_,
    int8: np.int8,
    int16: np.int16,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
    uint16: np.uint16,
    uint32: np.uint32,
    uint64: np.uint64,
    float16: np.float16,
    float32: np.float32,
    float64: np.float64,
}


# represent a Python range iterator
class range_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see bvh.h
class bvh_query_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see mesh.h
class mesh_query_aabb_t:
    def __init__(self):
        pass


# definition just for kernel type (cannot be a parameter), see hash_grid.h
class hash_grid_query_t:
    def __init__(self):
        pass


# maximum number of dimensions, must match array.h
ARRAY_MAX_DIMS = 4
LAUNCH_MAX_DIMS = 4

# must match array.h
ARRAY_TYPE_REGULAR = 0
ARRAY_TYPE_INDEXED = 1
ARRAY_TYPE_FABRIC = 2
ARRAY_TYPE_FABRIC_INDEXED = 3


# represents bounds for kernel launch (number of threads across multiple dimensions)
class launch_bounds_t(ctypes.Structure):
    _fields_ = [("shape", ctypes.c_int32 * LAUNCH_MAX_DIMS), ("ndim", ctypes.c_int32), ("size", ctypes.c_size_t)]

    def __init__(self, shape):
        if isinstance(shape, int):
            # 1d launch
            self.ndim = 1
            self.size = shape
            self.shape[0] = shape

        else:
            # nd launch
            self.ndim = len(shape)
            self.size = 1

            for i in range(self.ndim):
                self.shape[i] = shape[i]
                self.size = self.size * shape[i]

        # initialize the remaining dims to 1
        for i in range(self.ndim, LAUNCH_MAX_DIMS):
            self.shape[i] = 1


class shape_t(ctypes.Structure):
    _fields_ = [("dims", ctypes.c_int32 * ARRAY_MAX_DIMS)]

    def __init__(self):
        pass


class array_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_uint64),
        ("grad", ctypes.c_uint64),
        ("shape", ctypes.c_int32 * ARRAY_MAX_DIMS),
        ("strides", ctypes.c_int32 * ARRAY_MAX_DIMS),
        ("ndim", ctypes.c_int32),
    ]

    def __init__(self, data=0, grad=0, ndim=0, shape=(0,), strides=(0,)):
        self.data = data
        self.grad = grad
        self.ndim = ndim
        for i in range(ndim):
            self.shape[i] = shape[i]
            self.strides[i] = strides[i]

    # structured type description used when array_t is packed in a struct and shared via numpy structured array.
    @classmethod
    def numpy_dtype(cls):
        return cls._numpy_dtype_

    # structured value used when array_t is packed in a struct and shared via a numpy structured array
    def numpy_value(self):
        return (self.data, self.grad, list(self.shape), list(self.strides), self.ndim)


# NOTE: must match array_t._fields_
array_t._numpy_dtype_ = {
    "names": ["data", "grad", "shape", "strides", "ndim"],
    "formats": ["u8", "u8", f"{ARRAY_MAX_DIMS}i4", f"{ARRAY_MAX_DIMS}i4", "i4"],
    "offsets": [
        array_t.data.offset,
        array_t.grad.offset,
        array_t.shape.offset,
        array_t.strides.offset,
        array_t.ndim.offset,
    ],
    "itemsize": ctypes.sizeof(array_t),
}


class indexedarray_t(ctypes.Structure):
    _fields_ = [
        ("data", array_t),
        ("indices", ctypes.c_void_p * ARRAY_MAX_DIMS),
        ("shape", ctypes.c_int32 * ARRAY_MAX_DIMS),
    ]

    def __init__(self, data, indices, shape):
        if data is None:
            self.data = array().__ctype__()
            for i in range(ARRAY_MAX_DIMS):
                self.indices[i] = ctypes.c_void_p(None)
                self.shape[i] = 0
        else:
            self.data = data.__ctype__()
            for i in range(data.ndim):
                if indices[i] is not None:
                    self.indices[i] = ctypes.c_void_p(indices[i].ptr)
                else:
                    self.indices[i] = ctypes.c_void_p(None)
                self.shape[i] = shape[i]


def type_ctype(dtype):
    if dtype == float:
        return ctypes.c_float
    elif dtype == int:
        return ctypes.c_int32
    else:
        # scalar type
        return dtype._type_


def type_length(dtype):
    if dtype == float or dtype == int or isinstance(dtype, warp2.codegen.Struct):
        return 1
    else:
        return dtype._length_


def type_scalar_type(dtype):
    return getattr(dtype, "_wp_scalar_type_", dtype)


def type_size_in_bytes(dtype):
    if dtype.__module__ == "ctypes":
        return ctypes.sizeof(dtype)
    elif isinstance(dtype, warp2.codegen.Struct):
        return ctypes.sizeof(dtype.ctype)
    elif dtype == float or dtype == int:
        return 4
    elif hasattr(dtype, "_type_"):
        return getattr(dtype, "_length_", 1) * ctypes.sizeof(dtype._type_)

    else:
        return 0


def type_to_warp(dtype):
    if dtype == float:
        return float32
    elif dtype == int:
        return int32
    else:
        return dtype


def type_typestr(dtype):
    if dtype == bool:
        return "?"
    elif dtype == float16:
        return "<f2"
    elif dtype == float32:
        return "<f4"
    elif dtype == float64:
        return "<f8"
    elif dtype == int8:
        return "b"
    elif dtype == uint8:
        return "B"
    elif dtype == int16:
        return "<i2"
    elif dtype == uint16:
        return "<u2"
    elif dtype == int32:
        return "<i4"
    elif dtype == uint32:
        return "<u4"
    elif dtype == int64:
        return "<i8"
    elif dtype == uint64:
        return "<u8"
    elif isinstance(dtype, warp2.codegen.Struct):
        return f"|V{ctypes.sizeof(dtype.ctype)}"
    elif issubclass(dtype, ctypes.Array):
        return type_typestr(dtype._wp_scalar_type_)
    else:
        raise Exception("Unknown ctype")


# converts any known type to a human readable string, good for error messages, reporting etc
def type_repr(t):
    if is_array(t):
        return str(f"array(ndim={t.ndim}, dtype={t.dtype})")
    if type_is_vector(t):
        return str(f"vector(length={t._shape_[0]}, dtype={t._wp_scalar_type_})")
    if type_is_matrix(t):
        return str(f"matrix(shape=({t._shape_[0]}, {t._shape_[1]}), dtype={t._wp_scalar_type_})")
    if isinstance(t, warp2.codegen.Struct):
        return type_repr(t.cls)
    if t in scalar_types:
        return t.__name__

    try:
        return t.__module__ + "." + t.__qualname__
    except AttributeError:
        return str(t)


def type_is_int(t):
    if t == int:
        t = int32

    return t in int_types


def type_is_float(t):
    if t == float:
        t = float32

    return t in float_types


# returns True if the passed *type* is a vector
def type_is_vector(t):
    if hasattr(t, "_wp_generic_type_str_") and t._wp_generic_type_str_ == "vec_t":
        return True
    else:
        return False


# returns True if the passed *type* is a matrix
def type_is_matrix(t):
    if hasattr(t, "_wp_generic_type_str_") and t._wp_generic_type_str_ == "mat_t":
        return True
    else:
        return False


# returns true for all value types (int, float, bool, scalars, vectors, matrices)
def type_is_value(x):
    if (x == int) or (x == float) or (x == builtins.bool) or (x in scalar_types) or issubclass(x, ctypes.Array):
        return True
    else:
        return False


# equivalent of the above but for values
def is_int(x):
    return type_is_int(type(x))


def is_float(x):
    return type_is_float(type(x))


def is_value(x):
    return type_is_value(type(x))


# returns true if the passed *instance* is one of the array types
def is_array(a):
    return isinstance(a, array_types)


def types_equal(a, b, match_generic=False):
    # convert to canonical types
    if a == float:
        a = float32
    elif a == int:
        a = int32

    if b == float:
        b = float32
    elif b == int:
        b = int32

    compatible_bool_types = [builtins.bool, bool]

    def are_equal(p1, p2):
        if match_generic:
            if p1 == Any or p2 == Any:
                return True
            if p1 == Scalar and p2 in scalar_types:
                return True
            if p2 == Scalar and p1 in scalar_types:
                return True
            if p1 == Scalar and p2 == Scalar:
                return True
            if p1 == Float and p2 in float_types:
                return True
            if p2 == Float and p1 in float_types:
                return True
            if p1 == Float and p2 == Float:
                return True

        # convert to canonical types
        if p1 == float:
            p1 = float32
        elif p1 == int:
            p1 = int32

        if p2 == float:
            p2 = float32
        elif b == int:
            p2 = int32

        if p1 in compatible_bool_types and p2 in compatible_bool_types:
            return True
        else:
            return p1 == p2

    if (
            hasattr(a, "_wp_generic_type_str_")
            and hasattr(b, "_wp_generic_type_str_")
            and a._wp_generic_type_str_ == b._wp_generic_type_str_
    ):
        return all([are_equal(p1, p2) for p1, p2 in zip(a._wp_type_params_, b._wp_type_params_)])
    if is_array(a) and type(a) is type(b):
        return True
    else:
        return are_equal(a, b)


def strides_from_shape(shape: Tuple, dtype):
    ndims = len(shape)
    strides = [None] * ndims

    i = ndims - 1
    strides[i] = type_size_in_bytes(dtype)

    while i > 0:
        strides[i - 1] = strides[i] * shape[i]
        i -= 1

    return tuple(strides)


class array(Array):
    # member attributes available during code-gen (e.g.: d = array.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(
            self,
            data=None,
            dtype: DType = Any,
            shape=None,
            strides=None,
            length=None,
            ptr=None,
            capacity=None,
            device=None,
            pinned=False,
            copy=True,
            owner=True,  # TODO: replace with deleter=None
            ndim=None,
            grad=None,
            requires_grad=False,
    ):
        """Constructs a new Warp array object

        When the ``data`` argument is a valid list, tuple, or ndarray the array will be constructed from this object's data.
        For objects that are not stored sequentially in memory (e.g.: a list), then the data will first
        be flattened before being transferred to the memory space given by device.

        The second construction path occurs when the ``ptr`` argument is a non-zero uint64 value representing the
        start address in memory where existing array data resides, e.g.: from an external or C-library. The memory
        allocation should reside on the same device given by the device argument, and the user should set the length
        and dtype parameter appropriately.

        If neither ``data`` nor ``ptr`` are specified, the ``shape`` or ``length`` arguments are checked next.
        This construction path can be used to create new uninitialized arrays, but users are encouraged to call
        ``wp.empty()``, ``wp.zeros()``, or ``wp.full()`` instead to create new arrays.

        If none of the above arguments are specified, a simple type annotation is constructed.  This is used when annotating
        kernel arguments or struct members (e.g.,``arr: wp.array(dtype=float)``).  In this case, only ``dtype`` and ``ndim``
        are taken into account and no memory is allocated for the array.

        Args:
            data (Union[list, tuple, ndarray]) An object to construct the array from, can be a Tuple, List, or generally any type convertible to an np.array
            dtype (Union): One of the built-in types, e.g.: :class:`warp2.mat33`, if dtype is Any and data an ndarray then it will be inferred from the array data type
            shape (tuple): Dimensions of the array
            strides (tuple): Number of bytes in each dimension between successive elements of the array
            length (int): Number of elements of the data type (deprecated, users should use `shape` argument)
            ptr (uint64): Address of an external memory address to alias (data should be None)
            capacity (int): Maximum size in bytes of the ptr allocation (data should be None)
            device (Devicelike): Device the array lives on
            copy (bool): Whether the incoming data will be copied or aliased, this is only possible when the incoming `data` already lives on the device specified and types match
            owner (bool): Should the array object try to deallocate memory when it is deleted
            requires_grad (bool): Whether or not gradients will be tracked for this array, see :class:`warp2.Tape` for details
            grad (array): The gradient array to use
            pinned (bool): Whether to allocate pinned host memory, which allows asynchronous host-device transfers (only applicable with device="cpu")

        """

        self.owner = False
        self.ctype = None
        self._requires_grad = False
        self._grad = None
        # __array_interface__ or __cuda_array_interface__, evaluated lazily and cached
        self._array_interface = None
        self.is_transposed = False

        # canonicalize dtype
        if dtype == int:
            dtype = int32
        elif dtype == float:
            dtype = float32

        # convert shape to tuple (or leave shape=None if neither shape nor length were specified)
        if shape is not None:
            if isinstance(shape, int):
                shape = (shape,)
            else:
                shape = tuple(shape)
                if len(shape) > ARRAY_MAX_DIMS:
                    raise RuntimeError(
                        f"Failed to create array with shape {shape}, the maximum number of dimensions is {ARRAY_MAX_DIMS}"
                    )
        elif length is not None:
            # backward compatibility
            shape = (length,)

        # determine the construction path from the given arguments
        if data is not None:
            # data or ptr, not both
            if ptr is not None:
                raise RuntimeError("Can only construct arrays with either `data` or `ptr` arguments, not both")
            self._init_from_data(data, dtype, shape, device, copy, pinned)
        elif ptr is not None:
            self._init_from_ptr(ptr, dtype, shape, strides, capacity, device, owner, pinned)
        elif shape is not None:
            self._init_new(dtype, shape, strides, device, pinned)
        else:
            self._init_annotation(dtype, ndim or 1)

        # initialize gradient, if needed
        if self.device is not None:
            if grad is not None:
                # this will also check whether the gradient array is compatible
                self.grad = grad
            else:
                # allocate gradient if needed
                self._requires_grad = requires_grad
                if requires_grad:
                    with warp2.ScopedStream(self.device.null_stream):
                        self._alloc_grad()

    def _init_from_data(self, data, dtype, shape, device, copy, pinned):
        if not hasattr(data, "__len__"):
            raise RuntimeError(f"Data must be a sequence or array, got scalar {data}")

        if hasattr(dtype, "_wp_scalar_type_"):
            dtype_shape = dtype._shape_
            dtype_ndim = len(dtype_shape)
            scalar_dtype = dtype._wp_scalar_type_
        else:
            dtype_shape = ()
            dtype_ndim = 0
            scalar_dtype = dtype

        # convert input data to ndarray (handles lists, tuples, etc.) and determine dtype
        if dtype == Any:
            # infer dtype from data
            try:
                arr = np.array(data, copy=False, ndmin=1)
            except Exception as e:
                raise RuntimeError(f"Failed to convert input data to an array: {e}")
            dtype = np_dtype_to_warp_type.get(arr.dtype)
            if dtype is None:
                raise RuntimeError(f"Unsupported input data dtype: {arr.dtype}")
        elif isinstance(dtype, warp2.codegen.Struct):
            if isinstance(data, np.ndarray):
                # construct from numpy structured array
                if data.dtype != dtype.numpy_dtype():
                    raise RuntimeError(
                        f"Invalid source data type for array of structs, expected {dtype.numpy_dtype()}, got {data.dtype}"
                    )
                arr = data
            elif isinstance(data, (list, tuple)):
                # construct from a sequence of structs
                try:
                    # convert each struct instance to its corresponding ctype
                    ctype_list = [v.__ctype__() for v in data]
                    # convert the list of ctypes to a contiguous ctypes array
                    ctype_arr = (dtype.ctype * len(ctype_list))(*ctype_list)
                    # convert to numpy
                    arr = np.frombuffer(ctype_arr, dtype=dtype.ctype)
                except Exception as e:
                    raise RuntimeError(
                        f"Error while trying to construct Warp array from a sequence of Warp structs: {e}"
                    )
            else:
                raise RuntimeError(
                    "Invalid data argument for array of structs, expected a sequence of structs or a NumPy structured array"
                )
        else:
            # convert input data to the given dtype
            npdtype = warp_type_to_np_dtype.get(scalar_dtype)
            if npdtype is None:
                raise RuntimeError(
                    f"Failed to convert input data to an array with Warp type {warp2.context.type_str(dtype)}"
                )
            try:
                arr = np.array(data, dtype=npdtype, copy=False, ndmin=1)
            except Exception as e:
                raise RuntimeError(f"Failed to convert input data to an array with type {npdtype}: {e}")

        # determine whether the input needs reshaping
        target_npshape = None
        if shape is not None:
            target_npshape = (*shape, *dtype_shape)
        elif dtype_ndim > 0:
            # prune inner dimensions of length 1
            while arr.ndim > 1 and arr.shape[-1] == 1:
                arr = np.squeeze(arr, axis=-1)
            # if the inner dims don't match exactly, check if the innermost dim is a multiple of type length
            if arr.ndim < dtype_ndim or arr.shape[-dtype_ndim:] != dtype_shape:
                if arr.shape[-1] == dtype._length_:
                    target_npshape = (*arr.shape[:-1], *dtype_shape)
                elif arr.shape[-1] % dtype._length_ == 0:
                    target_npshape = (*arr.shape[:-1], arr.shape[-1] // dtype._length_, *dtype_shape)
                else:
                    if dtype_ndim == 1:
                        raise RuntimeError(
                            f"The inner dimensions of the input data are not compatible with the requested vector type {warp2.context.type_str(dtype)}: expected an inner dimension that is a multiple of {dtype._length_}"
                        )
                    else:
                        raise RuntimeError(
                            f"The inner dimensions of the input data are not compatible with the requested matrix type {warp2.context.type_str(dtype)}: expected inner dimensions {dtype._shape_} or a multiple of {dtype._length_}"
                        )

        if target_npshape is not None:
            try:
                arr = arr.reshape(target_npshape)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reshape the input data to the given shape {shape} and type {warp2.context.type_str(dtype)}: {e}"
                )

        # determine final shape and strides
        if dtype_ndim > 0:
            # make sure the inner dims are contiguous for vector/matrix types
            scalar_size = type_size_in_bytes(dtype._wp_scalar_type_)
            inner_contiguous = arr.strides[-1] == scalar_size
            if inner_contiguous and dtype_ndim > 1:
                inner_contiguous = arr.strides[-2] == scalar_size * dtype_shape[-1]

            if not inner_contiguous:
                arr = np.ascontiguousarray(arr)

            shape = arr.shape[:-dtype_ndim] or (1,)
            strides = arr.strides[:-dtype_ndim] or (type_size_in_bytes(dtype),)
        else:
            shape = arr.shape or (1,)
            strides = arr.strides or (type_size_in_bytes(dtype),)

        device = warp2.get_device(device)

        if device.is_cpu and not copy and not pinned:
            # reference numpy memory directly
            self._init_from_ptr(arr.ctypes.data, dtype, shape, strides, None, device, False, False)
            # keep a ref to the source array to keep allocation alive
            self._ref = arr
        else:
            # copy data into a new array
            self._init_new(dtype, shape, None, device, pinned)
            src = array(
                ptr=arr.ctypes.data,
                dtype=dtype,
                shape=shape,
                strides=strides,
                device="cpu",
                copy=False,
                owner=False,
            )
            warp2.copy(self, src)

    def _init_from_ptr(self, ptr, dtype, shape, strides, capacity, device, owner, pinned):
        if dtype == Any:
            raise RuntimeError("A concrete data type is required to create the array")

        device = warp2.get_device(device)

        size = 1
        for d in shape:
            size *= d

        contiguous_strides = strides_from_shape(shape, dtype)

        if strides is None:
            strides = contiguous_strides
            is_contiguous = True
            if capacity is None:
                capacity = size * type_size_in_bytes(dtype)
        else:
            is_contiguous = strides == contiguous_strides
            if capacity is None:
                capacity = shape[0] * strides[0]

        self.dtype = dtype
        self.ndim = len(shape)
        self.size = size
        self.capacity = capacity
        self.shape = shape
        self.strides = strides
        self.ptr = ptr
        self.device = device
        self.owner = owner
        self.pinned = pinned if device.is_cpu else False
        self.is_contiguous = is_contiguous

    def _init_new(self, dtype, shape, strides, device, pinned):
        if dtype == Any:
            raise RuntimeError("A concrete data type is required to create the array")

        device = warp2.get_device(device)

        size = 1
        for d in shape:
            size *= d

        contiguous_strides = strides_from_shape(shape, dtype)

        if strides is None:
            strides = contiguous_strides
            is_contiguous = True
            capacity = size * type_size_in_bytes(dtype)
        else:
            is_contiguous = strides == contiguous_strides
            capacity = shape[0] * strides[0]

        if capacity > 0:
            ptr = device.allocator.alloc(capacity, pinned=pinned)
            if ptr is None:
                raise RuntimeError(f"Array allocation failed on device: {device} for {capacity} bytes")
        else:
            ptr = None

        self.dtype = dtype
        self.ndim = len(shape)
        self.size = size
        self.capacity = capacity
        self.shape = shape
        self.strides = strides
        self.ptr = ptr
        self.device = device
        self.owner = True
        self.pinned = pinned if device.is_cpu else False
        self.is_contiguous = is_contiguous

    def _init_annotation(self, dtype, ndim):
        self.dtype = dtype
        self.ndim = ndim
        self.size = 0
        self.capacity = 0
        self.shape = (0,) * ndim
        self.strides = (0,) * ndim
        self.ptr = None
        self.device = None
        self.owner = False
        self.pinned = False
        self.is_contiguous = False

    @property
    def __array_interface__(self):
        # raising an AttributeError here makes hasattr() return False
        if self.device is None or not self.device.is_cpu:
            raise AttributeError(f"__array_interface__ not supported because device is {self.device}")

        if self._array_interface is None:
            # get flat shape (including type shape)
            if isinstance(self.dtype, warp2.codegen.Struct):
                # struct
                arr_shape = self.shape
                arr_strides = self.strides
                descr = self.dtype.numpy_dtype()
            elif issubclass(self.dtype, ctypes.Array):
                # vector type, flatten the dimensions into one tuple
                arr_shape = (*self.shape, *self.dtype._shape_)
                dtype_strides = strides_from_shape(self.dtype._shape_, self.dtype._type_)
                arr_strides = (*self.strides, *dtype_strides)
                descr = None
            else:
                # scalar type
                arr_shape = self.shape
                arr_strides = self.strides
                descr = None

            self._array_interface = {
                "data": (self.ptr if self.ptr is not None else 0, False),
                "shape": tuple(arr_shape),
                "strides": tuple(arr_strides),
                "typestr": type_typestr(self.dtype),
                "descr": descr,  # optional description of structured array layout
                "version": 3,
            }

        return self._array_interface

    @property
    def __cuda_array_interface__(self):
        # raising an AttributeError here makes hasattr() return False
        if self.device is None or not self.device.is_cuda:
            raise AttributeError(f"__cuda_array_interface__ is not supported because device is {self.device}")

        if self._array_interface is None:
            # get flat shape (including type shape)
            if issubclass(self.dtype, ctypes.Array):
                # vector type, flatten the dimensions into one tuple
                arr_shape = (*self.shape, *self.dtype._shape_)
                dtype_strides = strides_from_shape(self.dtype._shape_, self.dtype._type_)
                arr_strides = (*self.strides, *dtype_strides)
            else:
                # scalar or struct type
                arr_shape = self.shape
                arr_strides = self.strides

            self._array_interface = {
                "data": (self.ptr if self.ptr is not None else 0, False),
                "shape": tuple(arr_shape),
                "strides": tuple(arr_strides),
                "typestr": type_typestr(self.dtype),
                "version": 2,
            }

        return self._array_interface

    def __del__(self):
        if self.owner:
            # use CUDA context guard to avoid side effects during garbage collection
            with self.device.context_guard:
                self.device.allocator.free(self.ptr, self.capacity, self.pinned)

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.device is None:
            # for 'empty' arrays we just return the type information, these are used in kernel function signatures
            return f"array{self.dtype}"
        else:
            return str(self.numpy())

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.ndim == 1:
                raise RuntimeError("Item indexing is not supported on wp.array objects")
            key = [key]
        elif isinstance(key, (slice, array)):
            key = [key]
        elif isinstance(key, Tuple):
            contains_slice = False
            contains_indices = False
            for k in key:
                if isinstance(k, slice):
                    contains_slice = True
                if isinstance(k, array):
                    contains_indices = True
            if not contains_slice and not contains_indices and len(key) == self.ndim:
                raise RuntimeError("Item indexing is not supported on wp.array objects")
        else:
            raise RuntimeError(f"Invalid index: {key}")

        new_key = []
        for i in range(0, len(key)):
            new_key.append(key[i])
        for i in range(len(key), self.ndim):
            new_key.append(slice(None, None, None))
        key = tuple(new_key)

        new_shape = []
        new_strides = []
        ptr_offset = 0
        new_dim = self.ndim

        # maps dimension index to an array of indices, if given
        index_arrays = {}

        for idx, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, step = k.start, k.stop, k.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = self.shape[idx]
                if step is None:
                    step = 1
                if start < 0:
                    start = self.shape[idx] + start
                if stop < 0:
                    stop = self.shape[idx] + stop

                if start < 0 or start >= self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")
                if stop < 1 or stop > self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")
                if stop <= start:
                    raise RuntimeError(f"Invalid indexing in slice: {start}:{stop}:{step}")

                new_shape.append(-((stop - start) // -step))  # ceil division
                new_strides.append(self.strides[idx] * step)

                ptr_offset += self.strides[idx] * start

            elif isinstance(k, array):
                # note: index array properties will be checked during indexedarray construction
                index_arrays[idx] = k

                # shape and strides are unchanged for this dimension
                new_shape.append(self.shape[idx])
                new_strides.append(self.strides[idx])

            else:  # is int
                start = k
                if start < 0:
                    start = self.shape[idx] + start
                if start < 0 or start >= self.shape[idx]:
                    raise RuntimeError(f"Invalid indexing in slice: {k}")
                new_dim -= 1

                ptr_offset += self.strides[idx] * start

        # handle grad
        if self.grad is not None:
            new_grad = array(
                ptr=self.grad.ptr + ptr_offset if self.grad.ptr is not None else None,
                dtype=self.grad.dtype,
                shape=tuple(new_shape),
                strides=tuple(new_strides),
                device=self.grad.device,
                pinned=self.grad.pinned,
                owner=False,
            )
            # store back-ref to stop data being destroyed
            new_grad._ref = self.grad
        else:
            new_grad = None

        a = array(
            ptr=self.ptr + ptr_offset if self.ptr is not None else None,
            dtype=self.dtype,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            device=self.device,
            pinned=self.pinned,
            owner=False,
            grad=new_grad,
        )

        # store back-ref to stop data being destroyed
        a._ref = self

        if index_arrays:
            indices = [None] * self.ndim
            for dim, index_array in index_arrays.items():
                indices[dim] = index_array
            return indexedarray(a, indices)
        else:
            return a

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        if self.ctype is None:
            data = 0 if self.ptr is None else ctypes.c_uint64(self.ptr)
            grad = 0 if self.grad is None or self.grad.ptr is None else ctypes.c_uint64(self.grad.ptr)
            self.ctype = array_t(data=data, grad=grad, ndim=self.ndim, shape=self.shape, strides=self.strides)

        return self.ctype

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        if grad is None:
            self._grad = None
            self._requires_grad = False
        else:
            # make sure the given gradient array is compatible
            if (
                    grad.dtype != self.dtype
                    or grad.shape != self.shape
                    or grad.strides != self.strides
                    or grad.device != self.device
            ):
                raise ValueError("The given gradient array is incompatible")
            self._grad = grad
            self._requires_grad = True

        # trigger re-creation of C-representation
        self.ctype = None

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: builtins.bool):
        if value and self._grad is None:
            self._alloc_grad()
        elif not value:
            self._grad = None

        self._requires_grad = value

        # trigger re-creation of C-representation
        self.ctype = None

    def _alloc_grad(self):
        self._grad = array(
            dtype=self.dtype, shape=self.shape, strides=self.strides, device=self.device, pinned=self.pinned
        )
        self._grad.zero_()

        # trigger re-creation of C-representation
        self.ctype = None

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = array.shape[0])
        # Note: we use a shared dict for all array instances
        if array._vars is None:
            array._vars = {"shape": warp2.codegen.Var("shape", shape_t)}
        return array._vars

    def zero_(self):
        """Zeroes-out the array entires."""
        if self.is_contiguous:
            # simple memset is usually faster than generic fill
            self.device.memset(self.ptr, 0, self.size * type_size_in_bytes(self.dtype))
        else:
            self.fill_(0)

    def fill_(self, value):
        """Set all array entries to `value`

        args:
            value: The value to set every array entry to. Must be convertible to the array's ``dtype``.

        Raises:
            ValueError: If `value` cannot be converted to the array's ``dtype``.

        Examples:
            ``fill_()`` can take lists or other sequences when filling arrays of vectors or matrices.

            >>> arr = wp.zeros(2, dtype=wp.mat22)
            >>> arr.numpy()
            array([[[0., 0.],
                    [0., 0.]],
            <BLANKLINE>
                   [[0., 0.],
                    [0., 0.]]], dtype=float32)
            >>> arr.fill_([[1, 2], [3, 4]])
            >>> arr.numpy()
            array([[[1., 2.],
                    [3., 4.]],
            <BLANKLINE>
                   [[1., 2.],
                    [3., 4.]]], dtype=float32)
        """
        if self.size == 0:
            return

        # try to convert the given value to the array dtype
        try:
            if isinstance(self.dtype, warp2.codegen.Struct):
                if isinstance(value, self.dtype.cls):
                    cvalue = value.__ctype__()
                elif value == 0:
                    # allow zero-initializing structs using default constructor
                    cvalue = self.dtype().__ctype__()
                else:
                    raise ValueError(
                        f"Invalid initializer value for struct {self.dtype.cls.__name__}, expected struct instance or 0"
                    )
            elif issubclass(self.dtype, ctypes.Array):
                # vector/matrix
                cvalue = self.dtype(value)
            else:
                # scalar
                if type(value) in warp2.types.scalar_types:
                    value = value.value
                if self.dtype == float16:
                    cvalue = self.dtype._type_(float_to_half_bits(value))
                else:
                    cvalue = self.dtype._type_(value)
        except Exception as e:
            raise ValueError(f"Failed to convert the value to the array data type: {e}")

        cvalue_ptr = ctypes.pointer(cvalue)
        cvalue_size = ctypes.sizeof(cvalue)

        # prefer using memtile for contiguous arrays, because it should be faster than generic fill
        if self.is_contiguous:
            self.device.memtile(self.ptr, cvalue_ptr, cvalue_size, self.size)
        else:
            carr = self.__ctype__()
            carr_ptr = ctypes.pointer(carr)

            if self.device.is_cuda:
                warp2.context.runtime.core.array_fill_device(
                    self.device.context, carr_ptr, ARRAY_TYPE_REGULAR, cvalue_ptr, cvalue_size
                )
            else:
                warp2.context.runtime.core.array_fill_host(carr_ptr, ARRAY_TYPE_REGULAR, cvalue_ptr, cvalue_size)

    def assign(self, src):
        """Wraps ``src`` in an :class:`warp2.array` if it is not already one and copies the contents to ``self``."""
        if is_array(src):
            warp2.copy(self, src)
        else:
            warp2.copy(self, array(data=src, dtype=self.dtype, copy=False, device="cpu"))

    def numpy(self):
        """Converts the array to a :class:`numpy.ndarray` (aliasing memory through the array interface protocol)
        If the array is on the GPU, a synchronous device-to-host copy (on the CUDA default stream) will be
        automatically performed to ensure that any outstanding work is completed.
        """
        if self.ptr:
            # use the CUDA default stream for synchronous behaviour with other streams
            with warp2.ScopedStream(self.device.null_stream):
                a = self.to("cpu")
            # convert through __array_interface__
            # Note: this handles arrays of structs using `descr`, so the result will be a structured NumPy array
            return np.array(a, copy=False)
        else:
            # return an empty numpy array with the correct dtype and shape
            if isinstance(self.dtype, warp2.codegen.Struct):
                npdtype = self.dtype.numpy_dtype()
                npshape = self.shape
            elif issubclass(self.dtype, ctypes.Array):
                npdtype = warp_type_to_np_dtype[self.dtype._wp_scalar_type_]
                npshape = (*self.shape, *self.dtype._shape_)
            else:
                npdtype = warp_type_to_np_dtype[self.dtype]
                npshape = self.shape
            return np.empty(npshape, dtype=npdtype)

    def cptr(self):
        """Return a ctypes cast of the array address.

        Notes:

        #. Only CPU arrays support this method.
        #. The array must be contiguous.
        #. Accesses to this object are **not** bounds checked.
        #. For ``float16`` types, a pointer to the internal ``uint16`` representation is returned.
        """
        if not self.ptr:
            return None

        if self.device != "cpu" or not self.is_contiguous:
            raise RuntimeError(
                "Accessing array memory through a ctypes ptr is only supported for contiguous CPU arrays."
            )

        if isinstance(self.dtype, warp2.codegen.Struct):
            p = ctypes.cast(self.ptr, ctypes.POINTER(self.dtype.ctype))
        else:
            p = ctypes.cast(self.ptr, ctypes.POINTER(self.dtype._type_))

        # store backref to the underlying array to avoid it being deallocated
        p._ref = self

        return p

    def list(self):
        """Returns a flattened list of items in the array as a Python list."""
        a = self.numpy()

        if isinstance(self.dtype, warp2.codegen.Struct):
            # struct
            a = a.flatten()
            data = a.ctypes.data
            stride = a.strides[0]
            return [self.dtype.from_ptr(data + i * stride) for i in range(self.size)]
        elif issubclass(self.dtype, ctypes.Array):
            # vector/matrix - flatten, but preserve inner vector/matrix dimensions
            a = a.reshape((self.size, *self.dtype._shape_))
            data = a.ctypes.data
            stride = a.strides[0]
            return [self.dtype.from_ptr(data + i * stride) for i in range(self.size)]
        else:
            # scalar
            return list(a.flatten())

    def to(self, device):
        """Returns a Warp array with this array's data moved to the specified device, no-op if already on device."""
        device = warp2.get_device(device)
        if self.device == device:
            return self
        else:
            return warp2.clone(self, device=device)

    def flatten(self):
        """Returns a zero-copy view of the array collapsed to 1-D. Only supported for contiguous arrays."""
        if self.ndim == 1:
            return self

        if not self.is_contiguous:
            raise RuntimeError("Flattening non-contiguous arrays is unsupported.")

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=(self.size,),
            device=self.device,
            pinned=self.pinned,
            copy=False,
            owner=False,
            grad=None if self.grad is None else self.grad.flatten(),
        )

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def reshape(self, shape):
        """Returns a reshaped array. Only supported for contiguous arrays.

        Args:
            shape : An int or tuple of ints specifying the shape of the returned array.
        """
        if not self.is_contiguous:
            raise RuntimeError("Reshaping non-contiguous arrays is unsupported.")

        # convert shape to tuple
        if shape is None:
            raise RuntimeError("shape parameter is required.")
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        if len(shape) > ARRAY_MAX_DIMS:
            raise RuntimeError(
                f"Arrays may only have {ARRAY_MAX_DIMS} dimensions maximum, trying to create array with {len(shape)} dims."
            )

        # check for -1 dimension and reformat
        if -1 in shape:
            idx = self.size
            denom = 1
            minus_one_count = 0
            for i, d in enumerate(shape):
                if d == -1:
                    idx = i
                    minus_one_count += 1
                else:
                    denom *= d
            if minus_one_count > 1:
                raise RuntimeError("Cannot infer shape if more than one index is -1.")
            new_shape = list(shape)
            new_shape[idx] = int(self.size / denom)
            shape = tuple(new_shape)

        size = 1
        for d in shape:
            size *= d

        if size != self.size:
            raise RuntimeError("Reshaped array must have the same total size as the original.")

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=shape,
            strides=None,
            device=self.device,
            pinned=self.pinned,
            copy=False,
            owner=False,
            grad=None if self.grad is None else self.grad.reshape(shape),
        )

        # store back-ref to stop data being destroyed
        a._ref = self
        return a

    def view(self, dtype):
        """Returns a zero-copy view of this array's memory with a different data type.
        ``dtype`` must have the same byte size of the array's native ``dtype``.
        """
        if type_size_in_bytes(dtype) != type_size_in_bytes(self.dtype):
            raise RuntimeError("Cannot cast dtypes of unequal byte size")

        # return an alias of the array memory with different type information
        a = array(
            ptr=self.ptr,
            dtype=dtype,
            shape=self.shape,
            strides=self.strides,
            device=self.device,
            pinned=self.pinned,
            copy=False,
            owner=False,
            grad=None if self.grad is None else self.grad.view(dtype),
        )

        a._ref = self
        return a

    def contiguous(self):
        """Returns a contiguous array with this array's data. No-op if array is already contiguous."""
        if self.is_contiguous:
            return self

        a = warp2.empty_like(self)
        warp2.copy(a, self)
        return a

    def transpose(self, axes=None):
        """Returns an zero-copy view of the array with axes transposed.

        Note: The transpose operation will return an array with a non-contiguous access pattern.

        Args:
            axes (optional): Specifies the how the axes are permuted. If not specified, the axes order will be reversed.
        """
        # noop if 1d array
        if self.ndim == 1:
            return self

        if axes is None:
            # reverse the order of the axes
            axes = range(self.ndim)[::-1]
        elif len(axes) != len(self.shape):
            raise RuntimeError("Length of parameter axes must be equal in length to array shape")

        shape = []
        strides = []
        for a in axes:
            if not isinstance(a, int):
                raise RuntimeError(f"axis index {a} is not of type int")
            if a >= len(self.shape):
                raise RuntimeError(f"axis index {a} must be smaller than the number of axes in array")
            shape.append(self.shape[a])
            strides.append(self.strides[a])

        a = array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=tuple(shape),
            strides=tuple(strides),
            device=self.device,
            pinned=self.pinned,
            copy=False,
            owner=False,
            grad=None if self.grad is None else self.grad.transpose(axes=axes),
        )

        a.is_transposed = not self.is_transposed

        a._ref = self
        return a


# aliases for arrays with small dimensions
def array1d(*args, **kwargs):
    kwargs["ndim"] = 1
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=2)
def array2d(*args, **kwargs):
    kwargs["ndim"] = 2
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=3)
def array3d(*args, **kwargs):
    kwargs["ndim"] = 3
    return array(*args, **kwargs)


# equivalent to calling array(..., ndim=4)
def array4d(*args, **kwargs):
    kwargs["ndim"] = 4
    return array(*args, **kwargs)


# TODO: Rewrite so that we take only shape, not length and optional shape
def from_ptr(ptr, length, dtype=None, shape=None, device=None):
    return array(
        dtype=dtype,
        length=length,
        capacity=length * type_size_in_bytes(dtype),
        ptr=0 if ptr == 0 else ctypes.cast(ptr, ctypes.POINTER(ctypes.c_size_t)).contents.value,
        shape=shape,
        device=device,
        owner=False,
        requires_grad=False,
    )


# A base class for non-contiguous arrays, providing the implementation of common methods like
# contiguous(), to(), numpy(), list(), assign(), zero_(), and fill_().
class noncontiguous_array_base(Generic[T]):
    def __init__(self, array_type_id):
        self.type_id = array_type_id
        self.is_contiguous = False

    # return a contiguous copy
    def contiguous(self):
        a = warp2.empty_like(self)
        warp2.copy(a, self)
        return a

    # copy data from one device to another, nop if already on device
    def to(self, device):
        device = warp2.get_device(device)
        if self.device == device:
            return self
        else:
            return warp2.clone(self, device=device)

    # return a contiguous numpy copy
    def numpy(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp2.ScopedStream(self.device.null_stream):
            return self.contiguous().numpy()

    # returns a flattened list of items in the array as a Python list
    def list(self):
        # use the CUDA default stream for synchronous behaviour with other streams
        with warp2.ScopedStream(self.device.null_stream):
            return self.contiguous().list()

    # equivalent to wrapping src data in an array and copying to self
    def assign(self, src):
        if is_array(src):
            warp2.copy(self, src)
        else:
            warp2.copy(self, array(data=src, dtype=self.dtype, copy=False, device="cpu"))

    def zero_(self):
        self.fill_(0)

    def fill_(self, value):
        if self.size == 0:
            return

        # try to convert the given value to the array dtype
        try:
            if isinstance(self.dtype, warp2.codegen.Struct):
                if isinstance(value, self.dtype.cls):
                    cvalue = value.__ctype__()
                elif value == 0:
                    # allow zero-initializing structs using default constructor
                    cvalue = self.dtype().__ctype__()
                else:
                    raise ValueError(
                        f"Invalid initializer value for struct {self.dtype.cls.__name__}, expected struct instance or 0"
                    )
            elif issubclass(self.dtype, ctypes.Array):
                # vector/matrix
                cvalue = self.dtype(value)
            else:
                # scalar
                if type(value) in warp2.types.scalar_types:
                    value = value.value
                if self.dtype == float16:
                    cvalue = self.dtype._type_(float_to_half_bits(value))
                else:
                    cvalue = self.dtype._type_(value)
        except Exception as e:
            raise ValueError(f"Failed to convert the value to the array data type: {e}")

        cvalue_ptr = ctypes.pointer(cvalue)
        cvalue_size = ctypes.sizeof(cvalue)

        ctype = self.__ctype__()
        ctype_ptr = ctypes.pointer(ctype)

        if self.device.is_cuda:
            warp2.context.runtime.core.array_fill_device(
                self.device.context, ctype_ptr, self.type_id, cvalue_ptr, cvalue_size
            )
        else:
            warp2.context.runtime.core.array_fill_host(ctype_ptr, self.type_id, cvalue_ptr, cvalue_size)


# helper to check index array properties
def check_index_array(indices, expected_device):
    if not isinstance(indices, array):
        raise ValueError(f"Indices must be a Warp array, got {type(indices)}")
    if indices.ndim != 1:
        raise ValueError(f"Index array must be one-dimensional, got {indices.ndim}")
    if indices.dtype != int32:
        raise ValueError(f"Index array must use int32, got dtype {indices.dtype}")
    if indices.device != expected_device:
        raise ValueError(f"Index array device ({indices.device} does not match data array device ({expected_device}))")


class indexedarray(noncontiguous_array_base[T]):
    # member attributes available during code-gen (e.g.: d = arr.shape[0])
    # (initialized when needed)
    _vars = None

    def __init__(self, data: array = None, indices: Union[array, List[array]] = None, dtype=None, ndim=None):
        super().__init__(ARRAY_TYPE_INDEXED)

        # canonicalize types
        if dtype is not None:
            if dtype == int:
                dtype = int32
            elif dtype == float:
                dtype = float32

        self.data = data
        self.indices = [None] * ARRAY_MAX_DIMS

        if data is not None:
            if not isinstance(data, array):
                raise ValueError("Indexed array data must be a Warp array")
            if dtype is not None and dtype != data.dtype:
                raise ValueError(f"Requested dtype ({dtype}) does not match dtype of data array ({data.dtype})")
            if ndim is not None and ndim != data.ndim:
                raise ValueError(
                    f"Requested dimensionality ({ndim}) does not match dimensionality of data array ({data.ndim})"
                )

            self.dtype = data.dtype
            self.ndim = data.ndim
            self.device = data.device
            self.pinned = data.pinned

            # determine shape from original data shape and index counts
            shape = list(data.shape)

            if indices is not None:
                if isinstance(indices, (list, tuple)):
                    if len(indices) > self.ndim:
                        raise ValueError(
                            f"Number of indices provided ({len(indices)}) exceeds number of dimensions ({self.ndim})"
                        )

                    for i in range(len(indices)):
                        if indices[i] is not None:
                            check_index_array(indices[i], data.device)
                            self.indices[i] = indices[i]
                            shape[i] = len(indices[i])

                elif isinstance(indices, array):
                    # only a single index array was provided
                    check_index_array(indices, data.device)
                    self.indices[0] = indices
                    shape[0] = len(indices)

                else:
                    raise ValueError("Indices must be a single Warp array or a list of Warp arrays")

            self.shape = tuple(shape)

        else:
            # allow empty indexedarrays in type annotations
            self.dtype = dtype
            self.ndim = ndim or 1
            self.device = None
            self.pinned = False
            self.shape = (0,) * self.ndim

        # update size (num elements)
        self.size = 1
        for d in self.shape:
            self.size *= d

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        if self.device is None:
            # type annotation
            return f"indexedarray{self.dtype}"
        else:
            return str(self.numpy())

    # construct a C-representation of the array for passing to kernels
    def __ctype__(self):
        return indexedarray_t(self.data, self.indices, self.shape)

    @property
    def vars(self):
        # member attributes available during code-gen (e.g.: d = arr.shape[0])
        # Note: we use a shared dict for all indexedarray instances
        if indexedarray._vars is None:
            indexedarray._vars = {"shape": warp2.codegen.Var("shape", shape_t)}
        return indexedarray._vars


# aliases for indexedarrays with small dimensions
def indexedarray1d(*args, **kwargs):
    kwargs["ndim"] = 1
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=2)
def indexedarray2d(*args, **kwargs):
    kwargs["ndim"] = 2
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=3)
def indexedarray3d(*args, **kwargs):
    kwargs["ndim"] = 3
    return indexedarray(*args, **kwargs)


# equivalent to calling indexedarray(..., ndim=4)
def indexedarray4d(*args, **kwargs):
    kwargs["ndim"] = 4
    return indexedarray(*args, **kwargs)


from warp2.fabric import fabricarray, indexedfabricarray  # noqa: E402

array_types = (array, indexedarray, fabricarray, indexedfabricarray)


def array_type_id(a):
    if isinstance(a, array):
        return ARRAY_TYPE_REGULAR
    elif isinstance(a, indexedarray):
        return ARRAY_TYPE_INDEXED
    elif isinstance(a, fabricarray):
        return ARRAY_TYPE_FABRIC
    elif isinstance(a, indexedfabricarray):
        return ARRAY_TYPE_FABRIC_INDEXED
    else:
        raise ValueError("Invalid array type")


def type_is_generic(t):
    if t in (Any, Scalar, Float, Int):
        return True
    elif is_array(t):
        return type_is_generic(t.dtype)
    elif hasattr(t, "_wp_scalar_type_"):
        # vector/matrix type, check if dtype is generic
        if type_is_generic(t._wp_scalar_type_):
            return True
        # check if any dimension is generic
        for d in t._shape_:
            if d == 0:
                return True
    else:
        return False


def type_is_generic_scalar(t):
    return t in (Scalar, Float, Int)


def type_matches_template(arg_type, template_type):
    """Check if an argument type matches a template.

    This function is used to test whether the arguments passed to a generic @wp.kernel or @wp.func
    match the template type annotations.  The template_type can be generic, but the arg_type must be concrete.
    """

    # canonicalize types
    arg_type = type_to_warp(arg_type)
    template_type = type_to_warp(template_type)

    # arg type must be concrete
    if type_is_generic(arg_type):
        return False

    # if template type is not generic, the argument type must match exactly
    if not type_is_generic(template_type):
        return types_equal(arg_type, template_type)

    # template type is generic, check that the argument type matches
    if template_type == Any:
        return True
    elif is_array(template_type):
        # ensure the argument type is a non-generic array with matching dtype and dimensionality
        if type(arg_type) is not type(template_type):
            return False
        if not type_matches_template(arg_type.dtype, template_type.dtype):
            return False
        if arg_type.ndim != template_type.ndim:
            return False
    elif template_type == Float:
        return arg_type in float_types
    elif template_type == Int:
        return arg_type in int_types
    elif template_type == Scalar:
        return arg_type in scalar_types
    elif hasattr(template_type, "_wp_scalar_type_"):
        # vector/matrix type
        if not hasattr(arg_type, "_wp_scalar_type_"):
            return False
        if not type_matches_template(arg_type._wp_scalar_type_, template_type._wp_scalar_type_):
            return False
        ndim = len(template_type._shape_)
        if len(arg_type._shape_) != ndim:
            return False
        # for any non-generic dimensions, make sure they match
        for i in range(ndim):
            if template_type._shape_[i] != 0 and arg_type._shape_[i] != template_type._shape_[i]:
                return False

    return True


def infer_argument_types(args, template_types, arg_names=None):
    """Resolve argument types with the given list of template types."""

    if len(args) != len(template_types):
        raise RuntimeError("Number of arguments must match number of template types.")

    arg_types = []

    for i in range(len(args)):
        arg = args[i]
        arg_type = type(arg)
        arg_name = arg_names[i] if arg_names else str(i)
        if arg_type in warp2.types.array_types:
            arg_types.append(arg_type(dtype=arg.dtype, ndim=arg.ndim))
        elif arg_type in warp2.types.scalar_types:
            arg_types.append(arg_type)
        elif arg_type in [int, float]:
            # canonicalize type
            arg_types.append(warp2.types.type_to_warp(arg_type))
        elif hasattr(arg_type, "_wp_scalar_type_"):
            # vector/matrix type
            arg_types.append(arg_type)
        elif issubclass(arg_type, warp2.codegen.StructInstance):
            # a struct
            arg_types.append(arg._cls)
        # elif arg_type in [warp2.types.launch_bounds_t, warp2.types.shape_t, warp2.types.range_t]:
        #     arg_types.append(arg_type)
        # elif arg_type in [warp2.hash_grid_query_t, warp2.mesh_query_aabb_t, warp2.bvh_query_t]:
        #     arg_types.append(arg_type)
        elif arg is None:
            # allow passing None for arrays
            t = template_types[i]
            if warp2.types.is_array(t):
                arg_types.append(type(t)(dtype=t.dtype, ndim=t.ndim))
            else:
                raise TypeError(f"Unable to infer the type of argument '{arg_name}', got None")
        else:
            # TODO: attempt to figure out if it's a vector/matrix type given as a numpy array, list, etc.
            raise TypeError(f"Unable to infer the type of argument '{arg_name}', got {arg_type}")

    return arg_types


simple_type_codes = {
    int: "i4",
    float: "f4",
    builtins.bool: "b",
    bool: "b",
    str: "str",  # accepted by print()
    int8: "i1",
    int16: "i2",
    int32: "i4",
    int64: "i8",
    uint8: "u1",
    uint16: "u2",
    uint32: "u4",
    uint64: "u8",
    float16: "f2",
    float32: "f4",
    float64: "f8",
    shape_t: "sh",
    range_t: "rg",
    launch_bounds_t: "lb",
    hash_grid_query_t: "hgq",
    mesh_query_aabb_t: "mqa",
    bvh_query_t: "bvhq",
}


def get_type_code(arg_type):
    if arg_type == Any:
        # special case for generics
        # note: since Python 3.11 Any is a type, so we check for it first
        return "?"
    elif isinstance(arg_type, type):
        if hasattr(arg_type, "_wp_scalar_type_"):
            # vector/matrix type
            dtype_code = get_type_code(arg_type._wp_scalar_type_)
            # check for "special" vector/matrix subtypes
            if hasattr(arg_type, "_wp_generic_type_str_"):
                type_str = arg_type._wp_generic_type_str_
                if type_str == "quat_t":
                    return f"q{dtype_code}"
                elif type_str == "transform_t":
                    return f"t{dtype_code}"
                # elif type_str == "spatial_vector_t":
                #     return f"sv{dtype_code}"
                # elif type_str == "spatial_matrix_t":
                #     return f"sm{dtype_code}"
            # generic vector/matrix
            ndim = len(arg_type._shape_)
            if ndim == 1:
                dim_code = "?" if arg_type._shape_[0] == 0 else str(arg_type._shape_[0])
                return f"v{dim_code}{dtype_code}"
            elif ndim == 2:
                dim_code0 = "?" if arg_type._shape_[0] == 0 else str(arg_type._shape_[0])
                dim_code1 = "?" if arg_type._shape_[1] == 0 else str(arg_type._shape_[1])
                return f"m{dim_code0}{dim_code1}{dtype_code}"
            else:
                raise TypeError("Invalid vector/matrix dimensionality")
        else:
            # simple type
            type_code = simple_type_codes.get(arg_type)
            if type_code is not None:
                return type_code
            else:
                raise TypeError(f"Unrecognized type '{arg_type}'")
    elif isinstance(arg_type, array):
        return f"a{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, indexedarray):
        return f"ia{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, fabricarray):
        return f"fa{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, indexedfabricarray):
        return f"ifa{arg_type.ndim}{get_type_code(arg_type.dtype)}"
    elif isinstance(arg_type, warp2.codegen.Struct):
        return warp2.codegen.make_full_qualified_name(arg_type.cls)
    elif arg_type == Scalar:
        # generic scalar type
        return "s?"
    elif arg_type == Float:
        # generic float
        return "f?"
    elif arg_type == Int:
        # generic int
        return "i?"
    elif isinstance(arg_type, Callable):
        # TODO: elaborate on Callable type?
        return "c"
    else:
        raise TypeError(f"Unrecognized type '{arg_type}'")


def get_signature(arg_types, func_name=None, arg_names=None):
    type_codes = []
    for i, arg_type in enumerate(arg_types):
        try:
            type_codes.append(get_type_code(arg_type))
        except Exception as e:
            if arg_names is not None:
                arg_str = f"'{arg_names[i]}'"
            else:
                arg_str = str(i + 1)
            if func_name is not None:
                func_str = f" of function {func_name}"
            else:
                func_str = ""
            raise RuntimeError(f"Failed to determine type code for argument {arg_str}{func_str}: {e}")

    return "_".join(type_codes)


def is_generic_signature(sig):
    return "?" in sig
