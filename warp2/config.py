#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

version = "1.0.0-beta.5"

verify_fp = False  # verify inputs and outputs are finite after each launch
verify_cuda = False  # if true will check CUDA errors after each kernel launch / memory operation
print_launches = False  # if true will print out launch information

mode = "release"
verbose = False  # print extra informative messages
quiet = False  # suppress all output except errors and warnings

host_compiler = None  # user can specify host compiler here, otherwise will attempt to find one automatically

cache_kernels = True
kernel_cache_dir = None  # path to kernel cache directory, if None a default path will be used

cuda_output = (
    None  # preferred CUDA output format for kernels ("ptx" or "cubin"), determined automatically if unspecified
)

ptx_target_arch = 70  # target architecture for PTX generation, defaults to the lowest architecture that supports all of Warp's features

enable_backward = True  # whether to compiler the backward passes of the kernels

llvm_cuda = False  # use Clang/LLVM instead of NVRTC to compile CUDA
