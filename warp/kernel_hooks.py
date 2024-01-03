#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

class KernelHooks:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward
