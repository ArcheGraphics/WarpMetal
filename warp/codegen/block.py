#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

class Block:
    # Represents a basic block of instructions, e.g.: list
    # of straight line instructions inside a for-loop or conditional

    def __init__(self):
        # list of statements inside this block
        self.body_forward = []
        self.body_replay = []
        self.body_reverse = []

        # list of vars declared in this block
        self.vars = []
