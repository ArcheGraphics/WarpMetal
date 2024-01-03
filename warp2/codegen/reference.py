#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

class Reference:
    def __init__(self, value_type):
        self.value_type = value_type


def is_reference(type):
    return isinstance(type, Reference)


def strip_reference(arg):
    if is_reference(arg):
        return arg.value_type
    else:
        return arg
