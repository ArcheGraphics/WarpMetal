from warp import kernel, floor


@kernel
def sample_vel(x: float, y: float):
    lx = int(floor(x))
    ly = int(floor(y))

    tx = x - float(lx)
    ty = y - float(ly)


@kernel
def sample_pos(x: float, y: float):
    lx = int(floor(x))
    ly = int(floor(y))

    tx = x - float(lx)
    ty = y - float(ly)


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    module = sample_vel.module
    module.codegen()
