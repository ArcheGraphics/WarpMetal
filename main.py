from warp2.module.context import func, kernel


@kernel
def sample_vel(x: float, y: float):
    lx = int(x)
    ly = int(y)

    tx = x - float(lx)
    ty = y - float(ly)

    return tx


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    module = sample_vel.module
    module.codegen()
