import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def update_trigger(potential, trigger, axes, cmap):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')
    tc_pad = np.pad(trigger, pad_width=1, mode='edge')

    tc = tc_pad[1:-1,  1:-1].copy()
    pc = pc_pad[1:-1,  1:-1].copy()

    count_zero_neighbors = (
        (tc_pad[:-2, 1:-1] == 0).astype(int) +  # 上
        (tc_pad[2:, 1:-1] == 0).astype(int) + # 下
        (tc_pad[1:-1, :-2] == 0).astype(int) +  # 左
        (tc_pad[1:-1, 2:] == 0).astype(int)     # 右
    )

    diff = pc_pad[:-2, 1:-1].copy() * (tc_pad[:-2, 1:-1] == 0).astype(int)
    diff += pc_pad[2:, 1:-1] * (tc_pad[2:, 1:-1] == 0).astype(int)
    diff += pc_pad[1:-1, :-2] * (tc_pad[1:-1, :-2] == 0).astype(int)
    diff += pc_pad[1:-1,  2:] * (tc_pad[1:-1,  2:] == 0).astype(int)
    diff /= count_zero_neighbors

    # pc[tc == 0] = diff[tc == 0]
    pc[tc > 0] = (diff + 10)[tc > 0]
    pc[tc < 0] = (diff - 10)[tc < 0]

    pc[count_zero_neighbors == 0] = potential[count_zero_neighbors == 0]

    mask = (
        (tc_pad[:-2, 1:-1] == 0) |  # 上
        (tc_pad[2:, 1:-1] == 0) |   # 下
        (tc_pad[1:-1, :-2] == 0) |  # 左
        (tc_pad[1:-1, 2:] == 0)     # 右
    )

    tc[mask] = 0

    return pc, tc

def update_potential(potential, axes, cmap):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    pc = pc_pad[1:-1,  1:-1].copy()
    pc += pc_pad[1:-1,  :-2]
    pc += pc_pad[1:-1,   2:]
    pc += pc_pad[:-2, 1:-1]
    pc += pc_pad[2:, 1:-1]
    pc /= 5

    s = 0.9535

    pc[0, :] = s * (2 * pc[1, :] - pc[2, :]) + (1-s) * pc[1, :]
    pc[-1, :] = s * (2 * pc[-2, :] - pc[-3, :]) + (1-s) * pc[-2, :]
    pc[:, 0] = s * (2 * pc[:, 1] - pc[:, 2]) + (1-s) * pc[:, 1]
    pc[:, -1] = s * (2 * pc[:, -2] - pc[:, -3]) + (1-s) * pc[:, -2]

    # pc[0, :] = pc[1, :]
    # pc[-1, :] = pc[-2, :]
    # pc[:, 0] = pc[:, 1]
    # pc[:, -1] = pc[:, -2]

    return pc


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    boundary_mask = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape

    y, x = np.ogrid[:h, :w]

    condition_electrode = (30 < x) & (x < 70) & (45 < y) & (y < 55)
    boundary_mask[condition_electrode] = True

    trigger = np.zeros_like(boundary_mask, dtype=int)
    condition_top = (0 <= x) & (x < 100) & (y <= 50)
    condition_bot = (0 <= x) & (x < 100) & (50 < y)
    trigger[condition_top] = 1
    trigger[condition_bot] = -1
    trigger[condition_electrode] = 0

    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[boundary_mask > 0] = 128

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Potential")
    axes[0].imshow(potential, cmap=cmap)
    axes[1].set_title("Trigger")
    axes[1].imshow(trigger, cmap=cmap)

    # update potential
    plt.ion()

    for i in range(300):
        if (i < 50): pc, tc = update_trigger(potential, trigger, axes, cmap)
        if (i > 50): pc = update_potential(potential, axes, cmap)
        pc[boundary_mask] = potential[boundary_mask]

        potential = pc
        trigger = tc

        if (i%1 == 0):
        # if (i%100 == 0 and i > 200):
            print(i)
            axes[0].imshow(potential, cmap=cmap)
            axes[1].imshow(trigger, cmap=cmap)
            axes[2].imshow(potential, cmap="gray")

            plt.pause(0.5)

    plt.ioff()

    plt.show()
