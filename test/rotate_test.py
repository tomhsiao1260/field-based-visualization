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

    # pc = pc_pad[1:-1,  1:-1].copy()
    pc = pc_pad[1:-1,  :-2].copy()
    pc += pc_pad[1:-1,   2:]
    pc += pc_pad[:-2, 1:-1]
    pc += pc_pad[2:, 1:-1]
    pc /= 4

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

def update_electrode_condition(x, y, theta):
    x_center, y_center = 50, 50

    x_shifted = x - x_center
    y_shifted = y - y_center

    x_rotated = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted
    y_rotated = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    x_rotated += x_center
    y_rotated += y_center

    condition = (30 <= x_rotated) & (x_rotated < 70) & (45 <= y_rotated) & (y_rotated < 55)

    return condition

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    boundary_mask = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape

    y, x = np.ogrid[:h, :w]

    condition_electrode = (30 < x) & (x < 70) & (45 < y) & (y < 55)
    boundary_mask[condition_electrode] = True

    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[(0 <= x) & (x < 100) & (0 <= y) & (y < 50)] = 255
    potential[(0 <= x) & (x < 100) & (50 <= y) & (y < 100)] = 0
    # potential[(100 <= x + y)] = 255
    # potential[(100 > x + y)] = 0
    potential[boundary_mask > 0] = 128

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Potential")
    axes[0].imshow(potential, cmap=cmap)

    # update potential
    plt.ion()

    for i in range(5000):
        if (i > 0):
            theta = np.pi / 1000 * 500
            condition = update_electrode_condition(x, y, theta)

            boundary_mask[:, :] = False
            boundary_mask[condition] = True
            pc[boundary_mask] = 128

        pc = update_potential(potential, axes, cmap)
        pc[boundary_mask] = potential[boundary_mask]
        potential = pc

        # if (i%1 == 0):
        if (i%100 == 0 and i > 200):
            print(i)
            axes[0].imshow(potential, cmap=cmap)
            axes[1].imshow(potential, cmap="gray")

            plt.pause(0.01)

    plt.ioff()

    plt.show()
