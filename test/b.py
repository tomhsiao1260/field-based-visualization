import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def update_potential(potential, axes, cmap, m=False):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    grow = (top == -1) & (bot != -1)
    decay = (top != -1) & (bot == -1)
    static = (top == -1) & (bot == -1) & (left == -1) & (right == -1)

    top[grow & (top == -1)] = 255
    bot[grow & (bot == -1)] = 255
    left[grow & (left == -1)] = 255
    right[grow & (right == -1)] = 255

    top[decay & (top == -1)] = 0
    bot[decay & (bot == -1)] = 0
    left[decay & (left == -1)] = 0
    right[decay & (right == -1)] = 0

    pc = (top + bot + left + right) / 4
    pc[static] = -1

    s = 0.9535

    if m:
        pc[0, :] = s * (2 * pc[1, :] - pc[2, :]) + (1-s) * pc[1, :]
        pc[-1, :] = s * (2 * pc[-2, :] - pc[-3, :]) + (1-s) * pc[-2, :]
        pc[:, 0] = s * (2 * pc[:, 1] - pc[:, 2]) + (1-s) * pc[:, 1]
        pc[:, -1] = s * (2 * pc[:, -2] - pc[:, -3]) + (1-s) * pc[:, -2]
    else:
        pc[0, :] = pc[1, :]
        pc[-1, :] = pc[-2, :]
        pc[:, 0] = pc[:, 1]
        pc[:, -1] = pc[:, -2]

    return pc

def update_electrode_condition(x, y, center=(0,0), theta=0):
    # _, w = x.shape
    # h, _ = y.shape
    xc, yc = center

    x_shifted = x - xc
    y_shifted = y - yc

    x_rotated = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted
    y_rotated = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    condition = (-40 <= x_rotated) & (x_rotated < 40) & (-2 <= y_rotated) & (y_rotated < 2)

    return condition

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    boundary_mask = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape
    y, x = np.ogrid[:h, :w]

    center = (0, 20)
    theta = np.pi / 1000 * 0
    condition_0 = update_electrode_condition(x, y, center, theta)

    center = (100, 80)
    theta = np.pi / 1000 * 0
    condition_1 = update_electrode_condition(x, y, center, theta)

    boundary_mask[condition_0] = True
    boundary_mask[condition_1] = True

    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[:, :] = -1
    potential[condition_0] = 200
    potential[condition_1] = 50

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Potential")
    axes[0].imshow(potential, cmap=cmap)

    # update potential
    plt.ion()

    for i in range(5000):
        m = False
        if (i>1000): m = True
        pc = update_potential(potential, axes, cmap, m)
        pc[boundary_mask] = potential[boundary_mask]
        potential = pc

        a = (potential == -1).astype(bool)

        # if (i%1 == 0):
        if (i%100 == 0 and i > 200):
            print(i)
            axes[0].imshow(potential, cmap=cmap)
            axes[1].imshow(potential, cmap="gray")
            axes[2].imshow(a, cmap="gray")

            plt.pause(0.1)

    plt.ioff()

    plt.show()
