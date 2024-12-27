import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

# see config_template.py & generate a config.py file (version 0)
from config import electrode_label_level_pairs
from config import volume_dir, electrode_dir

def down_sampling(array, rescale=(1,1,1), mean=True):
    rz, ry, rx = rescale
    nz, ny, nx = array.shape

    tz = (nz // rz) * rz
    ty = (ny // ry) * ry
    tx = (nx // rx) * rx
    trimmed_array = array[:tz, :ty, :tx]

    if (mean):
        downscaled = trimmed_array.reshape(
            tz // rz, rz,
            ty // ry, ry,
            tx // rx, rx
        ).mean(axis=(1, 3, 5)).astype(array.dtype)
    else:
        downscaled = np.max(
            trimmed_array.reshape(
                tz // rz, rz,
                ty // ry, ry,
                tx // rx, rx
        ), axis=(1, 3, 5)).astype(array.dtype)

    return downscaled

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1].copy()
    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    static = (top == -1) & (bot == -1) & (left == -1) & (right == -1)

    top[top == -1] = center[top == -1]
    bot[bot == -1] = center[bot == -1]
    left[left == -1] = center[left == -1]
    right[right == -1] = center[right == -1]

    pc = (top + bot + left + right) / 4

    # pc[pc > 255] = 255
    # pc[pc < 0] = 0
    pc[static] = -1

    # pc[0, :] = pc[1, :]
    # pc[-1, :] = pc[-2, :]
    # pc[:, 0] = pc[:, 1]
    # pc[:, -1] = pc[:, -2]

    return pc

def fix_gradient(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1].copy()
    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    # h, w = potential.shape
    # y, x = np.ogrid[:h, :w]

    # targetX, targetY = (0, h)
    # dirX, dirY = targetX - x, targetY - y

    decay = (top == -1) & (bot != -1)
    grow = (top != -1) & (bot == -1)
    static = (top == -1) & (bot == -1) & (left == -1) & (right == -1)

    s = 3

    center[grow & (center == -1)] = top[grow & (center == -1)] + s
    top[grow & (top == -1)] = top[grow & (top == -1)] + s
    bot[grow & (bot == -1)] = top[grow & (bot == -1)] + s
    left[grow & (left == -1)] = top[grow & (left == -1)] + s
    right[grow & (right == -1)] = top[grow & (right == -1)] + s

    center[decay & (center == -1)] = bot[decay & (center == -1)] - s
    top[decay & (top == -1)] = bot[decay & (top == -1)] - s
    bot[decay & (bot == -1)] = bot[decay & (bot == -1)] - s
    left[decay & (left == -1)] = bot[decay & (left == -1)] - s
    right[decay & (right == -1)] = bot[decay & (right == -1)] - s

    top[top == -1] = center[top == -1]
    bot[bot == -1] = center[bot == -1]
    left[left == -1] = center[left == -1]
    right[right == -1] = center[right == -1]

    pc = (top + bot + left + right) / 4

    # grad_top = center - top
    # grad_bot = center - bot
    # grad_left = center - left
    # grad_right = center - right

    # max_gradient = 1

    # # 限制梯度：若梯度超過 max_gradient，則進行修正
    # grad_top = np.where(grad_top > max_gradient, max_gradient, grad_top)
    # grad_bot = np.where(grad_bot > max_gradient, max_gradient, grad_bot)
    # grad_left = np.where(grad_left > max_gradient, max_gradient, grad_left)
    # grad_right = np.where(grad_right > max_gradient, max_gradient, grad_right)

    # # 修正中心值：根據調整後的梯度進行平滑更新
    # pc = center - 1.0 * (
    #     (grad_top + grad_bot + grad_left + grad_right) / 4
    # )

    # pc[pc > 255] = 255
    # pc[pc < 0] = 0
    pc[static] = -1

    pc[0, :] = potential[0, :]
    pc[-1, :] = potential[-1, :]
    pc[:, 0] = potential[:, 0]
    pc[:, -1] = potential[:, -1]

    return pc

def fix_boundary(potential):
    s = 0.9535

    pc = potential.copy()

    pc[0, :] = s * (2 * pc[1, :] - pc[2, :]) + (1-s) * pc[1, :]
    pc[-1, :] = s * (2 * pc[-2, :] - pc[-3, :]) + (1-s) * pc[-2, :]
    pc[:, 0] = s * (2 * pc[:, 1] - pc[:, 2]) + (1-s) * pc[:, 1]
    pc[:, -1] = s * (2 * pc[:, -2] - pc[:, -3]) + (1-s) * pc[:, -2]

    # pc[0, :] = pc[1, :]
    # pc[-1, :] = pc[-2, :]
    # pc[:, 0] = pc[:, 1]
    # pc[:, -1] = pc[:, -2]

    pc[potential == -1] = potential[potential == -1]

    return pc

def update_electrode_condition(mask, center=(0,0), theta=0):
    xc, yc = center
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]

    x_shifted = x - xc
    y_shifted = y - yc

    x_rotated = np.cos(theta) * x_shifted - np.sin(theta) * y_shifted
    y_rotated = np.sin(theta) * x_shifted + np.cos(theta) * y_shifted

    condition = (-w//4 <= x_rotated) & (x_rotated < w//4) & (-2 <= y_rotated) & (y_rotated < 2)

    return condition

if __name__ == "__main__":
    ### path & params
    rescale = (1, 3, 3)

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    d, h, w = electrode.shape
    electrode = electrode[d//2, :, :][np.newaxis, ...]
    # electrode = electrode[:1, :, :]
    electrode = down_sampling(electrode, rescale, False)
    d, h, w = electrode.shape

    electrode_temp = np.zeros_like(electrode)

    for label, level in electrode_label_level_pairs:
        print('Processing electrode:', label)
        mask_label = (electrode == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_temp[z][skeleton] = label

    # # testing label
    # for z in range(d):
    #     center = (w//2, h//2)
    #     theta = np.pi / 1000 * 250
    #     condition = update_electrode_condition(electrode_temp[z], center, theta)
    #     electrode_temp[z][condition] = 10

    electrode = electrode_temp

    ### plot init
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    potential = np.zeros_like(electrode, dtype=float)
    potential[:, :, :] = -1

    # rescale again
    rescale = (1, 2, 2)
    potential = down_sampling(potential, rescale)
    electrode = down_sampling(electrode, rescale, False)

    d, h, w = potential.shape

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Electrode")
    axes[0].contour(electrode[d//2, ::-1, :] * 255, colors='blue', linewidths=0.5)
    # axes[0].contour(electrode[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    axes[1].set_title("Potential")
    axes[1].imshow(potential[d//2, :, :], cmap=cmap)

    # update potential
    plt.ion()

    for i in range(10):
        # electrodes should remain constant
        for label, level in electrode_label_level_pairs:
            potential[electrode == label] = level * 255
        potential[electrode == 10] = 128

        if -1 in potential[d//2, 1:-1, 1:-1]:
            pc = fix_gradient(potential[d//2, :, :])
        else:
            pc = update_potential(potential[d//2, :, :])
            pc = fix_boundary(pc)

        potential[d//2, :, :] = pc

        a = (potential == -1).astype(bool)[d//2, :, :]

        if (i%100 == 0 and i > 100):
            print(i)
            axes[2].imshow(potential[d//2, :, :], cmap=cmap)
            axes[3].imshow(potential[d//2, :, :], cmap="gray")
            axes[4].imshow(a, cmap="gray")

            plt.pause(0.01)

        if (i%3 == 0 and i < 100):
            print(i)
            axes[1].imshow(potential[d//2, :, :], cmap=cmap)
            axes[3].imshow(potential[d//2, :, :], cmap="gray")
            axes[4].imshow(a, cmap="gray")

            plt.pause(0.01)

    plt.ioff()

    plt.show()
