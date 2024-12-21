import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1].copy()
    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    pc = (top + bot + left + right) / 4

    return pc

def fix_boundary(potential):
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

def fix_gradient(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1].copy()
    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    # local_max = np.maximum.reduce([top, bot, left, right])
    # local_min = np.minimum.reduce([top, bot, left, right])
    # gradient_range = local_max - local_min
    # pc = np.where(gradient_range > 1, pc, center)

    grad_top = center - top
    grad_bot = center - bot
    grad_left = center - left
    grad_right = center - right

    max_gradient = 1

    # 限制梯度：若梯度超過 max_gradient，則進行修正
    grad_top = np.where(grad_top > max_gradient, max_gradient, grad_top)
    grad_bot = np.where(grad_bot > max_gradient, max_gradient, grad_bot)
    grad_left = np.where(grad_left > max_gradient, max_gradient, grad_left)
    grad_right = np.where(grad_right > max_gradient, max_gradient, grad_right)

    # 修正中心值：根據調整後的梯度進行平滑更新
    pc = center - 0.99 * (
        (grad_top + grad_bot + grad_left + grad_right) / 4
    )

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
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    boundary_mask = np.zeros((100, 100), dtype=bool)
    boundary_mask2 = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape

    theta = np.pi / 1000 * 0
    condition = update_electrode_condition(boundary_mask, (w//2, h//2), theta)
    boundary_mask[condition] = True
    theta = np.pi / 1000 * 0
    condition = update_electrode_condition(boundary_mask, (w//2, 3*h//4), theta)
    boundary_mask2[condition] = True

    y, x = np.ogrid[:h, :w]
    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[(0 <= x) & (x < 100) & (0 <= y) & (y < 50)] = 255
    potential[(0 <= x) & (x < 100) & (50 <= y) & (y < 100)] = 0

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Potential")
    axes[0].imshow(potential, cmap=cmap)

    # update potential
    plt.ion()

    for i in range(1000):
        # theta = np.pi / 1000 * i
        # condition = update_electrode_condition(boundary_mask, (w//2, h//2), theta)
        # boundary_mask[:,:] = False
        # boundary_mask[condition] = True

        potential[boundary_mask] = 255-50
        # potential[boundary_mask2] = 50

        # pc = update_potential(potential)

        if(i>-1): pc = fix_gradient(potential)

        pc = fix_boundary(pc)

        potential = pc

        if (i%10 == 0):
        # if (i%100 == 0 and i > 200):
            print(i)
            axes[0].imshow(potential, cmap=cmap)
            axes[1].imshow(potential, cmap="gray")

            plt.pause(0.01)

    plt.ioff()

    plt.show()
