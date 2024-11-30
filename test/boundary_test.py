import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def update_potential(potential, axes, cmap):
    # pc_pad = np.pad(potential, pad_width=1, mode='reflect')
    pc_pad = np.pad(potential, pad_width=1, mode='edge')
    # pc_pad = np.pad(potential, pad_width=1, mode='constant', constant_values=0)

    # pc_pad[0, :] = pc_pad[1, :].copy()
    # pc_pad[-1, :] = pc_pad[-2, :].copy()
    # pc_pad[:, 0] = pc_pad[:, 1].copy()
    # pc_pad[:, -1] = pc_pad[:, -2].copy()

    # boundary_mask = np.zeros_like(pc_pad, dtype=bool)
    # boundary_mask[1:-1, 1:-1] = True
    # boundary_mask[2:-2, 2:-2] = False

    # pc_pad = update_potential_level(pc_pad, boundary_mask, axes, cmap)

    # dy, dx = np.gradient(pc_pad)

    pc = pc_pad[1:-1,  1:-1].copy()
    pc += pc_pad[1:-1,  :-2]
    pc += pc_pad[1:-1,   2:]
    pc += pc_pad[:-2, 1:-1]
    pc += pc_pad[2:, 1:-1]
    pc /= 5

    pc[0, :] = pc[1, :]
    pc[-1, :] = pc[-2, :]
    pc[:, 0] = pc[:, 1]
    pc[:, -1] = pc[:, -2]

    return pc

def update_potential_level(potential, boundary, axes, cmap):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')
    bc_pad = np.pad(boundary, pad_width=1, mode='constant', constant_values=0)
    pc = potential.copy()

    p_avg  = pc_pad[1:-1,  :-2].copy()
    p_avg += pc_pad[1:-1,   2:]
    p_avg += pc_pad[:-2, 1:-1]
    p_avg += pc_pad[2:, 1:-1]
    p_avg /= 4

    # p_avg[boundary > 0] = 100
    p_avg[boundary > 0] -= pc[boundary > 0]
    p_avg[boundary == 0] = 0

    axes[1].imshow(boundary, cmap=cmap)
    axes[2].imshow(p_avg, cmap='gray')

    # print('avg ', p_avg)

    p_avg = np.pad(p_avg, pad_width=1, mode='constant', constant_values=0)

    bc = p_avg[1:-1,  :-2].copy()
    bc += p_avg[1:-1,   2:]
    bc += p_avg[:-2, 1:-1]
    bc += p_avg[2:, 1:-1]

    axes[3].imshow(bc, cmap='gray')

    counts = np.zeros_like(potential)
    counts += (bc_pad[1:-1,  :-2] > 0)
    counts += (bc_pad[1:-1,  2:] > 0)
    counts += (bc_pad[:-2, 1:-1] > 0)
    counts += (bc_pad[2:, 1:-1] > 0)

    nonzero = counts > 0
    pc[nonzero] += bc[nonzero] / counts[nonzero] * 0.3
    pc[~nonzero] = potential[~nonzero]
    pc[boundary > 0] = potential[boundary > 0]

    axes[4].imshow(dy, cmap='gray')
    axes[5].imshow(dx, cmap='gray')

    return pc

if __name__ == '__main__':
    # plot init
    fig, axes = plt.subplots(1, 6, figsize=(10, 3))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # mask
    boundary_mask = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape

    y, x = np.ogrid[:h, :w]
    boundary_mask[:1, :] = True
    # boundary_mask[(y-h//2)**2 + (x-w//2)**2 < h*w//20] = True
    # boundary_mask[(y-h//2)**2 + (x-w//2)**2 < h*w//160] = True

    # potential
    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[boundary_mask > 0] = 255

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    # update potential
    plt.ion()

    for i in range(10000):
        pc = update_potential(potential, axes, cmap)
        pc[boundary_mask] = potential[boundary_mask]

        potential = pc

        if (i%100 == 0 and i > 200):
        # if (i%100 == 0 and i > 200):
            print(i)
            axes[0].imshow(potential, cmap=cmap)

            plt.pause(0.1)

    plt.ioff()

    # show the plot
    plt.show()

