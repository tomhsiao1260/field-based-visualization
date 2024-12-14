import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def update_potential(potential, axes, cmap):
    # pc_pad = np.pad(potential, pad_width=1, mode='reflect')
    pc_pad = np.pad(potential, pad_width=1, mode='edge')
    # pc_pad = np.pad(potential, pad_width=1, mode='constant', constant_values=0)

    # dy, dx = np.gradient(pc_pad)

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

if __name__ == '__main__':
    # plot init
    fig, axes = plt.subplots(1, 6, figsize=(10, 3))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # mask
    boundary_mask = np.zeros((100, 100), dtype=bool)
    h, w = boundary_mask.shape

    y, x = np.ogrid[:h, :w]
    # boundary_mask[:1, :] = True
    # boundary_mask[abs(x-y)>70] = True
    boundary_mask[(0 < x) & (x < 100) & (25 < y) & (y < 35)] = True
    # boundary_mask[(y-h//2)**2 + (x-w//2)**2 < h*w//160] = True
    # boundary_mask[(y-h//4)**2 + (x-w//4)**2 < h*w//160] = True
    # boundary_mask[((y-1*h//4)**2 + (x-1*w//4)**2 < h*w//160) | ((y-3*h//4)**2 + (x-3*w//4)**2 < h*w//160)] = True

    # potential
    potential = np.zeros_like(boundary_mask, dtype=float)
    potential[boundary_mask > 0] = 128

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

            plt.pause(0.01)

    plt.ioff()

    # show the plot
    plt.show()

