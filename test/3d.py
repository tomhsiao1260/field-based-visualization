import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

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

def update_potential(potential, axes, cmap, has_negative_one=True):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    top = pc_pad[1:-1, :-2, 1:-1].copy()
    bot = pc_pad[1:-1, 2:, 1:-1].copy()
    left = pc_pad[1:-1, 1:-1, :-2].copy()
    right = pc_pad[1:-1, 1:-1, 2:].copy()
    front = pc_pad[:-2, 1:-1, 1:-1].copy()
    back = pc_pad[2:, 1:-1, 1:-1].copy()

    if has_negative_one:
        decay = (top == -1) & (bot != -1)
        grow = (top != -1) & (bot == -1)
        static = (top == -1) & (bot == -1) & (left == -1) & (right == -1) & (front == -1) & (back == -1)

        s = 5

        top[grow & (top == -1)] = top[grow & (top == -1)] + s
        bot[grow & (bot == -1)] = top[grow & (bot == -1)] + s
        left[grow & (left == -1)] = top[grow & (left == -1)] + s
        right[grow & (right == -1)] = top[grow & (right == -1)] + s
        front[grow & (front == -1)] = top[grow & (front == -1)] + s
        back[grow & (back == -1)] = top[grow & (back == -1)] + s

        top[decay & (top == -1)] = bot[decay & (top == -1)] - s
        bot[decay & (bot == -1)] = bot[decay & (bot == -1)] - s
        left[decay & (left == -1)] = bot[decay & (left == -1)] - s
        right[decay & (right == -1)] = bot[decay & (right == -1)] - s
        front[decay & (front == -1)] = bot[decay & (front == -1)] - s
        back[decay & (back == -1)] = bot[decay & (back == -1)] - s

    pc = (top + bot + left + right + front + back) / 6

    if has_negative_one:
        pc[pc > 255] = 255
        pc[pc < 0] = 0
        pc[static] = -1

    # boundary fix
    s = 0.9535

    if has_negative_one:
        pc[0, :, :] = pc[1, :, :]
        pc[-1, :, :] = pc[-2, :, :]
        pc[:, 0, :] = pc[:, 1, :]
        pc[:, -1, :] = pc[:, -2, :]
        pc[:, :, 0] = pc[:, :, 1]
        pc[:, :, -1] = pc[:, :, -2]
    else:
        pc[ 0, :, :] = s * (2 * pc[ 1, :, :] - pc[ 2, :, :]) + (1-s) * pc[ 1, :, :]
        pc[-1, :, :] = s * (2 * pc[-2, :, :] - pc[-3, :, :]) + (1-s) * pc[-2, :, :]
        pc[ :, 0, :] = s * (2 * pc[ :, 1, :] - pc[ :, 2, :]) + (1-s) * pc[ :, 1, :]
        pc[ :,-1, :] = s * (2 * pc[ :,-2, :] - pc[ :,-3, :]) + (1-s) * pc[ :,-2, :]
        pc[ :, :, 0] = s * (2 * pc[ :, :, 1] - pc[ :, :, 2]) + (1-s) * pc[ :, :, 1]
        pc[ :, :,-1] = s * (2 * pc[ :, :,-2] - pc[ :, :,-3]) + (1-s) * pc[ :, :,-2]

    return pc

if __name__ == "__main__":
    ### path & params

    # zmin, ymin, xmin, electrode_label_level_pairs = 5049, 2533, 3380, [(2, 0.60)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 5049, 1765, 3380, [(1, 0.70)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 4281, 2533, 3380, [(2, 0.30)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 4281, 1765, 3380, [(1, 0.50), (2, 0.95)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 3513, 1900, 3400, [(1, 0.15), (2, 0.70)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 2736, 1831, 3413, [(1, 0.20), (2, 0.75)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 1968, 1860, 3424, [(1, 0.20), (2, 0.95)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 1200, 1800, 2990, [(1, 0.60)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 1200, 1800, 2990, [(1, 0.60)]
    zmin, ymin, xmin, electrode_label_level_pairs = 1200, 1537, 3490, [(1, 0.65)]

    dirname = f'/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{zmin:05}_{ymin:05}_{xmin:05}/'

    electrode_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_mask.nrrd')

    rescale = (3, 3, 3)

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    electrode = down_sampling(electrode, rescale, False)
    d, h, w = electrode.shape

    electrode_temp = np.zeros_like(electrode)

    for label, level in electrode_label_level_pairs:
        print('Processing electrode:', label)
        mask_label = (electrode == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_temp[z][skeleton] = label

    electrode = electrode_temp

    ### plot init
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    potential = np.zeros_like(electrode, dtype=float)
    potential[:, :, :] = -1

    # rescale again
    rescale = (2, 2, 2)
    potential = down_sampling(potential, rescale)
    electrode = down_sampling(electrode, rescale, False)

    d, h, w = potential.shape

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Electrode")
    axes[0].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)
    axes[5].contour(electrode[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    axes[1].set_title("Potential")
    axes[0].imshow(potential[d//2, :, :], cmap=cmap)
    axes[5].imshow(potential[:, :, w//2], cmap=cmap)

    # update potential
    plt.ion()

    for i in range(3000):
        # electrodes should remain constant
        for label, level in electrode_label_level_pairs:
            potential[electrode == label] = level * 255

        has_negative_one = np.any(potential == -1)

        pc = update_potential(potential, axes, cmap, has_negative_one)
        potential = pc

        if (i%100 == 0 and not has_negative_one):
        # if (i%100 == 0 and i > 100):
            print(i)
            axes[1].imshow(potential[d//2, :, :], cmap=cmap)
            axes[2].imshow(potential[d//2, :, :], cmap="gray")

            axes[6].imshow(potential[:, :, w//2], cmap=cmap)
            axes[7].imshow(potential[:, :, w//2], cmap="gray")

            plt.pause(0.01)

        if (i%2 == 0 and has_negative_one):
            print(i)
            axes[0].imshow(potential[d//2, :, :], cmap=cmap)
            axes[2].imshow(potential[d//2, :, :], cmap="gray")
            axes[3].imshow(a[d//2, :, :], cmap="gray")

            axes[5].imshow(potential[:, :, w//2], cmap=cmap)
            axes[7].imshow(potential[:, :, w//2], cmap="gray")

            # a = (potential == -1).astype(bool)
            # axes[8].imshow(a[:, :, w//2], cmap="gray")

            plt.pause(0.01)

    plt.ioff()

    plt.show()
