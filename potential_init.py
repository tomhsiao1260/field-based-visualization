# used to generate the initial potential

import os
import nrrd
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

from config import potential_init_dir, electrode_dir

def copy_boundary(a, b):
    a[0, :, :] = b[0, :, :]
    a[-1,:, :] = b[-1,:, :]
    a[:, 0, :] = b[:, 0, :]
    a[:,-1, :] = b[:,-1, :]
    a[:, :, 0] = b[:, :, 0]
    a[:, :,-1] = b[:, :,-1]
    return a

def fill_boundary(a):
    a[ 0, :, :] = a[ 1, :, :]
    a[-1, :, :] = a[-2, :, :]
    a[ :, 0, :] = a[ :, 1, :]
    a[ :,-1, :] = a[ :,-2, :]
    a[ :, :, 0] = a[ :, :, 1]
    a[ :, :,-1] = a[ :, :,-2]
    return a

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

# potential (0~1) & side mask (bool) for a given electrode (bool)
def generate_potential_side(electrode, target=(0,0,0), plot_info=None):
    potential = generate_normalized_potential(electrode, plot_info)
    contour = np.zeros_like(potential, dtype=bool)
    contour[potential > 0.95] = True

    side = generate_side(contour, target, plot_info)
    return potential, side

# generate normalized potential (0~1) from electrode (bool)
def generate_normalized_potential(electrode, plot_info=None):
    d, h, w = electrode.shape
    potential = np.zeros_like(electrode, dtype=float)
    if plot: axes, cmap = plot_info

    print('potential generation ...')
    for i in tqdm(range(501)):
        # electrodes should remain constant
        potential[electrode] = 1
        potential = update_potential(potential)

        if plot:
            if (i%20 == 0):
                axes[0].set_title("Potential (norm)")
                axes[0].imshow(potential[d//2, :, :], cmap=cmap)
                axes[6].imshow(potential[:, :, w//2], cmap=cmap)
                axes[0].contour(electrode[d//2, :, :], colors='blue', linewidths=0.5)
                axes[6].contour(electrode[:, :, w//2], colors='blue', linewidths=0.5)
                plt.pause(0.01)

    # fill boundary
    potential = fill_boundary(potential)

    return potential

# generate side mask for a given contour (bool)
def generate_side(contour, target=(0,0,0), plot_info=None):
    d, h, w = contour.shape
    side = np.zeros_like(contour, dtype=bool)
    if plot_info: axes, cmap = plot_info

    # init trigger points (target: xyz)
    side[target[2], target[1], target[0]] = True

    print('side generation ...')
    for i in tqdm(range(w+h+d)):
        side = update_side(side, contour)

        if plot_info:
            if (i%30 == 0):
                axes[1].set_title("Side")
                axes[1].imshow(side[d//2, :, :], cmap="gray")
                axes[7].imshow(side[:, :, w//2], cmap="gray")
                plt.pause(0.01)

    # fill boundary
    side = fill_boundary(side)

    return side

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1, 1:-1].copy()
    top    = pc_pad[1:-1,  :-2, 1:-1].copy()
    bot    = pc_pad[1:-1,   2:, 1:-1].copy()
    left   = pc_pad[1:-1, 1:-1,  :-2].copy()
    right  = pc_pad[1:-1, 1:-1,   2:].copy()
    front  = pc_pad[ :-2, 1:-1, 1:-1].copy()
    back   = pc_pad[  2:, 1:-1, 1:-1].copy()

    diff_top   = center - top
    diff_bot   = center - bot
    diff_left  = center - left
    diff_right = center - right
    diff_front = center - front
    diff_back  = center - back

    max_diff   = 0.01
    diff_top   = np.where(diff_top > max_diff, max_diff, diff_top)
    diff_bot   = np.where(diff_bot > max_diff, max_diff, diff_bot)
    diff_left  = np.where(diff_left > max_diff, max_diff, diff_left)
    diff_right = np.where(diff_right > max_diff, max_diff, diff_right)
    diff_front = np.where(diff_front > max_diff, max_diff, diff_front)
    diff_back  = np.where(diff_back > max_diff, max_diff, diff_back)

    pc = center - (diff_top + diff_bot + diff_left + diff_right + diff_front + diff_back) / 6

    # fixed boundary
    pc = copy_boundary(pc, potential)

    return pc

# update side mask & fill in the corresponding side
def update_side(side, contour):
    side_pad = np.pad(side, pad_width=1, mode='edge')

    top   = side_pad[1:-1,  :-2, 1:-1].copy()
    bot   = side_pad[1:-1,   2:, 1:-1].copy()
    left  = side_pad[1:-1, 1:-1,  :-2].copy()
    right = side_pad[1:-1, 1:-1,   2:].copy()
    front = side_pad[ :-2, 1:-1, 1:-1].copy()
    back  = side_pad[  2:, 1:-1, 1:-1].copy()

    sc = (top == True) | (bot == True) | (left == True) | (right == True) | (front == True) | (back == True)
    sc[contour != 0] = False # stop the propagation

    # fixed boundary
    sc = copy_boundary(sc, side)

    return sc

# # potential init (for better convergence)
# def potential_init(electrode, electrode_label_level_pairs):
#     potential = None
#     d, h, w = electrode.shape

#     for label, level in electrode_label_level_pairs:
#         electrode_single = np.zeros_like(electrode, dtype=bool)
#         electrode_single[electrode == label] = True

#         target_a = (0, 0)
#         target_b = (w-1, h-1)
#         normailized_potential, side_a, side_b = generate_potential_side(electrode_single, target_a, target_b)

#         # combine side a & b potential
#         potential = level * np.ones_like(normailized_potential, dtype=float)
#         potential[side_a] = level * normailized_potential[side_a] # forward
#         potential[side_b] = level * (2 - normailized_potential[side_b]) # reverse
#         potential *= 255

#     return potential

if __name__ == "__main__":
    ### params
    parser = argparse.ArgumentParser(description='potential init calculation')
    parser.add_argument('--z', type=int, help='z index')
    parser.add_argument('--y', type=int, help='y index')
    parser.add_argument('--x', type=int, help='x index')
    parser.add_argument('--plot', action="store_true", help='plot the potential')
    parser.add_argument('--labels', type=int, nargs='+', help='list of electrode labels')
    parser.add_argument('--auto_conductor', action="store_true", help='auto generate the conductor mask')
    args = parser.parse_args()

    zmin, ymin, xmin = args.z, args.y, args.x
    plot, labels, auto_conductor = args.plot, args.labels, args.auto_conductor

    if auto_conductor:
        rescale = (5*1, 5*1, 5*1)
    else:
        rescale = (3*2, 3*2, 3*2)

    ### load electrode
    electrode_path = electrode_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    electrode, header = nrrd.read(electrode_path)
    electrode = np.asarray(electrode)[:, :, :]
    electrode = down_sampling(electrode, rescale, False)
    d, h, w = electrode.shape

    ### plot init
    plot_info = None

    if (plot):
        fig, axes = plt.subplots(2, 6, figsize=(10, 4))

        axes = axes.ravel()
        for ax in axes: ax.axis('off')

        colors = ['#000000', '#ffffff'] * 20
        cmap = ListedColormap(colors)
        plot_info = (axes, cmap)

        plt.ion()

    ### handling each electrode
    for label in labels:
        print('Processing electrode:', label)

        mask_label = (electrode == label).astype(np.uint8)
        electrode_single = np.zeros_like(electrode, dtype=bool)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_single[z][skeleton] = True

        target = (w//2, 0, d//2)
        normailized_potential, side = generate_potential_side(electrode_single, target, plot_info)

        # combine potential via side mask
        potential = np.ones_like(normailized_potential, dtype=float)
        potential[side] = normailized_potential[side] # forward
        potential[~side] = 2 - normailized_potential[~side] # reverse

        p_min = np.min(potential)
        p_max = np.max(potential)
        potential = (potential - p_min) / (p_max - p_min)
        potential = 255 * potential
        # potential = (255 + 50 * 2) * potential - 50

        if (plot):
            axes[2].set_title("Potential")
            axes[2].imshow(potential[d//2, :, :], cmap=cmap)
            axes[3].imshow(potential[d//2, :, :], cmap="gray")
            axes[8].imshow(potential[:, :, w//2], cmap=cmap)
            axes[9].imshow(potential[:, :, w//2], cmap="gray")

        # save init potential
        potential_init_path = potential_init_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
        nrrd.write(potential_init_path, potential)

    print('complete')
    if (plot):
        plt.ioff()
        plt.tight_layout()
        plt.show()




