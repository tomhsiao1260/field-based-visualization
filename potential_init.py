### Yao Hsiao - Field-Based Visualization - 2024

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
def generate_potential_side(electrode, plot_info=None):
    potential = generate_normalized_potential(electrode, plot_info)
    contour = np.zeros_like(potential, dtype=bool)
    contour[potential > 0.99] = True

    side = generate_side(contour, plot_info)
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

        # prevent propagation error from other z slices
        if (i < 300): potential = update_potential(potential, True)
        if (i >= 300): potential = update_potential(potential, False)

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
def generate_side(contour, plot_info=None):
    if plot_info: axes, cmap = plot_info

    d, h, w = contour.shape
    side_a = np.zeros_like(contour, dtype=bool)
    side_b = np.zeros_like(contour, dtype=bool)

    # init trigger lines
    side_a[:, :-5, :] = contour[:, 5:, :] # curve move up
    side_b[:, 5:, :] = contour[:, :-5, :] # curve move down

    print('side generation ...')
    for i in tqdm(range(w+h+d)):
        if (i < w+h):
            side_a = update_side(side_a, side_b, True)
            side_b = update_side(side_b, side_a, True)
        else:
            side_a = update_side(side_a, side_b, False)
            side_b = update_side(side_b, side_a, False)


        if plot_info:
            if (i%30 == 0):
                axes[1].set_title("Side")
                axes[1].imshow(side_a[d//2, :, :], cmap="gray")
                axes[7].imshow(side_a[:, :, w//2], cmap="gray")
                plt.pause(0.01)

    # fill boundary
    side = fill_boundary(side_a)

    return side

def update_potential(potential, plane_xy=False):
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

    if plane_xy:
        pc = center - (diff_top + diff_bot + diff_left + diff_right) / 4
    else:
        pc = center - (diff_top + diff_bot + diff_left + diff_right + diff_front + diff_back) / 6
        # pc = center - (diff_top + diff_bot + diff_left + diff_right + 0.1 * diff_front + 0.1 * diff_back) / 4.2

    # fixed boundary
    pc = copy_boundary(pc, potential)

    return pc

# update side mask & fill in the corresponding side
def update_side(side, contour, plane_xy=False):
    side_pad = np.pad(side, pad_width=1, mode='edge')

    top   = side_pad[1:-1,  :-2, 1:-1].copy()
    bot   = side_pad[1:-1,   2:, 1:-1].copy()
    left  = side_pad[1:-1, 1:-1,  :-2].copy()
    right = side_pad[1:-1, 1:-1,   2:].copy()
    front = side_pad[ :-2, 1:-1, 1:-1].copy()
    back  = side_pad[  2:, 1:-1, 1:-1].copy()

    if plane_xy:
        sc = (top == True) | (bot == True) | (left == True) | (right == True)
    else:
        sc = (top == True) | (bot == True) | (left == True) | (right == True) | (front == True) | (back == True)

    sc[contour != 0] = False # stop the propagation

    # fixed boundary
    sc = copy_boundary(sc, side)

    return sc

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

        plt.tight_layout()
        plt.ion()

    ### handling each electrode (only labels 1, 2 so far)
    p1, p2, s1, s2 = None, None, None, None

    for label in labels:
        if label > 2:
            print('Skip electrode ', label)
            continue
        else:
            print('Processing electrode:', label)

        mask_label = (electrode == label).astype(np.uint8)
        electrode_single = np.zeros_like(electrode, dtype=bool)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_single[z][skeleton] = True

        normailized_potential, side = generate_potential_side(electrode_single, plot_info)
        normailized_potential[~side] = 2 - normailized_potential[~side] # reverse one side

        if (plot):
            axes[2].set_title("Potential (flip)")
            axes[2].imshow(normailized_potential[d//2, :, :], cmap=cmap)
            axes[3].imshow(normailized_potential[d//2, :, :], cmap="gray")
            axes[8].imshow(normailized_potential[:, :, w//2], cmap=cmap)
            axes[9].imshow(normailized_potential[:, :, w//2], cmap="gray")

        if (label == 1):
            p1 = normailized_potential.copy()
            s1 = side.copy()
        if (label == 2):
            p2 = normailized_potential.copy()
            s2 = side.copy()

    # combine potential via side mask & save
    if (p1 is not None) & (p2 is not None):
        m1 = np.min(p2[~s1]) # min p2 value on label 1
        m2 = np.min(p1[s2]) # min p1 value on label 2
        p2_ = p2 + max(1-m1, m2-1) # shift p2 to fit p1 curve

        # combine (top & bottom & middle)
        p = p1.copy() # top
        p[~s2] = p2_[~s2] # bottom

        # middile
        mask = ~s1 & s2
        w1 = 1 / (abs(1-p1) + 1e-4) # wieght: closer electrode -> larger
        w2 = 1 / (abs(1-p2) + 1e-4)
        p[mask] = (w1[mask] * p1[mask] + w2[mask] * p2_[mask]) / (w1[mask] + w2[mask])
    elif p1 is not None: p = p1
    elif p2 is not None: p = p2

    p_min = np.min(p)
    p_max = np.max(p)
    potential = (p - p_min) / (p_max - p_min)

    potential = (255 + 50 * 2) * potential - 50
    potential[potential < 0] = 0
    potential[potential > 255] = 255

    # plot & save
    if (plot):
        axes[4].set_title("Potential")
        axes[4].imshow(potential[d//2, :, :], cmap=cmap)
        axes[5].imshow(potential[d//2, :, :], cmap="gray")
        axes[10].imshow(potential[:, :, w//2], cmap=cmap)
        axes[11].imshow(potential[:, :, w//2], cmap="gray")

    potential_init_path = potential_init_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    nrrd.write(potential_init_path, potential)

    print('complete')
    if (plot):
        plt.ioff()
        plt.show()




