import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

from config import electrode_label_level_pairs, electrode_dir

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
def generate_potential_side(electrode, target_a=(0, 0), target_b=(0, 0)):
    potential = generate_normalized_potential(electrode)
    contour = np.zeros_like(potential, dtype=bool)
    contour[potential > 0.99] = True

    side_a, side_b = generate_side(contour, target_a, target_b)
    return potential, side_a, side_b

# generate normalized potential (0~1) from electrode (bool)
def generate_normalized_potential(electrode):
    d, h, w = electrode.shape
    potential = np.zeros_like(electrode, dtype=float)

    print('potential generation ...')
    for i in tqdm(range(500)):
        # electrodes should remain constant
        potential[electrode] = 1
        potential = update_potential(potential)

    # fill boundary
    potential[ 0, :, :] = potential[ 1, :, :]
    potential[-1, :, :] = potential[-2, :, :]
    potential[ :, 0, :] = potential[ :, 1, :]
    potential[ :,-1, :] = potential[ :,-2, :]
    potential[ :, :, 0] = potential[ :, :, 1]
    potential[ :, :,-1] = potential[ :, :,-2]

    return potential

# generate side mask a & b for a given contour (bool)
def generate_side(contour, target_a=(0, 0), target_b=(0, 0)):
    d, h, w = contour.shape
    side_a = np.zeros_like(contour, dtype=bool)
    side_b = np.zeros_like(contour, dtype=bool)

    # init trigger points
    side_a[:, target_a[1], target_a[0]] = True
    side_b[:, target_b[1], target_b[0]] = True

    print('side generation ...')
    for i in tqdm(range(w+h)):
        side_a = update_side(side_a, contour)
        side_b = update_side(side_b, contour)

    return side_a, side_b

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
    pc[0, :, :] = potential[0, :, :]
    pc[-1,:, :] = potential[-1,:, :]
    pc[:, 0, :] = potential[:, 0, :]
    pc[:,-1, :] = potential[:,-1, :]
    pc[:, :, 0] = potential[:, :, 0]
    pc[:, :,-1] = potential[:, :,-1]

    return pc

# update side a & b to fill in the corresponding side
def update_side(side, contour):
    side_pad = np.pad(side, pad_width=1, mode='edge')

    top   = side_pad[1:-1,  :-2, 1:-1].copy()
    bot   = side_pad[1:-1,   2:, 1:-1].copy()
    left  = side_pad[1:-1, 1:-1,  :-2].copy()
    right = side_pad[1:-1, 1:-1,   2:].copy()

    side = (top == True) | (bot == True) | (left == True) | (right == True)
    side[contour != 0] = False # stop the propagation

    return side

if __name__ == "__main__":
    ### params
    rescale = (3*2, 3*2, 3*2)

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    electrode = down_sampling(electrode, rescale, False)
    d, h, w = electrode.shape

    ### plot init
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    ### handling each electrode
    for label, level in electrode_label_level_pairs:
        print('Processing electrode:', label)

        mask_label = (electrode == label).astype(np.uint8)
        electrode_single = np.zeros_like(electrode, dtype=bool)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_single[z][skeleton] = True

        axes[0].set_title("Electrode")
        axes[0].contour(electrode_single[d//2, ::-1, :], colors='blue', linewidths=0.5)

        target_a = (0, 0)
        target_b = (w-1, h-1)
        normailized_potential, side_a, side_b = generate_potential_side(electrode_single, target_a, target_b)

        axes[1].set_title("Potential (norm)")
        axes[1].imshow(normailized_potential[d//2, :, :], cmap=cmap)

        axes[2].set_title("Side A")
        axes[3].set_title("Side B")
        axes[2].imshow(side_a[d//2, :, :], cmap="gray")
        axes[3].imshow(side_b[d//2, :, :], cmap="gray")

        # combine side a & b potential
        potential = level * np.ones_like(normailized_potential, dtype=float)
        potential[side_a] = level * normailized_potential[side_a] # forward
        potential[side_b] = level * (2 - normailized_potential[side_b]) # reverse
        potential *= 255

        axes[4].set_title("Potential")
        axes[4].imshow(potential[d//2, :, :], cmap=cmap)
        axes[5].imshow(potential[d//2, :, :], cmap="gray")

    plt.show()




