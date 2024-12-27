import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt

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

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='edge')

    center = pc_pad[1:-1, 1:-1].copy()
    top = pc_pad[:-2, 1:-1].copy()
    bot = pc_pad[2:, 1:-1].copy()
    left = pc_pad[1:-1,  :-2].copy()
    right = pc_pad[1:-1,   2:].copy()

    # pc = (top + bot + left + right) / 4

    grad_top = center - top
    grad_bot = center - bot
    grad_left = center - left
    grad_right = center - right

    max_gradient = 1

    grad_top = np.where(grad_top > max_gradient, max_gradient, grad_top)
    grad_bot = np.where(grad_bot > max_gradient, max_gradient, grad_bot)
    grad_left = np.where(grad_left > max_gradient, max_gradient, grad_left)
    grad_right = np.where(grad_right > max_gradient, max_gradient, grad_right)

    pc = center - 1.0 * (
        (grad_top + grad_bot + grad_left + grad_right) / 4
    )

    # fixed boundary
    pc[0, :] = potential[0, :]
    pc[-1, :] = potential[-1, :]
    pc[:, 0] = potential[:, 0]
    pc[:, -1] = potential[:, -1]

    return pc

def update_mask(mask, electrode):
    m_pad = np.pad(mask, pad_width=1, mode='edge')

    top = m_pad[:-2, 1:-1].copy()
    bot = m_pad[2:, 1:-1].copy()
    left = m_pad[1:-1,  :-2].copy()
    right = m_pad[1:-1,   2:].copy()

    m = (top == True) | (bot == True) | (left == True) | (right == True)
    m[electrode != 0] = False

    return m

if __name__ == "__main__":
    ### path & params
    rescale = (1, 3*2, 3*2)

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    d, h, w = electrode.shape

    electrode = electrode[d//2, :, :][np.newaxis, ...]
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

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    axes[0].set_title("Electrode")
    axes[0].contour(electrode[d//2, ::-1, :] * 255, colors='blue', linewidths=0.5)

    ### potential init
    potential = np.zeros_like(electrode, dtype=float)

    ### mask init
    region_mask_1 = np.zeros_like(electrode, dtype=bool)
    region_mask_2 = np.zeros_like(electrode, dtype=bool)
    mask_electrode = np.zeros_like(electrode, dtype=bool)

    region_mask_1[d//2, h-1, 0] = True
    region_mask_2[d//2, 0, w-1] = True

    l = 0
    for label, level in electrode_label_level_pairs: l = level

    plt.ion()
    for i in range(301):
        # electrodes should remain constant
        potential[electrode == label] = l * 255
        mask_electrode[d//2, :, :] =  (l * 255 - 1 < potential[d//2, :, :]) & (potential[d//2, :, :] < l * 255 + 1)

        pc = update_potential(potential[d//2, :, :])

        potential[d//2, :, :] = pc

        if (i%10 == 0):
            print(i)
            axes[1].set_title("Potential")
            axes[1].imshow(potential[d//2, :, :], cmap=cmap)
            axes[2].imshow(potential[d//2, :, :], cmap="gray")
            axes[3].imshow(mask_electrode[d//2, :, :] * 255, cmap="gray")

            plt.pause(0.01)
    plt.ioff()

    # fill boundary
    potential[d//2, 0, :] = potential[d//2, 1, :]
    potential[d//2, -1, :] = potential[d//2, -2, :]
    potential[d//2, :, 0] = potential[d//2, :, 1]
    potential[d//2, :, -1] = potential[d//2, :, -2]

    mask_electrode[d//2, :, :] =  (l * 255 - 1 < potential[d//2, :, :]) & (potential[d//2, :, :] < l * 255 + 1)

    plt.ion()
    for i in range(w+h):

        region_mask_1[d//2, :, :] = update_mask(region_mask_1[d//2, :, :], mask_electrode[d//2, :, :])
        region_mask_2[d//2, :, :] = update_mask(region_mask_2[d//2, :, :], mask_electrode[d//2, :, :])

        if (i%10 == 0):
            print(i)
            axes[4].set_title("Region Mask 1")
            axes[4].imshow(region_mask_1[d//2, :, :], cmap="gray")

            axes[5].set_title("Region Mask 2")
            axes[5].imshow(region_mask_2[d//2, :, :], cmap="gray")

            potential_final = potential.copy()
            potential_final[region_mask_1] = 2 * 255 * l - potential[region_mask_1]

            axes[6].set_title("Final Potential")
            axes[6].imshow(potential_final[d//2, :, :], cmap=cmap)
            axes[7].imshow(potential_final[d//2, :, :], cmap="gray")

            plt.pause(0.01)
    plt.ioff()

    plt.show()


