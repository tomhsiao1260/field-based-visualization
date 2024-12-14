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

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='reflect')

    # pc  = pc_pad[1:-1, 1:-1,  :-2].copy()
    # pc += pc_pad[1:-1, 1:-1,   2:]
    # pc += pc_pad[1:-1,  :-2, 1:-1]
    # pc += pc_pad[1:-1,   2:, 1:-1]
    # pc += pc_pad[ :-2, 1:-1, 1:-1]
    # pc += pc_pad[  2:, 1:-1, 1:-1]
    # pc /= 6

    pc  = pc_pad[1:-1, 1:-1,  :-2].copy()
    pc += pc_pad[1:-1, 1:-1,   2:]
    pc += pc_pad[1:-1,  :-2, 1:-1]
    pc += pc_pad[1:-1,   2:, 1:-1]
    pc /= 4

    

    # # boundary fix
    # s = 0.9535

    # pc[ 0, :, :] = s * (2 * pc[ 1, :, :] - pc[ 2, :, :]) + (1-s) * pc[ 1, :, :]
    # pc[-1, :, :] = s * (2 * pc[-2, :, :] - pc[-3, :, :]) + (1-s) * pc[-2, :, :]
    # pc[ :, 0, :] = s * (2 * pc[ :, 1, :] - pc[ :, 2, :]) + (1-s) * pc[ :, 1, :]
    # pc[ :,-1, :] = s * (2 * pc[ :,-2, :] - pc[ :,-3, :]) + (1-s) * pc[ :,-2, :]
    # pc[ :, :, 0] = s * (2 * pc[ :, :, 1] - pc[ :, :, 2]) + (1-s) * pc[ :, :, 1]
    # pc[ :, :,-1] = s * (2 * pc[ :, :,-2] - pc[ :, :,-3]) + (1-s) * pc[ :, :,-2]

    return pc

if __name__ == "__main__":
    ### path & params

    # zmin, ymin, xmin, electrode_label_level_pairs = 5049, 2533, 3380, [(2, 0.60)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 5049, 1765, 3380, [(1, 0.70)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 4281, 2533, 3380, [(2, 0.30)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 4281, 1765, 3380, [(1, 0.50), (2, 0.95)]
    zmin, ymin, xmin, electrode_label_level_pairs = 3513, 1900, 3400, [(1, 0.15), (2, 0.70)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 2736, 1831, 3413, [(1, 0.20), (2, 0.75)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 1968, 1860, 3424, [(1, 0.20), (2, 0.95)]
    # zmin, ymin, xmin, electrode_label_level_pairs = 1200, 1537, 3490, [(1, 0.65)]

    dirname = f'/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{zmin:05}_{ymin:05}_{xmin:05}/'

    electrode_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_mask.nrrd')

    rescale_0 = (1, 3, 3)
    rescale_1 = (1, 2, 2)
    # rescale_0 = (3, 3, 1)
    # rescale_1 = (1, 1, 1)

    ### plot init
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    electrode = electrode[:2, :, :]
    # electrode = electrode[:, :, :2]
    electrode = down_sampling(electrode, rescale_0, False)

    d, h, w = electrode.shape

    electrode_temp = np.zeros_like(electrode)

    for label, level in electrode_label_level_pairs:
        print('Processing electrode:', label)
        mask_label = (electrode == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_temp[z][skeleton] = label

    electrode = electrode_temp

    axes[0].set_title("Volume")
    axes[0].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)
    axes[5].contour(electrode[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    # potential (init)
    potential = np.zeros_like(electrode, dtype=float)
    # in some cases, may need some adjustments for better convergence
    # for y in range(h): potential[:, y, :] = (y / h) * 255
    potential[:, :1, :] = 0
    potential[:, -1:, :] = 255

    boundary = np.zeros_like(electrode, dtype=bool)
    boundary[:, :1, :] = True
    boundary[:, -1:, :] = True

    # rescale again
    potential = down_sampling(potential, rescale_1)
    electrode = down_sampling(electrode, rescale_1, False)
    boundary = down_sampling(boundary, rescale_1, False)

    d, h, w = potential.shape

    # update potential
    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    plt.ion()
    for i in range(101):
        # electrodes should remain constant
        for label, level in electrode_label_level_pairs:
            potential[electrode == label] = level * 255

        # potential (free space)
        pc = update_potential(potential)
        pc[boundary] = potential[boundary]
        pc[electrode > 0] = potential[electrode > 0]
        potential = pc

        # plot
        if (i%10 == 0):
            print('frame ', i)

            axes[3].set_title("Potential")
            axes[3].imshow(potential[d//2, :, :], cmap=cmap)
            axes[8].imshow(potential[:, :, w//2], cmap=cmap)

            plt.pause(0.001)
    plt.ioff()

    print('complete')
    plt.tight_layout()
    plt.show()
