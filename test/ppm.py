import os
import sys
import nrrd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from app_alpha import update_flatten

if __name__ == "__main__":
    zmin, ymin, xmin = 5049, 1765, 3380
    zf, yf, xf = 200, 367, 698

    dirname = f'/Users/yao/Desktop/full-scrolls/community-uploads/yao/scroll1/{zmin:05}_{ymin:05}_{xmin:05}/'

    volume_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_volume.nrrd')
    flatten_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_flatten.nrrd')
    potential_dir = os.path.join(dirname, f'{zmin:05}_{ymin:05}_{xmin:05}_potential.nrrd')

    if not os.path.exists(volume_dir):
        sys.exit(f'volume {os.path.basename(volume_dir)} does not exist')
    if not os.path.exists(flatten_dir):
        sys.exit(f'flatten {os.path.basename(flatten_dir)} does not exist')
    if not os.path.exists(potential_dir):
        sys.exit(f'potential {os.path.basename(potential_dir)} does not exist')

    ### load data
    volume, header = nrrd.read(volume_dir)
    flatten, header = nrrd.read(flatten_dir)
    potential, header = nrrd.read(potential_dir)

    d, h, w = volume.shape

    # uint8 run faster
    coords_z, coords_y, coords_x = np.indices((256, 256, 256), dtype=np.uint8)

    flat_x = update_flatten(coords_x, potential)
    flat_y = update_flatten(coords_y, potential)

    # flat_coords = np.stack((coords_z, coords_y, coords_x), axis=-1)
    flat_coords = np.stack((coords_z, flat_y, flat_x), axis=-1)
    zo, yo, xo = flat_coords[zf * 256//d][yf * 256//h][xf * 256//w]
    zo, yo, xo = zo * d//256, yo * h//256, xo * w//256

    print(f'offset (z, y, x): {zmin} {ymin} {xmin}')
    print(f'original (z, y, x): {zo} {yo} {xo}')
    print(f'flatten (z, y, x): {zf} {yf} {xf}')

    ### plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax in axes: ax.axis('off')

    axes[0].set_title("Original")
    axes[0].imshow(volume[zo], cmap='gray')

    axes[1].set_title("Flatten")
    axes[1].imshow(flatten[zf], cmap='gray')

    x, y, w, h = xo-25, yo-25, 50, 50
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)

    x, y, w, h = xf-25, yf-25, 50, 50
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    axes[1].add_patch(rect)

    plt.show()









