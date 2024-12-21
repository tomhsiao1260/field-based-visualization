# original points -> flatten points
# flatten points -> original points

import os
import sys
import nrrd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from app_alpha import update_flatten

def add_rect(xc, yc):
    x, y, w, h = xc-35, yc-35, 70, 70
    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    return rect

# flatten -> original
def generate_original_points(potential, zf, yf, xf, d, h, w):
    # uint8 run faster
    coords_z, coords_y, coords_x = np.indices((256, 256, 256), dtype=np.uint8)

    flat_x = update_flatten(coords_x, potential)
    flat_y = update_flatten(coords_y, potential)

    # flat_coords = np.stack((coords_z, coords_y, coords_x), axis=-1)
    flat_coords = np.stack((coords_z, flat_y, flat_x), axis=-1)
    zo, yo, xo = flat_coords[zf * 256//d][yf * 256//h][xf * 256//w]
    zo, yo, xo = zo * d//256, yo * h//256, xo * w//256

    return zo, yo, xo

# original -> flatten
def generate_flatten_points(potential, zo, yo, xo, d, h, w):
    dp, hp, wp = potential.shape

    yf = potential[zo * dp//d][yo * hp//h][xo * wp//w]

    zf, yf, xf = zo, yf * h//256, xo

    return zf, yf, xf

if __name__ == "__main__":
    zmin, ymin, xmin = (5049, 1765, 3380)

    # (zo, yo, xo), (zf, yf, xf) = (None, None, None), (5249, 2132, 4078)
    (zo, yo, xo), (zf, yf, xf) = (5247, 2182, 4076), (None, None, None)

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

    if zf is not None:
        zf, yf, xf = zf - zmin, yf - ymin, xf - xmin
    if zo is not None:
        zo, yo, xo = zo - zmin, yo - ymin, xo - xmin
    if zf is None and zo is None:
        sys.exit(f'Please enter the coordinates (zf, yf, xf) or (zo, yo, xo).')

    ### load data
    volume, header = nrrd.read(volume_dir)
    flatten, header = nrrd.read(flatten_dir)
    potential, header = nrrd.read(potential_dir)
    d, h, w = volume.shape

    if zo is None:
        zo, yo, xo = generate_original_points(potential, zf, yf, xf, d, h, w)
    if zf is None:
        zf, yf, xf = generate_flatten_points(potential, zo, yo, xo, d, h, w)

    print(f'original (z, y, x): {zmin+zo} {ymin+yo} {xmin+xo}')
    print(f'flatten (z, y, x): {zmin+zf} {ymin+yf} {xmin+xf}')

    ### plot
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # draw xyz slices
    axes[0].set_title(f"zo {zmin+zo}")
    axes[1].set_title(f"yo {ymin+yo}")
    axes[2].set_title(f"xo {xmin+xo}")

    axes[0].imshow(volume[zo, :, :], cmap='gray')
    axes[1].imshow(volume[:, yo, :], cmap='gray')
    axes[2].imshow(volume[:, :, xo], cmap='gray')

    axes[3].set_title(f"zf {zmin+zf}")
    axes[4].set_title(f"yf {ymin+yf}")
    axes[5].set_title(f"xf {xmin+xf}")

    axes[3].imshow(flatten[zf, :, :], cmap='gray')
    axes[4].imshow(flatten[:, yf, :], cmap='gray')
    axes[5].imshow(flatten[:, :, xf], cmap='gray')

    # draw rectangle
    rect_z = add_rect(xo, yo)
    rect_y = add_rect(xo, zo)
    rect_x = add_rect(yo, zo)

    axes[0].add_patch(rect_z)
    axes[1].add_patch(rect_y)
    axes[2].add_patch(rect_x)

    rect_z = add_rect(xf, yf)
    rect_y = add_rect(xf, zf)
    rect_x = add_rect(yf, zf)

    axes[3].add_patch(rect_z)
    axes[4].add_patch(rect_y)
    axes[5].add_patch(rect_x)

    plt.tight_layout()
    plt.show()









