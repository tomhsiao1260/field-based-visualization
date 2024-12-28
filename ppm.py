# original points -> flatten points
# flatten points -> original points

import os
import sys
import nrrd
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from potential_generate import update_flatten
from config import volume_dir, volume_flatten_dir, potential_dir

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
    ### params
    parser = argparse.ArgumentParser(description='potential transform (for volume & electrode mask)')
    parser.add_argument('--z', type=int, help='z index')
    parser.add_argument('--y', type=int, help='y index')
    parser.add_argument('--x', type=int, help='x index')
    parser.add_argument('--zo', type=int, help='original z coords (0 to d)')
    parser.add_argument('--yo', type=int, help='original y coords (0 to h)')
    parser.add_argument('--xo', type=int, help='original x coords (0 to w)')
    parser.add_argument('--zf', type=int, help='flatten z coords (0 to d)')
    parser.add_argument('--yf', type=int, help='flatten y coords (0 to h)')
    parser.add_argument('--xf', type=int, help='flatten x coords (0 to w)')
    args = parser.parse_args()

    zmin, ymin, xmin = args.z, args.y, args.x
    zo, yo, xo = args.zo, args.yo, args.xo
    zf, yf, xf = args.zf, args.yf, args.xf

    volume_path = volume_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    potential_path = potential_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)
    volume_flatten_path = volume_flatten_dir.format(zmin, ymin, xmin, zmin, ymin, xmin)

    if not os.path.exists(volume_path):
        sys.exit(f'volume {os.path.basename(volume_path)} does not exist')
    if not os.path.exists(volume_flatten_path):
        sys.exit(f'flatten {os.path.basename(volume_flatten_path)} does not exist')
    if not os.path.exists(potential_path):
        sys.exit(f'potential {os.path.basename(potential_path)} does not exist')
    if zf is None and zo is None:
        sys.exit(f'Please enter the coordinates (zf, yf, xf) or (zo, yo, xo).')

    ### load data
    volume, header = nrrd.read(volume_path)
    potential, header = nrrd.read(potential_path)
    flatten, header = nrrd.read(volume_flatten_path)
    d, h, w = volume.shape

    # flatten points -> original points
    if zo is None:
        zo, yo, xo = generate_original_points(potential, zf, yf, xf, d, h, w)
    # original points -> flatten points
    if zf is None:
        zf, yf, xf = generate_flatten_points(potential, zo, yo, xo, d, h, w)

    print(f'original (z, y, x): {zo} {yo} {xo}')
    print(f'flatten (z, y, x): {zf} {yf} {xf}')

    ### plot
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # draw xyz slices
    axes[0].set_title(f"zo {zo}")
    axes[1].set_title(f"yo {yo}")
    axes[2].set_title(f"xo {xo}")

    axes[0].imshow(volume[zo, :, :], cmap='gray')
    axes[1].imshow(volume[:, yo, :], cmap='gray')
    axes[2].imshow(volume[:, :, xo], cmap='gray')

    axes[3].set_title(f"zf {zf}")
    axes[4].set_title(f"yf {yf}")
    axes[5].set_title(f"xf {xf}")

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









