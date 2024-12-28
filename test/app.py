### Yao Hsiao - Field-Based Visualization - 2024

# Testing Data:
# https://dl.ash2txt.org/community-uploads/yao/scroll1/03513_01900_03400/

import os
import sys
import cv2
import nrrd
import random
import argparse
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap
from concurrent.futures import ThreadPoolExecutor

# see config_template.py & generate a config.py file (version 0)
from config import electrode_label_level_pairs
from config import volume_dir, electrode_dir, conductor_dir
from config import potential_dir, potential_init_dir
from config import conductor_x_dir, conductor_z_dir

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

# Graph for conductors
def generate_graph(skel):
    # build the connection
    points = np.column_stack(np.where(skel > 0))
    G = nx.Graph()
    for y, x in points:
        G.add_node((y, x))

        # find connection
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                sy, sx = y + dy, x + dx
                if 0 <= sy < skel.shape[0] and 0 <= sx < skel.shape[1]:
                    if skel[sy, sx] > 0:
                        G.add_edge((y, x), (sy, sx))

    # remove small groups
    components = [c for c in nx.connected_components(G) if len(c) >= 3]
    G_filtered = G.subgraph(nodes for component in components for nodes in component).copy()

    return G_filtered, components

# Edge detector along z-axis
def generate_edge(volume):
    d, h, w = volume.shape
    blur = np.zeros_like(volume)
    edge = np.zeros_like(volume, dtype=bool)

    for z in range(d):
        blur[z, :, :] = cv2.GaussianBlur(volume[z, :, :], (7, 7), 0)
        edge[z, :, :] = cv2.Canny(blur[z, :, :], threshold1=90, threshold2=100)

    return edge

# Write graph into mask
def generate_mask(skeleton, num_worker=5):
    # graph
    d, h, w = skeleton.shape
    component_list, processing_list = [], []

    def processing(z):
        print('processing graphs ', z)
        G, components = generate_graph(skeleton[z, :, :])
        component_list.append(components)

    with ThreadPoolExecutor(max_workers=num_worker) as executor:
      futures = [ executor.submit(processing, z) for z in range(d) ]

    # mask
    mask = np.zeros_like(skeleton, dtype=np.uint8)

    for z in range(d):
        components = component_list[z][:255]
        random.shuffle(components)
        for i, component in enumerate(components, start=1):
            for node in component: mask[z, node[0], node[1]] = i

    return mask

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='reflect')

    pc  = pc_pad[1:-1, 1:-1,  :-2].copy()
    pc += pc_pad[1:-1, 1:-1,   2:]
    pc += pc_pad[1:-1,  :-2, 1:-1]
    pc += pc_pad[1:-1,   2:, 1:-1]
    pc += pc_pad[ :-2, 1:-1, 1:-1]
    pc += pc_pad[  2:, 1:-1, 1:-1]
    pc /= 6

    # # boundary fix
    # s = 0.9535

    # pc[ 0, :, :] = s * (2 * pc[ 1, :, :] - pc[ 2, :, :]) + (1-s) * pc[ 1, :, :]
    # pc[-1, :, :] = s * (2 * pc[-2, :, :] - pc[-3, :, :]) + (1-s) * pc[-2, :, :]
    # pc[ :, 0, :] = s * (2 * pc[ :, 1, :] - pc[ :, 2, :]) + (1-s) * pc[ :, 1, :]
    # pc[ :,-1, :] = s * (2 * pc[ :,-2, :] - pc[ :,-3, :]) + (1-s) * pc[ :,-2, :]
    # pc[ :, :, 0] = s * (2 * pc[ :, :, 1] - pc[ :, :, 2]) + (1-s) * pc[ :, :, 1]
    # pc[ :, :,-1] = s * (2 * pc[ :, :,-2] - pc[ :, :,-3]) + (1-s) * pc[ :, :,-2]

    return pc

def update_conductor(potential, conductor, counts, axis=0):
    pc = potential.copy()
    d = potential.shape[axis]

    for l in range(d):
        c = counts[l]

        if (axis == 0):
            mask = conductor[l, :, :]
            p = pc[l, :, :]
        elif (axis == 1):
            mask = conductor[:, l, :]
            p = pc[:, l, :]
        else:
            mask = conductor[:, :, l]
            p = pc[:, :, l]

        sums = np.bincount(mask.flatten(), weights=p.flatten(), minlength=256)
        averages = np.zeros_like(sums)
        nonzero = c > 0
        averages[nonzero] = sums[nonzero] / c[nonzero]

        if (axis == 0):
            pc[l, :, :] = averages[mask]
        elif (axis == 1):
            pc[:, l, :] = averages[mask]
        else:
            pc[:, :, l] = averages[mask]

    return pc

def update_flatten(volume, potential):
    d0, h0, w0 = volume.shape
    d, h, w = potential.shape

    flatten = np.zeros((d0, h0, w0), dtype=np.uint8)
    selected_points_top = [(x, h-1) for x in range(w)]
    selected_points_bottom = [(x, 0) for x in range(w)]

    for z in range(d0):
        points_array = generate_2d_points_array(potential[z*d//d0-1], selected_points_top, selected_points_bottom, h)

        map_x = points_array[..., 0].astype(np.float32) * (w0 / w)
        map_y = points_array[..., 1].astype(np.float32) * (h0 / h)

        map_x = cv2.resize(map_x, (w0, h0), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (w0, h0), interpolation=cv2.INTER_LINEAR)
        flatten[z] = cv2.remap(volume[z], map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return flatten

# potential sampling to generate flatten result
def generate_2d_points_array(potential, points_top, points_bottom, num_depth):
    points_top = np.array(points_top)
    points_bottom = np.array(points_bottom)

    points_grid = np.linspace(points_top, points_bottom, num_depth * 1, axis=0).astype(int)
    potential_grid = potential[points_grid[..., 1], points_grid[..., 0]]
    levels = np.linspace(0, 255, num_depth) 

    # y, x, 2
    num_points = points_top.shape[0]
    points_array = np.zeros((num_depth, num_points, 2))

    for i in range(num_points):
        points_array[:, i, 0] = np.interp(levels, potential_grid[:, i][::-1], points_grid[:, i, 0][::-1])
        points_array[:, i, 1] = np.interp(levels, potential_grid[:, i][::-1], points_grid[:, i, 1][::-1])

    return points_array

if __name__ == "__main__":
    ### params
    parser = argparse.ArgumentParser(description='potential calculation')
    parser.add_argument('--plot', action="store_true", help='plot the potential')
    parser.add_argument('--num_worker', type=int, default=8, help='worker numbers')
    parser.add_argument('--auto_conductor', action="store_true", help='auto generate the conductor mask')
    args = parser.parse_args()

    num_worker, plot, auto_conductor = args.num_worker, args.plot, args.auto_conductor

    if auto_conductor:
        rescale_a, rescale_b = (5, 5, 5), (1, 1, 1)
    else:
        rescale_a, rescale_b = (3, 3, 3), (2, 2, 2)

    ### plot init
    if (plot):
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        axes = axes.ravel()
        for ax in axes: ax.axis('off')

    ### load volume
    volume_origin, header = nrrd.read(volume_dir)
    volume = down_sampling(volume_origin, rescale_a)
    d, h, w = volume.shape

    ### load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    electrode = down_sampling(electrode, rescale_a, False)

    electrode_temp = np.zeros_like(electrode)

    for label, level in electrode_label_level_pairs:
        print('Processing electrode:', label)
        mask_label = (electrode == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_temp[z][skeleton] = label

    electrode = electrode_temp

    if (plot):
        axes[0].set_title("Volume")
        axes[0].imshow(volume[d//2, :, :], cmap='gray')
        axes[0].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)
        axes[5].imshow(volume[:, :, w//2], cmap='gray')
        axes[5].contour(electrode[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    ### load conductor
    if auto_conductor:
        conductor_zo = generate_edge(volume)
        conductor_xo = generate_edge(volume.transpose(2, 1, 0)).transpose(2, 1, 0)
    else:
        if not os.path.exists(conductor_dir): sys.exit('conductor not found, please use --auto_conductor instead')

        conductor_zo, header = nrrd.read(conductor_dir)
        conductor_zo = np.asarray(conductor_zo).astype(bool)
        conductor_zo = down_sampling(conductor_zo, rescale_a, False)
        conductor_xo = conductor_zo.copy()

    if (plot):
        axes[1].set_title("Conductor")
        axes[1].imshow(conductor_zo[d//2, :, :], cmap='gray')
        axes[6].imshow(conductor_xo[:, :, w//2], cmap='gray')
        axes[1].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)
        axes[6].contour(electrode[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    ### conductor graph

    # along z
    if os.path.exists(conductor_z_dir):
        conductor_z, _ = nrrd.read(conductor_z_dir)
    else:
        conductor_z = np.zeros_like(volume)
        for z in range(d): conductor_z[z, :, :] = skeletonize(conductor_zo[z, : , :])

        conductor_z = generate_mask(conductor_z, num_worker)
        nrrd.write(conductor_z_dir, conductor_z)

    # along x
    if os.path.exists(conductor_x_dir):
        conductor_x, _ = nrrd.read(conductor_x_dir)
    else:
        conductor_x = np.zeros_like(volume)
        for x in range(w): conductor_x[:, :, x] = skeletonize(conductor_xo[:, :, x])

        conductor_x = generate_mask(conductor_x.transpose(2, 1, 0), num_worker).transpose(2, 1, 0)
        nrrd.write(conductor_x_dir, conductor_x)

    if (plot):
        axes[2].set_title("Discrete Conductor")
        axes[2].imshow(conductor_z[d//2, :, :], cmap="nipy_spectral", origin="upper")
        axes[7].imshow(conductor_x[:, :, w//2], cmap="nipy_spectral", origin="upper")

    # potential (init)
    potential = np.zeros_like(conductor_z, dtype=float)
    # in some cases, may need some adjustments for better convergence
    # for y in range(h): potential[:, y, :] = (y / h) * 255
    # potential[:, :1, :] = 0
    # potential[:, -1:, :] = 255

    boundary = np.zeros_like(conductor_z, dtype=bool)
    # boundary[:, :1, :] = True
    # boundary[:, -1:, :] = True

    # rescale again
    potential = down_sampling(potential, rescale_b)
    electrode = down_sampling(electrode, rescale_b, False)
    boundary = down_sampling(boundary, rescale_b, False)

    conductor_x = down_sampling(conductor_x, (rescale_b[0], rescale_b[1], 1), False)
    conductor_z = down_sampling(conductor_z, (1, rescale_b[1], rescale_b[2]), False)
    conductor_x = conductor_x[:, :, ::rescale_b[2]]
    conductor_z = conductor_z[::rescale_b[0], :, :]

    d, h, w = potential.shape

    if (plot):
        axes[2].imshow(conductor_z[d//2, :, :], cmap="nipy_spectral", origin="upper")
        axes[7].imshow(conductor_x[:, :, w//2], cmap="nipy_spectral", origin="upper")

    # load init potential if exists
    if os.path.exists(potential_init_dir):
        potential, header = nrrd.read(potential_init_dir)

    # update potential
    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    counts_z = np.zeros((d, 256), dtype=int)
    for z in range(d):
        counts_z[z] = np.bincount(conductor_z[z, :, :].flatten(), minlength=256)

    counts_x = np.zeros((w, 256), dtype=int)
    for x in range(w):
        counts_x[x] = np.bincount(conductor_x[:, :, x].flatten(), minlength=256)

    if (plot): plt.ion()
    for i in range(1501):
        # electrodes should remain constant
        for label, level in electrode_label_level_pairs:
            potential[electrode == label] = level * 255

        # potential (free space)
        pc = update_potential(potential)
        pc[boundary] = potential[boundary]
        pc[electrode > 0] = potential[electrode > 0]
        potential = pc

        # conductor x direction
        if (i%5 == 2):
            pcx = update_conductor(pc, conductor_x, counts_x, axis=2)
            pcx[conductor_x <= 0] = pc[conductor_x <= 0]
            pcx[boundary] = pc[boundary]
            pcx[electrode > 0] = pc[electrode > 0]
            potential = pcx

        # conductor z direction
        if (i%5 == 1):
            pcz = update_conductor(pc, conductor_z, counts_z, axis=0)
            pcz[conductor_z <= 0] = pc[conductor_z <= 0]
            pcz[boundary] = pc[boundary]
            pcz[electrode > 0] = pc[electrode > 0]
            potential = pcz

        # plot
        if (i%10 == 0):
            print('frame ', i)

            if (plot):
                axes[3].set_title("Potential")
                axes[3].imshow(potential[d//2, :, :], cmap=cmap)
                axes[8].imshow(potential[:, :, w//2], cmap=cmap)

            # save the flatten result
            if (i%500 == 0 and i != 0):
                print('update flatten result ...')

                flatten = update_flatten(volume_origin, potential)
                d0, h0, w0 = flatten.shape

                if (plot):
                    axes[4].set_title("Flatten")
                    axes[4].imshow(flatten[d0//2], cmap="gray")
                    axes[9].imshow(flatten[:, :, w0//2], cmap='gray')

                nrrd.write(potential_dir, potential.astype(np.uint8))

            if (plot): plt.pause(0.001)
    if (plot): plt.ioff()

    print('complete')
    if (plot):
        plt.tight_layout()
        plt.show()

