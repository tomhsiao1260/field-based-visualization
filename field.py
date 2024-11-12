import nrrd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.morphology import skeletonize

import random
import networkx as nx

def torch_to_tif(tensor):
    volume = tensor.numpy()
    volume = np.abs(volume)
    volume = volume.astype(np.uint8)
    return volume

def resize(data, size=(100, 100)):
    h, w = data.shape[:2]
    rh, rw = size

    sth, stw = h // rh, w // rw

    resize_data = np.zeros((rh, rw), dtype=data.dtype)
    for i in range(rh):
        for j in range(rw):
            window = data[i*sth:(i+1)*sth, j*stw:(j+1)*stw]
            resize_data[i, j] = np.max(window)
    
    return resize_data

if __name__ == '__main__':
    z, y, x, layer, size = 3513, 1900, 3400, 0, 300
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    mask, header = nrrd.read(mask_dir)
    mask = mask[layer]

    start_point = (0, 200, 0)
    box_size = (300, 300, 300)
    sz, sy, sx = start_point
    bz, by, bx = box_size
    mask = mask[sy:sy+by, sx:sx+bx]
    mask = resize(mask, (size, size))

    mask_top = np.zeros_like(mask)
    mask_bot = np.zeros_like(mask)

    mask_top[mask == 3] = 255
    mask_bot[mask == 1] = 255

    mask_top = skeletonize(mask_top).astype(float)
    mask_bot = skeletonize(mask_bot).astype(float)
    mask_top[mask_top > 0] = 255
    mask_bot[mask_bot > 0] = 255

    potential = np.zeros_like(mask).astype(float)
    potential[mask_top == 255] = 255
    potential[mask_bot == 255] = 50
    boundary = potential.copy()

    # First Derivative (Green: > 0, Red: < 0)
    thres = 0.3 * 255
    fd = torch.load('first_derivative.pt') * 255
    tensor = torch.zeros(fd.shape + (3,))
    tensor[..., 0][fd < -thres] = fd[fd < -thres]
    tensor[..., 1][fd > thres] = fd[fd > thres]

    edge = torch_to_tif(tensor[1:-1, 1:-1, 1:-1])
    edge = np.pad(edge, pad_width=((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')

    # skeletonize
    mask_red = skeletonize(edge[0,:,:,0])
    mask_green = skeletonize(edge[0,:,:,1])
    skeleton_image_red = np.zeros_like(mask_red).astype(float)
    skeleton_image_green = np.zeros_like(mask_green).astype(float)
    skeleton_image_red[mask_red] = 255
    skeleton_image_green[mask_green] = 255

    skeleton_image_red = resize(skeleton_image_red, (size, size))

    # build the connection
    points = np.column_stack(np.where(skeleton_image_red > 0))
    G = nx.Graph()
    for y, x in points:
        G.add_node((y, x))

        # find connection
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                sy, sx = y + dy, x + dx
                if 0 <= sy < skeleton_image_red.shape[0] and 0 <= sx < skeleton_image_red.shape[1]:
                    if skeleton_image_red[sy, sx] > 0:
                        G.add_edge((y, x), (sy, sx))

    # connected_components = list(nx.connected_components(G))
    connected_components = [c for c in nx.connected_components(G) if len(c) >= 50]
    print("graphs number:", len(connected_components))

    data = np.zeros((size, size, 3))
    colors = list(plt.cm.tab10.colors)
    random.shuffle(colors)
    mask_list = []

    for i, component in enumerate(connected_components):
        mask = np.full((size, size), False)

        for node in component:
            mask[node[0], node[1]] = True

        data[mask] = colors[i % len(colors)]
        mask_list.append(mask)

    for i in range(20000):
        pc = potential.copy()

        pc[1:-1, 1:-1]  = potential[1:-1, :-2]
        pc[1:-1, 1:-1] += potential[1:-1, 2:]
        pc[1:-1, 1:-1] += potential[:-2, 1:-1]
        pc[1:-1, 1:-1] += potential[2:, 1:-1]
        pc[1:-1, 1:-1] /= 4

        pc[boundary == 255] = 255
        pc[boundary == 50] = 50

        # pc[0, :]  = pc[1, :]
        # pc[-1, :] = pc[-2, :]
        # pc[:, 0]  = pc[:, 1]
        # pc[:, -1] = pc[:, -2]

        # for edge in mask_list: pc[edge] = pc[edge].mean()

        potential = pc

    num_intervals = 20
    boundaries = np.linspace(0, 255, num_intervals + 1)
    colors = plt.cm.tab20(np.random.choice(range(plt.cm.tab20.N), num_intervals, replace=False))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(data)
    axes[1].imshow(potential, cmap=cmap, norm=norm)
    axes[2].imshow(boundary, cmap='gray')
    axes[3].imshow(potential, cmap='gray', vmin=0, vmax=255)

    plt.axis("off")
    plt.show()
