import nrrd
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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

def find_endpoints(skel):
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
    components = [c for c in nx.connected_components(G) if len(c) >= 20]
    G_filtered = G.subgraph(nodes for component in components for nodes in component).copy()

    endpoints = [node for node, degree in G_filtered.degree() if degree == 1]

    # only select 2 endpoints with longest distance
    if len(endpoints) > 2:
        max_dist = 0
        main_endpoints = (None, None)
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                dist = np.linalg.norm(np.array(endpoints[i]) - np.array(endpoints[j]))
                if dist > max_dist:
                    max_dist = dist
                    main_endpoints = (endpoints[i], endpoints[j])
        endpoints = list(main_endpoints)

    # reorder start, end points
    y0, x0 = endpoints[0]
    y1, x1 = endpoints[1]
    if (x0 > x1): endpoints = [(y1, x1), (y0, x0)]

    return endpoints, G_filtered

def find_distance(path_coords):
    distances = [0]
    for i in range(1, len(path_coords)):
        dist = euclidean(path_coords[i-1], path_coords[i])
        distances.append(distances[-1] + dist)
    return distances

def process_slice_to_point(skeleton_image, interval, num_points=None, prev_start=None, prev_end=None):
    # find endpoints & shortest path
    endpoints, G = find_endpoints(skeleton_image)

    if (prev_start is None):
        start, end = endpoints[0], endpoints[1]
    else:
        # need to check start, end order
        (xs, ys), (xe, ye), (xps, yps) = endpoints[0], endpoints[1], prev_start
        p = (xs - xps) ** 2 + (ys - yps) ** 2
        q = (xe - xps) ** 2 + (ye - yps) ** 2
        if (p < q): start, end = endpoints[0], endpoints[1]
        if (p > q): end, start = endpoints[0], endpoints[1]
    # print(f"Start (y, x): {start}, End (y, x): {end}")

    path = nx.shortest_path(G, source=start, target=end)
    path_coords = [(x, y) for y, x in path]

    distances = find_distance(path_coords)
    total_length = distances[-1]
    # print(f"Total distance: {total_length}")

    selected_points = find_points(path_coords, distances, interval, num_points)
    # print("Selected Points:")
    # for point in selected_points: print(point)
    return selected_points, start, end

if __name__ == '__main__':
    z, y, x, layer, size = 3513, 1900, 3400, 0, 300
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'

    mask, header = nrrd.read(mask_dir)
    mask = mask[layer]

    start_point = (0, 200, 0)
    box_size = (300, 300, 300)
    sz, sy, sx = start_point
    bz, by, bx = box_size
    mask = mask[sy:sy+by, sx:sx+bx]
    mask = resize(mask, (size, size))

    image = tifffile.imread(tif_dir)
    image = image[layer]
    image = image[sy:sy+by, sx:sx+bx]
    image = resize(image, (size, size))

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

        for edge in mask_list: pc[edge] = pc[edge].mean()

        potential = pc

    num_intervals = 20
    boundaries = np.linspace(0, 255, num_intervals * 2 + 1)
    colors = ['#000000', '#ffffff'] * num_intervals
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, cmap.N)

    selected_points_list = []
    graph_start, graph_end = None, None
    interval, num_points = None, 100

    i = 10
    edge_image = np.zeros_like(potential, dtype=np.uint8)
    edge_image[(potential > boundaries[i]) & (potential < boundaries[i+1])] = 255
    edge_mask = skeletonize(edge_image)
    skeleton_image = np.zeros_like(edge_mask)
    skeleton_image[edge_mask] = 255

    # for i in range(len(boundaries)-1):
        # edge_image = np.zeros_like(potential, dtype=np.uint8)
        # edge_image[(potential > boundaries[i]) & (potential < boundaries[i+1])] = 255
        # edge_mask = skeletonize(edge_image)
        # skeleton_image = np.zeros_like(edge_mask)
        # skeleton_image[edge_mask] = 255

        # selected_points, start, end = process_slice_to_point(skeleton_image, interval, num_points, graph_start, graph_end)
        # if (i == 0): graph_start, graph_end = start, end

        # selected_points_list.append(selected_points)

    fig, axes = plt.subplots(1, 6, figsize=(5*6, 6))
    axes[2].imshow(data)
    axes[3].imshow(potential, cmap=cmap, norm=norm)
    axes[1].imshow(boundary, cmap='gray')
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[4].imshow(edge_image, cmap='gray', vmin=0, vmax=255)

    # for i, selected_points in enumerate(selected_points_list):
    #     if (i % 2 == 0):
    #         color = '#000000'
    #     else:
    #         color = '#ffffff'
    #     x_coords, y_coords = zip(*selected_points)
    #     axes[5].scatter(x_coords, y_coords, c=colors, s=1)

    for ax in axes: ax.axis("off")
    plt.show()
