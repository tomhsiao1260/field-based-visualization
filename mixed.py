import cv2
import nrrd
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap
from app import process_slice_to_point

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
    components = [c for c in nx.connected_components(G) if len(c) >= 50]
    G_filtered = G.subgraph(nodes for component in components for nodes in component).copy()

    return G_filtered, components

def update_potential(potential, boundary):
    pc = potential.copy()

    pc[1:-1, 1:-1]  = potential[1:-1, :-2]
    pc[1:-1, 1:-1] += potential[1:-1, 2:]
    pc[1:-1, 1:-1] += potential[:-2, 1:-1]
    pc[1:-1, 1:-1] += potential[2:, 1:-1]
    pc[1:-1, 1:-1] /= 4

    pc[0, :]  = pc[1, :]
    pc[-1, :] = pc[-2, :]
    pc[:, 0]  = pc[:, 1]
    pc[:, -1] = pc[:, -2]

    pc[boundary] = potential[boundary]
    return pc

def update_mask(potential, mask, counts):
    pc = potential.copy()

    sums = np.bincount(mask.flatten(), weights=pc.flatten(), minlength=256)
    averages = np.zeros_like(sums)
    nonzero = counts > 0
    averages[nonzero] = sums[nonzero] / counts[nonzero]

    pc = averages[mask]
    pc[mask == 0] = potential[mask == 0]
    return pc

def generate_2d_points_array(potential, points_top, points_bottom, num_depth):
    points_top = np.array(points_top)
    points_bottom = np.array(points_bottom)

    points_grid = np.linspace(points_top, points_bottom, num_depth * 5, axis=0).astype(int)
    potential_grid = potential[points_grid[..., 1], points_grid[..., 0]]
    levels = np.linspace(255, 0, num_depth) 

    # y, x, 2
    num_points = points_top.shape[0]
    points_array = np.zeros((num_depth, num_points, 2))

    for i in range(num_points):
        points_array[:, i, 0] = np.interp(levels, potential_grid[:, i][::-1], points_grid[:, i, 0][::-1])
        points_array[:, i, 1] = np.interp(levels, potential_grid[:, i][::-1], points_grid[:, i, 1][::-1])

    return points_array

if __name__ == "__main__":
    z, y, x, layer = 3513, 1900, 3400, 100
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # load tif image
    tif_image = tifffile.imread(tif_dir)
    tif_image = tif_image[layer]
    tif_image = tif_image[140:600]

    h, w = tif_image.shape
    axes[0].imshow(tif_image, cmap='gray')

    # load boundary
    boundary, header = nrrd.read(mask_dir)
    boundary = boundary[layer]
    boundary = boundary[140:600]
    top_label, bot_label = 3, 1

    # blur
    blur_image = cv2.GaussianBlur(tif_image, (3, 3), 0)
    axes[1].imshow(blur_image, cmap='gray')

    # edge
    edge_image = cv2.Canny(blur_image, threshold1=99, threshold2=100)
    axes[2].imshow(edge_image, cmap='gray')

    # skeleton
    skeleton_image = skeletonize(edge_image)

    # graph
    G, components = generate_graph(skeleton_image)
    pos = {node: (node[1], node[0]) for node in G.nodes}

    colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
    for component, color in zip(components, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=list(component), node_size=0.01, node_color=[color], ax=axes[3])
        nx.draw_networkx_edges(G, pos, edgelist=G.subgraph(component).edges, edge_color=[color], ax=axes[3])
    axes[3].imshow(skeleton_image, cmap='gray')

    # mask
    mask = np.zeros_like(tif_image, dtype=np.uint8)

    for i, component in enumerate(components[:255], start=1):
        for node in component: mask[node] = i

    axes[4].imshow(mask, cmap="nipy_spectral", origin="upper")

    # extract top, bottom boundary points
    num_points, interval = 500, None

    skeleton_image_top = np.zeros_like(boundary, dtype=bool)
    skeleton_image_top[boundary == top_label] = True
    skeleton_image_top = skeletonize(skeleton_image_top)

    skeleton_image_bot = np.zeros_like(boundary, dtype=bool)
    skeleton_image_bot[boundary == bot_label] = True
    skeleton_image_bot = skeletonize(skeleton_image_bot)

    selected_points_top, start, end = process_slice_to_point(skeleton_image_top, interval, num_points)
    selected_points_bottom, _, _ = process_slice_to_point(skeleton_image_bot, interval, num_points, start, end)

    axes[5].imshow(tif_image, cmap='gray')
    x_coords, y_coords = zip(*selected_points_top)
    axes[5].scatter(x_coords, y_coords, c='red', s=1)
    x_coords, y_coords = zip(*selected_points_bottom)
    axes[5].scatter(x_coords, y_coords, c='green', s=1)

    # potential
    potential = np.ones_like(mask,  dtype=float) * 128
    potential[boundary == top_label] = 255
    potential[boundary == bot_label] = 0
    potential[0, :] = 255
    potential[-1, :] = 0

    boundary_mask = np.zeros_like(mask, dtype=bool)
    boundary_mask[boundary == top_label] = True
    boundary_mask[boundary == bot_label] = True
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)
    counts = np.bincount(mask.flatten(), minlength=256)

    num_depth = 200
    flatten_image = np.zeros((num_depth, num_points), dtype=np.uint8)

    # dynamic plot (potential & flatten image)
    plt.ion()
    for i in range(15000):
        potential = update_potential(potential, boundary_mask)
        # potential = update_mask(potential, mask, counts)

        if (i%100 == 0):
            print(i)

            axes[6].imshow(potential, cmap=cmap)

            if (i%1000 == 0):
                points_array = generate_2d_points_array(potential, selected_points_top, selected_points_bottom, num_depth)

                for i in range(num_depth):
                    for j in range(num_points):
                        x, y = points_array[i, j]
                        flatten_image[i, j] = tif_image[int(y), int(x)]

                tifffile.imwrite("extracted_data.tif", flatten_image)
                axes[7].imshow(flatten_image, cmap='gray')

            plt.pause(0.01)
    plt.ioff()

    # show the plot
    plt.show()
