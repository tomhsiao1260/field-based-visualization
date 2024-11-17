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

def update_potential_stack(potential_stack, boundary_stack):
    pc = potential_stack.copy()

    pc[1:-1, 1:-1, 1:-1]  = potential_stack[1:-1, 1:-1,  :-2]
    pc[1:-1, 1:-1, 1:-1] += potential_stack[1:-1, 1:-1,   2:]
    pc[1:-1, 1:-1, 1:-1] += potential_stack[1:-1,  :-2, 1:-1]
    pc[1:-1, 1:-1, 1:-1] += potential_stack[1:-1,   2:, 1:-1]
    pc[1:-1, 1:-1, 1:-1] += potential_stack[  2:, 1:-1, 1:-1]
    pc[1:-1, 1:-1, 1:-1] += potential_stack[  2:, 1:-1, 1:-1]
    pc[1:-1, 1:-1, 1:-1] /= 6

    pc[ :,  :,  0] = pc[ :,  :,  1]
    pc[ :,  :, -1] = pc[ :,  :, -2]
    pc[ :,  0,  :] = pc[ :,  1,  :]
    pc[ :, -1,  :] = pc[ :, -2,  :]
    pc[ 0,  :,  :] = pc[ 1,  :,  :]
    pc[-1,  :,  :] = pc[-2,  :,  :]

    pc[boundary_stack] = potential_stack[boundary_stack]
    return pc

def update_mask_stack(potential_stack, mask_stack, counts_stack):
    pc = potential_stack.copy()
    d = potential_stack.shape[0]

    for l in range(d):
        sums = np.bincount(mask_stack[l].flatten(), weights=pc[l].flatten(), minlength=256)
        averages = np.zeros_like(sums)
        nonzero = counts_stack[l] > 0
        averages[nonzero] = sums[nonzero] / counts_stack[l][nonzero]
        pc[l] = averages[mask_stack[l]]

    pc[mask_stack == 0] = potential_stack[mask_stack == 0]
    return pc

if __name__ == "__main__":
    z, y, x, layer, d = 3513, 1900, 3400, 100, 100
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # load tif stack
    tif_stack = tifffile.imread(tif_dir)
    tif_stack = tif_stack[layer:layer+d, 140:600, :]

    d, h, w = tif_stack.shape
    axes[0].imshow(tif_stack[0], cmap='gray')

    # load boundary
    boundary, header = nrrd.read(mask_dir)
    boundary = boundary[layer:layer+d, 140:600, :]
    top_label, bot_label = 3, 1

    # blur & edge & skeleton
    blur_stack = np.zeros_like(tif_stack)
    edge_stack = np.zeros_like(tif_stack)
    skeleton_stack = np.zeros_like(tif_stack, dtype=bool)

    for l in range(d):
        blur_stack[l] = cv2.GaussianBlur(tif_stack[l], (3, 3), 0)
        edge_stack[l] = cv2.Canny(blur_stack[l], threshold1=99, threshold2=100)
        skeleton_stack[l] = skeletonize(edge_stack[l])

    axes[1].imshow(blur_stack[0], cmap='gray')
    axes[2].imshow(edge_stack[0], cmap='gray')

    # # graph
    # G_list, component_list = [], []

    # for l in range(d):
    #     print('processing graphs ', l)
    #     G, components = generate_graph(skeleton_stack[l])
    #     component_list.append(components)
    #     G_list.append(G)

    # G, components = G_list[0], component_list[0]
    # pos = {node: (node[1], node[0]) for node in G.nodes}
    # colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

    # for component, color in zip(components, colors):
    #     nx.draw_networkx_nodes(G, pos, nodelist=list(component), node_size=0.01, node_color=[color], ax=axes[3])
    #     nx.draw_networkx_edges(G, pos, edgelist=G.subgraph(component).edges, edge_color=[color], ax=axes[3])
    # axes[3].imshow(skeleton_stack[0], cmap='gray')

    # mask
    mask_stack = np.zeros_like(tif_stack, dtype=np.uint8)

    # for l in range(d):
    #     for i, component in enumerate(component_list[l][:255], start=1):
    #         for node in component: mask_stack[l, node[0], node[1]] = i

    axes[4].imshow(mask_stack[0], cmap="nipy_spectral", origin="upper")

    # extract top, bottom boundary points
    num_points, interval = 500, None
    points_top_list, points_bottom_list = [], []

    skeleton_top = np.zeros_like(boundary, dtype=bool)
    skeleton_bot = np.zeros_like(boundary, dtype=bool)
    skeleton_top[boundary == top_label] = True
    skeleton_bot[boundary == bot_label] = True

    for l in range(d):
        skeleton_top[l] = skeletonize(skeleton_top[l])
        skeleton_bot[l] = skeletonize(skeleton_bot[l])

        points_top, start, end = process_slice_to_point(skeleton_top[l], interval, num_points)
        points_bottom, _, _ = process_slice_to_point(skeleton_bot[l], interval, num_points, start, end)

        points_top_list.append(points_top)
        points_bottom_list.append(points_bottom)

    axes[5].imshow(tif_stack[0], cmap='gray')
    x_coords, y_coords = zip(*points_top_list[0])
    axes[5].scatter(x_coords, y_coords, c='red', s=1)
    x_coords, y_coords = zip(*points_bottom_list[0])
    axes[5].scatter(x_coords, y_coords, c='green', s=1)

    # potential (init)
    potential_stack = np.ones_like(mask_stack, dtype=float) * 128
    potential_stack[boundary == top_label] = 255
    potential_stack[boundary == bot_label] = 0

    points_top_stack = np.array(points_top_list)
    points_bottom_stack = np.array(points_bottom_list)
    levels = np.linspace(255, 0, num=1000)

    for l in range(d):
        for pt_top, pt_bottom in zip(points_top_stack[l], points_bottom_stack[l]):
            # outside the boundary
            x_top, y_top = pt_top.astype(int)
            x_bot, y_bot = pt_bottom.astype(int)
            potential_stack[l, :y_top, x_top: x_top + 10] = 255
            potential_stack[l, y_bot:, x_bot: x_bot + 10] = 0

            # inside the boundary
            line_points = np.linspace(pt_top, pt_bottom, num=1000).astype(int)
            for (x, y), level in zip(line_points, levels): potential_stack[l, y, x] = level

    boundary_mask_stack = np.zeros_like(mask_stack, dtype=bool)
    boundary_mask_stack[boundary == top_label] = True
    boundary_mask_stack[boundary == bot_label] = True
    boundary_mask_stack[:, 0, :] = True
    boundary_mask_stack[:, -1, :] = True

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)
    axes[6].imshow(potential_stack[0], cmap=cmap)

    # dynamic plot (potential & flatten image)
    # counts_stack = np.zeros((d, 256), dtype=int)
    # for l in range(d):
    #     counts_stack[l] = np.bincount(mask_stack[l].flatten(), minlength=256)

    plt.ion()
    for i in range(10000):
        potential_stack = update_potential_stack(potential_stack, boundary_mask_stack)

        # if (i%10 == 0):
        #     # boundary_mask_stack[mask_stack > 0] = True
        #     potential_stack = update_mask_stack(potential_stack, mask_stack, counts_stack)

        if (i%100 == 0):
            print('frame ', i)

            axes[6].imshow(potential_stack[10], cmap=cmap)
            axes[7].imshow(potential_stack[40], cmap=cmap)
            axes[8].imshow(potential_stack[70], cmap=cmap)

            plt.pause(0.01)
    plt.ioff()

    # show the plot
    plt.show()




