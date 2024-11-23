import cv2
import nrrd
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from matplotlib.colors import ListedColormap
from skimage.morphology import skeletonize
from skimage.morphology import skeletonize_3d
from scipy.ndimage import gaussian_filter

# def down_sampling(array, factor, mean=True):
#     nz, ny, nx = array.shape
#     tz = (nz // factor) * factor
#     ty = (ny // factor) * factor
#     tx = (nx // factor) * factor
#     trimmed_array = array[:tz, :ty, :tx]

#     if (mean):
#         downscaled = trimmed_array.reshape(
#             tz // factor, factor,
#             ty // factor, factor,
#             tx // factor, factor
#         ).mean(axis=(1, 3, 5)).astype(array.dtype)
#     else:
#         downscaled = np.median(
#             trimmed_array.reshape(
#                 tz // factor, factor,
#                 ty // factor, factor,
#                 tx // factor, factor
#             ),axis=(1, 3, 5)
#         ).astype(array.dtype)

#     return downscaled

def down_sampling(array, factor, mean=True):
    nz, ny, nx = array.shape
    ty = (ny // factor) * factor
    tx = (nx // factor) * factor
    trimmed_array = array[:, :ty, :tx]

    if (mean):
        downscaled = trimmed_array.reshape(
            nz, 1,
            ty // factor, factor,
            tx // factor, factor
        ).mean(axis=(1, 3, 5)).astype(array.dtype)
    else:
        downscaled = np.max(
            trimmed_array.reshape(
                nz, 1,
                ty // factor, factor,
                tx // factor, factor
            ),axis=(1, 3, 5)
        ).astype(array.dtype)

    return downscaled

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
    components = [c for c in nx.connected_components(G) if len(c) >= 0]
    G_filtered = G.subgraph(nodes for component in components for nodes in component).copy()

    return G_filtered, components

def update_potential_stack(potential_stack):
    pc = potential_stack.copy()

    pc[:, 1:-1, 1:-1]  = potential_stack[:, 1:-1,  :-2]
    pc[:, 1:-1, 1:-1] += potential_stack[:, 1:-1,   2:]
    pc[:, 1:-1, 1:-1] += potential_stack[:,  :-2, 1:-1]
    pc[:, 1:-1, 1:-1] += potential_stack[:,   2:, 1:-1]
    # pc[1:-1, 1:-1, 1:-1] += potential_stack[ :-2, 1:-1, 1:-1]
    # pc[1:-1, 1:-1, 1:-1] += potential_stack[  2:, 1:-1, 1:-1]
    pc[:, 1:-1, 1:-1] /= 4

    pc[ :,  :,  0] = pc[ :,  :,  1]
    pc[ :,  :, -1] = pc[ :,  :, -2]
    pc[ :,  0,  :] = pc[ :,  1,  :]
    pc[ :, -1,  :] = pc[ :, -2,  :]
    # pc[ 0,  :,  :] = pc[ 1,  :,  :]
    # pc[-1,  :,  :] = pc[-2,  :,  :]

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
    z, y, x, layer, factor = 3513, 1900, 3400, 100, 5
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    fig, axes = plt.subplots(3, 3, figsize=(7, 5))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # load tif stack
    tif_stack_origin = tifffile.imread(tif_dir)
    # tif_stack_origin = tif_stack_origin[:2, :, :]
    tif_stack_origin = tif_stack_origin[:2, 140:600, :]
    tif_stack = down_sampling(tif_stack_origin, factor)

    d, h, w = tif_stack.shape
    d0, h0, w0 = tif_stack_origin.shape

    axes[0].imshow(tif_stack[d//2], cmap='gray')

    # load & smooth boundary
    boundary, header = nrrd.read(mask_dir)
    # boundary = boundary[:2, :, :]
    boundary = boundary[:2, 140:600, :]
    boundary_temp = np.zeros_like(boundary)
    top_label, bot_label = 3, 1

    for label in [top_label, bot_label]:
        print('Processing label:', label)
        mask_label = (boundary == label).astype(np.uint8)
        smoothed = gaussian_filter(mask_label.astype(float), sigma=[5, 5, 5])
        skeleton = skeletonize_3d(smoothed > 0.5)
        boundary_temp[smoothed > 0.5] = label
        # boundary_temp[skeleton] = label

    boundary = boundary_temp
    boundary = down_sampling(boundary, factor, False)

    # blur & edge & skeleton
    blur_stack = np.zeros_like(tif_stack)
    edge_stack = np.zeros_like(tif_stack)
    skeleton_stack = np.zeros_like(tif_stack, dtype=bool)

    for l in range(d):
        blur_stack[l] = cv2.GaussianBlur(tif_stack[l], (7, 7), 0)
        edge_stack[l] = cv2.Canny(blur_stack[l], threshold1=90, threshold2=100)
        skeleton_stack[l] = skeletonize(edge_stack[l])

    axes[1].imshow(blur_stack[d//2], cmap='gray')

    # graph
    G_list, component_list = [], []

    for l in range(d):
        print('processing graphs ', l)
        G, components = generate_graph(skeleton_stack[l])
        component_list.append(components)
        G_list.append(G)

    G, components = G_list[d//2], component_list[d//2]
    pos = {node: (node[1], node[0]) for node in G.nodes}
    colors = plt.cm.tab20(np.linspace(0, 1, len(components)))

    for component, color in zip(components, colors):
        nx.draw_networkx_nodes(G, pos, nodelist=list(component), node_size=0.01, node_color=[color], ax=axes[2])
        nx.draw_networkx_edges(G, pos, edgelist=G.subgraph(component).edges, edge_color=[color], ax=axes[2])
    axes[2].imshow(skeleton_stack[d//2], cmap='gray')

    # mask
    mask_stack = np.zeros_like(tif_stack, dtype=np.uint8)

    for l in range(d):
        for i, component in enumerate(component_list[l][:255], start=1):
            if (i == top_label or i == bot_label): continue
            for node in component: mask_stack[l, node[0], node[1]] = i

    mask_stack[boundary == top_label] = top_label
    mask_stack[boundary == bot_label] = bot_label
    # mask_stack[:, :110//factor, :] = 0

    axes[3].imshow(mask_stack[d//2], cmap="nipy_spectral", origin="upper")

    # potential (init)
    potential_stack = np.zeros_like(mask_stack, dtype=float)
    for y in range(h): potential_stack[:, y, :] = (1 - (y / h)) * 255
    potential_stack[:, :1, :] = 255
    potential_stack[:, -1:, :] = 0

    boundary_mask_stack = np.zeros_like(mask_stack, dtype=bool)
    boundary_mask_stack[:, :1, :] = True
    boundary_mask_stack[:, -1:, :] = True

    axes[4].imshow(potential_stack[d//2], cmap='gray')

    # update potential
    flatten_stack = np.zeros((d0, h0, w0), dtype=np.uint8)
    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    counts_stack = np.zeros((d, 256), dtype=int)
    for l in range(d):
        counts_stack[l] = np.bincount(mask_stack[l].flatten(), minlength=256)

    # potential_stack = tifffile.imread("potential_.tif").astype(float)
    nrrd.write("mask_template.nrrd", np.zeros((d0, h0, w0), dtype=np.uint8))

    plt.ion()
    for i in range(1000):
        pc = update_potential_stack(potential_stack)
        pc[boundary_mask_stack] = potential_stack[boundary_mask_stack]

        pcm = update_mask_stack(pc, mask_stack, counts_stack)
        pcm[mask_stack <= 0] = pc[mask_stack <= 0]
        pcm[boundary_mask_stack] = pc[boundary_mask_stack]

        potential_stack = pcm

        potential_stack[boundary == top_label] = 200
        potential_stack[boundary == bot_label] = 50

        if (i%10 == 0):
            print('frame ', i)

            axes[4].imshow(potential_stack[d//2], cmap=cmap)
            axes[5].imshow(potential_stack[:, :, w//2], cmap=cmap)

            if (i%100 == 0):
                selected_points_top = [(x, 0) for x in range(w)]
                selected_points_bottom = [(x, h-1) for x in range(w)]

                for z in range(d0):
                    points_array = generate_2d_points_array(potential_stack[z//factor-1], selected_points_top, selected_points_bottom, h)

                    map_x = points_array[..., 0].astype(np.float32) * (w0 / w)
                    map_y = points_array[..., 1].astype(np.float32) * (h0 / h)

                    map_x = cv2.resize(map_x, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    map_y = cv2.resize(map_y, (w0, h0), interpolation=cv2.INTER_LINEAR)

                    flatten_stack[z] = cv2.remap(tif_stack_origin[z], map_x, map_y, interpolation=cv2.INTER_LINEAR)

                axes[6].imshow(flatten_stack[d0//2], cmap="gray")
                axes[7].imshow(flatten_stack[:, :, w0//2], cmap='gray')

                nrrd.write("extracted_data.nrrd", flatten_stack.astype(np.uint8))
                tifffile.imwrite("extracted_data.tif", flatten_stack.astype(np.uint8))
                tifffile.imwrite("potential.tif", potential_stack.astype(np.uint8))

            plt.pause(0.01)
    plt.ioff()

    # show the plot
    plt.show()

