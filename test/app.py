import cv2
import nrrd
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from matplotlib.colors import ListedColormap
from skimage.morphology import skeletonize

def down_sampling(array, factor, mean=True):
    nz, ny, nx = array.shape
    tz = (nz // factor) * factor
    ty = (ny // factor) * factor
    tx = (nx // factor) * factor
    trimmed_array = array[:tz, :ty, :tx]
    # trimmed_array = array[:, :ty, :tx]

    if (mean):
        downscaled = trimmed_array.reshape(
            # nz, 1,
            tz // factor, factor,
            ty // factor, factor,
            tx // factor, factor
        ).mean(axis=(1, 3, 5)).astype(array.dtype)
    else:
        downscaled = np.max(
            trimmed_array.reshape(
                # nz, 1,
                tz // factor, factor,
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
    components = [c for c in nx.connected_components(G) if len(c) >= 3]
    G_filtered = G.subgraph(nodes for component in components for nodes in component).copy()

    return G_filtered, components

def update_potential_stack(potential_stack):
    pc_pad = np.pad(potential_stack, pad_width=1, mode='reflect')

    pc  = pc_pad[1:-1, 1:-1,  :-2].copy()
    pc += pc_pad[1:-1, 1:-1,   2:]
    pc += pc_pad[1:-1,  :-2, 1:-1]
    pc += pc_pad[1:-1,   2:, 1:-1]
    pc += pc_pad[ :-2, 1:-1, 1:-1]
    pc += pc_pad[  2:, 1:-1, 1:-1]
    pc /= 6

    return pc

def update_potential_level(potential_stack, boundary):
    counts = np.zeros_like(potential_stack)
    p_avg = potential_stack.copy()
    pc = potential_stack.copy()
    bc = potential_stack.copy()
    bf = boundary.copy().astype(potential_stack.dtype)

    p_avg[1:-1, 1:-1, 1:-1]  = pc[1:-1, 1:-1,  :-2]
    p_avg[1:-1, 1:-1, 1:-1] += pc[1:-1, 1:-1,   2:]
    p_avg[1:-1, 1:-1, 1:-1] += pc[1:-1,  :-2, 1:-1]
    p_avg[1:-1, 1:-1, 1:-1] += pc[1:-1,   2:, 1:-1]
    p_avg[1:-1, 1:-1, 1:-1] += pc[ :-2, 1:-1, 1:-1]
    p_avg[1:-1, 1:-1, 1:-1] += pc[  2:, 1:-1, 1:-1]
    p_avg[1:-1, 1:-1, 1:-1] /= 6

    p_avg[boundary == 0] = 0
    p_avg[boundary == 1] -= 100
    p_avg[boundary == 3] -= 180

    bc[1:-1, 1:-1, 1:-1]  = p_avg[1:-1, 1:-1,  :-2]
    bc[1:-1, 1:-1, 1:-1] += p_avg[1:-1, 1:-1,   2:]
    bc[1:-1, 1:-1, 1:-1] += p_avg[1:-1,  :-2, 1:-1]
    bc[1:-1, 1:-1, 1:-1] += p_avg[1:-1,   2:, 1:-1]
    bc[1:-1, 1:-1, 1:-1] += p_avg[ :-2, 1:-1, 1:-1]
    bc[1:-1, 1:-1, 1:-1] += p_avg[  2:, 1:-1, 1:-1]

    counts[1:-1, 1:-1, 1:-1]  = (boundary[1:-1, 1:-1,  :-2] > 0)
    counts[1:-1, 1:-1, 1:-1] += (boundary[1:-1, 1:-1,  :-2] > 0)
    counts[1:-1, 1:-1, 1:-1] += (boundary[1:-1,  :-2, 1:-1] > 0)
    counts[1:-1, 1:-1, 1:-1] += (boundary[1:-1,   2:, 1:-1] > 0)
    counts[1:-1, 1:-1, 1:-1] += (boundary[ :-2, 1:-1, 1:-1] > 0)
    counts[1:-1, 1:-1, 1:-1] += (boundary[  2:, 1:-1, 1:-1] > 0)

    nonzero = counts > 0
    pc[nonzero] += bc[nonzero] / counts[nonzero] * 0.3
    pc[boundary > 0] = potential_stack[boundary > 0]

    return pc

def update_mask_stack(potential_stack, mask_stack, counts_stack, axis=0):
    pc = potential_stack.copy()
    d = potential_stack.shape[axis]

    for l in range(d):
        counts = counts_stack[l]

        if (axis == 0):
            mask = mask_stack[l, :, :]
            p = pc[l, :, :]
        elif (axis == 1):
            mask = mask_stack[:, l, :]
            p = pc[:, l, :]
        else:
            mask = mask_stack[:, :, l]
            p = pc[:, :, l]

        sums = np.bincount(mask.flatten(), weights=p.flatten(), minlength=256)
        averages = np.zeros_like(sums)
        nonzero = counts > 0
        averages[nonzero] = sums[nonzero] / counts[nonzero]

        if (axis == 0):
            pc[l, :, :] = averages[mask]
        elif (axis == 1):
            pc[:, l, :] = averages[mask]
        else:
            pc[:, :, l] = averages[mask]

    return pc

def generate_mask_stack(skeleton_stack):
    d, h, w = skeleton_stack.shape

    # graph
    G_list, component_list = [], []

    for z in range(d):
        print('processing graphs ', z)
        G, components = generate_graph(skeleton_stack[z, :, :])
        component_list.append(components)
        G_list.append(G)

    # mask
    mask_stack = np.zeros_like(skeleton_stack, dtype=np.uint8)

    for z in range(d):
        for i, component in enumerate(component_list[z][:255], start=1):
            for node in component: mask_stack[z, node[0], node[1]] = i

    return mask_stack

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
    z, y, x, factor = 3513, 1900, 3400, 5
    tif_dir = f'/Users/yao/Desktop/field-based-visualization/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/field-based-visualization/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    fig, axes = plt.subplots(2, 5, figsize=(7, 3))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # load tif stack
    tif_stack_origin = tifffile.imread(tif_dir)
    # tif_stack_origin = tif_stack_origin[:3, :, :]
    # tif_stack_origin = tif_stack_origin[:2, 140:600, :]
    tif_stack = down_sampling(tif_stack_origin, factor)

    d, h, w = tif_stack.shape
    d0, h0, w0 = tif_stack_origin.shape

    axes[0].imshow(tif_stack[d//2, :, :], cmap='gray')
    axes[5].imshow(tif_stack[:, :, w//2], cmap='gray')

    # load & smooth boundary
    top_label, bot_label = 3, 1

    boundary, header = nrrd.read(mask_dir)
    # boundary = boundary[:3, :, :]
    # boundary = boundary[:2, 140:600, :]
    boundary = down_sampling(boundary, factor, False)
    boundary_temp = np.zeros_like(boundary)

    for label in [top_label, bot_label]:
        print('Processing label:', label)
        mask_label = (boundary == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            boundary_temp[z][skeleton] = label

        # boundary_temp[boundary == label] = label

    boundary = boundary_temp

    axes[0].contour(boundary[d//2, :, :] * 255, colors='blue', linewidths=0.5)
    axes[5].contour(boundary[:, :, w//2] * 255, colors='blue', linewidths=0.5)

    # blur & edge & skeleton
    blur_stack_z = np.zeros_like(tif_stack)
    edge_stack_z = np.zeros_like(tif_stack)
    skeleton_stack_z = np.zeros_like(tif_stack, dtype=bool)

    for z in range(d):
        blur_stack_z[z, :, :] = cv2.GaussianBlur(tif_stack[z, :, :], (7, 7), 0)
        edge_stack_z[z, :, :] = cv2.Canny(blur_stack_z[z, :, :], threshold1=90, threshold2=100)
        skeleton_stack_z[z, :, :] = skeletonize(edge_stack_z[z, : , :])

    blur_stack_x = np.zeros_like(tif_stack)
    edge_stack_x = np.zeros_like(tif_stack)
    skeleton_stack_x = np.zeros_like(tif_stack, dtype=bool)

    for x in range(w):
        blur_stack_x[:, :, x] = cv2.GaussianBlur(tif_stack[:, :, x], (7, 7), 0)
        edge_stack_x[:, :, x] = cv2.Canny(blur_stack_x[:, :, x], threshold1=90, threshold2=100)
        skeleton_stack_x[:, :, x] = skeletonize(edge_stack_x[:, :, x])

    axes[1].imshow(blur_stack_z[d//2, :, :], cmap='gray')
    axes[6].imshow(blur_stack_x[:, :, w//2], cmap='gray')

    # mask
    mask_stack_z = generate_mask_stack(skeleton_stack_z)
    # mask_stack_z[:, :110//factor, :] = 0

    s = skeleton_stack_x.transpose(2, 1, 0)
    mask_stack_x = generate_mask_stack(s)
    mask_stack_x = mask_stack_x.transpose(2, 1, 0)
    # mask_stack_x[:, :110//factor, :] = 0

    axes[2].imshow(mask_stack_z[d//2, :, :], cmap="nipy_spectral", origin="upper")
    axes[7].imshow(mask_stack_x[:, :, w//2], cmap="nipy_spectral", origin="upper")

    # potential (init)
    potential_stack = np.zeros_like(mask_stack_z, dtype=float)
    for y in range(h): potential_stack[:, y, :] = (1 - (y / h)) * 255
    potential_stack[:, :1, :] = 255
    potential_stack[:, -1:, :] = 0

    boundary_mask_stack = np.zeros_like(mask_stack_z, dtype=bool)
    boundary_mask_stack[:, :1, :] = True
    boundary_mask_stack[:, -1:, :] = True

    # update potential
    flatten_stack = np.zeros((d0, h0, w0), dtype=np.uint8)
    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    counts_stack_z = np.zeros((d, 256), dtype=int)
    for z in range(d):
        counts_stack_z[z] = np.bincount(mask_stack_z[z, :, :].flatten(), minlength=256)

    counts_stack_x = np.zeros((w, 256), dtype=int)
    for x in range(w):
        counts_stack_x[x] = np.bincount(mask_stack_x[:, :, x].flatten(), minlength=256)

    # potential_stack = tifffile.imread("potential_.tif").astype(float)
    nrrd.write("mask_template.nrrd", np.zeros((d0, h0, w0), dtype=np.uint8))
    top_level, bot_level = 180, 100

    plt.ion()
    for i in range(3000):
        potential_stack[boundary == top_label] = top_level
        potential_stack[boundary == bot_label] = bot_level

        pc = update_potential_stack(potential_stack)
        # pc[boundary_mask_stack] = potential_stack[boundary_mask_stack]
        pc[boundary > 0] = potential_stack[boundary > 0]

        # pc = update_potential_level(pc, boundary)
        # pc[boundary > 0] = potential_stack[boundary > 0]

        potential_stack = pc

        if (i%5 == 2):
            pcx = update_mask_stack(pc, mask_stack_x, counts_stack_x, axis=2)
            pcx[mask_stack_x <= 0] = pc[mask_stack_x <= 0]
            # pcx[boundary_mask_stack] = pc[boundary_mask_stack]
            pcx[boundary > 0] = pc[boundary > 0]
            potential_stack = pcx

        if (i%5 == 1):
            pcz = update_mask_stack(pc, mask_stack_z, counts_stack_z, axis=0)
            pcz[mask_stack_z <= 0] = pc[mask_stack_z <= 0]
            # pcz[boundary_mask_stack] = pc[boundary_mask_stack]
            pcz[boundary > 0] = pc[boundary > 0]
            potential_stack = pcz

        if (i%100 == 0):
            print('frame ', i)

            axes[3].imshow(potential_stack[d//2, :, :], cmap=cmap)
            axes[8].imshow(potential_stack[:, :, w//2], cmap=cmap)

            if (i%500 == 0):
            # if (i == 800 or i == 1300 or i == 1990 or i== 2990):
            # if (i%100 == 0):
                selected_points_top = [(x, 0) for x in range(w)]
                selected_points_bottom = [(x, h-1) for x in range(w)]

                for z in range(d0):
                    points_array = generate_2d_points_array(potential_stack[z//factor-1], selected_points_top, selected_points_bottom, h)

                    map_x = points_array[..., 0].astype(np.float32) * (w0 / w)
                    map_y = points_array[..., 1].astype(np.float32) * (h0 / h)

                    map_x = cv2.resize(map_x, (w0, h0), interpolation=cv2.INTER_LINEAR)
                    map_y = cv2.resize(map_y, (w0, h0), interpolation=cv2.INTER_LINEAR)

                    flatten_stack[z] = cv2.remap(tif_stack_origin[z], map_x, map_y, interpolation=cv2.INTER_LINEAR)
                    flatten_stack[z, :(h0 * (255 - top_level) // 255), :] = 0
                    flatten_stack[z, (h0 * (255 - bot_level) // 255):, :] = 0

                axes[4].imshow(flatten_stack[d0//2], cmap="gray")
                axes[9].imshow(flatten_stack[:, :, w0//2], cmap='gray')

                nrrd.write("extracted_data.nrrd", flatten_stack.astype(np.uint8))
                tifffile.imwrite("extracted_data.tif", flatten_stack.astype(np.uint8))
                tifffile.imwrite("potential.tif", potential_stack.astype(np.uint8))

            plt.pause(0.001)
    plt.ioff()

    # show the plot
    plt.show()

