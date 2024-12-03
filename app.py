import nrrd
import random
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap
from concurrent.futures import ThreadPoolExecutor

def down_sampling(array, rescale, mean=True):
    nz, ny, nx = array.shape
    tz = (nz // rescale) * rescale
    ty = (ny // rescale) * rescale
    tx = (nx // rescale) * rescale
    trimmed_array = array[:tz, :ty, :tx]
    # trimmed_array = array[:, :ty, :tx]

    if (mean):
        downscaled = trimmed_array.reshape(
            # nz, 1,
            tz // rescale, rescale,
            ty // rescale, rescale,
            tx // rescale, rescale
        ).mean(axis=(1, 3, 5)).astype(array.dtype)
    else:
        downscaled = np.max(
            trimmed_array.reshape(
                # nz, 1,
                tz // rescale, rescale,
                ty // rescale, rescale,
                tx // rescale, rescale
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

def generate_mask_stack(skeleton_stack, num_worker=5):
    # graph
    d, h, w = skeleton_stack.shape
    component_list, processing_list = [], []

    def processing(z):
        print('processing graphs ', z)
        G, components = generate_graph(skeleton_stack[z, :, :])
        component_list.append(components)

    with ThreadPoolExecutor(max_workers=num_worker) as executor:
      futures = [ executor.submit(processing, z) for z in range(d) ]

    # mask
    mask_stack = np.zeros_like(skeleton_stack, dtype=np.uint8)

    for z in range(d):
        components = component_list[z][:255]
        random.shuffle(components)
        for i, component in enumerate(components, start=1):
            for node in component: mask_stack[z, node[0], node[1]] = i

    return mask_stack

def update_potential(potential):
    pc_pad = np.pad(potential, pad_width=1, mode='reflect')

    pc  = pc_pad[1:-1, 1:-1,  :-2].copy()
    pc += pc_pad[1:-1, 1:-1,   2:]
    pc += pc_pad[1:-1,  :-2, 1:-1]
    pc += pc_pad[1:-1,   2:, 1:-1]
    pc += pc_pad[ :-2, 1:-1, 1:-1]
    pc += pc_pad[  2:, 1:-1, 1:-1]
    pc /= 6

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

if __name__ == "__main__":
    # params & path
    z, y, x = 3513, 1900, 3400

    rescale, num_worker = 2, 8
    top_electrode_label, bot_electrode_label = 3, 1
    top_electrode_level, bot_electrode_level = 180, 100

    volume_dir = f'/Users/yao/Desktop/field-based-visualization/{z:05}_{y:05}_{x:05}_volume.tif'
    electrode_dir = f'/Users/yao/Desktop/field-based-visualization/{z:05}_{y:05}_{x:05}_mask.nrrd'
    conductor_dir = f'/Users/yao/Desktop/field-based-visualization/{z:05}_{y:05}_{x:05}_fiber.nrrd'

    # plot init
    fig, axes = plt.subplots(1, 6, figsize=(15, 5))
    for ax in axes: ax.axis('off')

    # load volume
    volume = tifffile.imread(volume_dir)
    volume = down_sampling(volume, rescale)
    d, h, w = volume.shape

    # load electrode
    electrode, header = nrrd.read(electrode_dir)
    electrode = np.asarray(electrode)
    electrode = down_sampling(electrode, rescale, False)

    electrode_temp = np.zeros_like(electrode)

    for label in [top_electrode_label, bot_electrode_label]:
        print('Processing electrode:', label)
        mask_label = (electrode == label).astype(np.uint8)

        for z in range(d):
            skeleton = skeletonize(mask_label[z])
            electrode_temp[z][skeleton] = label

    electrode = electrode_temp

    axes[0].set_title("Volume")
    axes[0].imshow(volume[d//2, :, :], cmap='gray')
    axes[0].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)

    # load conductor
    conductor, header = nrrd.read(conductor_dir)
    conductor = np.asarray(conductor).astype(bool)
    conductor = down_sampling(conductor, rescale, False)

    for z in range(d): conductor[z] = skeletonize(conductor[z])

    axes[1].set_title("Conductor")
    axes[1].imshow(conductor[d//2, :, :], cmap='gray')
    axes[1].contour(electrode[d//2, :, :] * 255, colors='blue', linewidths=0.5)

    # conductor along z & x
    conductor_z = generate_mask_stack(conductor, num_worker)

    s = conductor.transpose(2, 1, 0)
    conductor_x = generate_mask_stack(s, num_worker)
    conductor_x = conductor_x.transpose(2, 1, 0)

    axes[2].set_title("Discrete Conductor Z")
    axes[3].set_title("Discrete Conductor X")
    axes[2].imshow(conductor_z[d//2, :, :], cmap="nipy_spectral", origin="upper")
    axes[3].imshow(conductor_x[:, :, w//2], cmap="nipy_spectral", origin="upper")

    # potential (init)
    potential = np.zeros_like(conductor_z, dtype=float)
    for y in range(h): potential[:, y, :] = (1 - (y / h)) * 255

    # rescale again
    potential = down_sampling(potential, rescale)
    electrode = down_sampling(electrode, rescale, False)
    conductor_x = down_sampling(conductor_x, rescale, False)
    conductor_z = down_sampling(conductor_z, rescale, False)

    d, h, w = potential.shape

    # update potential
    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)

    counts_z = np.zeros((d, 256), dtype=int)
    for z in range(d):
        counts_z[z] = np.bincount(conductor_z[z, :, :].flatten(), minlength=256)

    counts_x = np.zeros((w, 256), dtype=int)
    for x in range(w):
        counts_x[x] = np.bincount(conductor_x[:, :, x].flatten(), minlength=256)

    plt.ion()
    for i in range(3000):
        potential[electrode == top_electrode_label] = top_electrode_level
        potential[electrode == bot_electrode_label] = bot_electrode_level

        pc = update_potential(potential)
        pc[electrode > 0] = potential[electrode > 0]
        potential = pc

        if (i%5 == 2):
            pcx = update_conductor(pc, conductor_x, counts_x, axis=2)
            pcx[conductor_x <= 0] = pc[conductor_x <= 0]
            pcx[electrode > 0] = pc[electrode > 0]
            potential = pcx

        if (i%5 == 1):
            pcz = update_conductor(pc, conductor_z, counts_z, axis=0)
            pcz[conductor_z <= 0] = pc[conductor_z <= 0]
            pcz[electrode > 0] = pc[electrode > 0]
            potential = pcz

        if (i%100 == 0):
            print('frame ', i)

            axes[4].set_title("Potential Slice Z")
            axes[5].set_title("Potential Slice X")
            axes[4].imshow(potential[d//2, :, :], cmap=cmap)
            axes[5].imshow(potential[:, :, w//2], cmap=cmap)

            plt.pause(0.001)
    plt.ioff()

    plt.tight_layout()
    plt.show()

