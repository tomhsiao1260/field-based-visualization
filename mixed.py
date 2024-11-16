import cv2
import nrrd
import tifffile
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

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

if __name__ == "__main__":
    z, y, x, layer = 3513, 1900, 3400, 0
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    plot_num = 6
    fig, axes = plt.subplots(1, plot_num, figsize=(plot_num*6, 6))
    for ax in axes: ax.axis("off")

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
    print('jfiwe', len(components))

    for i, component in enumerate(components[:255], start=1):
        for node in component: mask[node] = i

    axes[4].imshow(mask, cmap="nipy_spectral", origin="upper")

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
    plt.ion()

    for i in range(15000):
        potential = update_potential(potential, boundary_mask)
        potential = update_mask(potential, mask, counts)

        if (i%100 == 0):
            print(i)
            axes[5].imshow(potential, cmap=cmap)
            plt.pause(0.01)

    plt.ioff()

    # show the plot
    plt.show()
