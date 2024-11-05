import torch
import tifffile
import numpy as np

import random
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def torch_to_tif(tensor):
    volume = tensor.numpy()
    volume = np.abs(volume)
    volume = volume.astype(np.uint8)
    return volume

if __name__ == "__main__":
    z, y, x, layer = 3513, 1900, 3400, 0
    output_dir = '/Users/yao/Desktop/distort-space-test'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'

    # Region you want to see (z, y, x)
    start_point = (0, 200, 0)
    box_size = (300, 300, 300)

    # Load tif data
    volume = tifffile.imread(tif_dir)
    # Crop the volume
    sz, sy, sx = start_point
    bz, by, bx = box_size
    volume = volume[sz:sz+bz, sy:sy+by, sx:sx+bx]
    # tifffile.imwrite('origin.tif', volume)

    # First Derivative (Green: > 0, Red: < 0)
    thres = 0.2 * 255
    fd = torch.load('first_derivative.pt') * 255
    tensor = torch.zeros(fd.shape + (3,))
    tensor[..., 0][fd < -thres] = fd[fd < -thres]
    tensor[..., 1][fd > thres] = fd[fd > thres]

    edge = torch_to_tif(tensor[1:-1, 1:-1, 1:-1])
    edge = np.pad(edge, pad_width=((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')
    # tifffile.imwrite('first_derivative.tif', edge)

    # skeletonize
    mask_red = skeletonize(edge[0,:,:,0])
    mask_green = skeletonize(edge[0,:,:,1])
    skeleton_image_red = np.zeros_like(mask_red)
    skeleton_image_green = np.zeros_like(mask_green)
    skeleton_image_red[mask_red] = 255
    skeleton_image_green[mask_green] = 255

    # if (True):
    if (False):
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton_image_red, cmap='gray')
        plt.show()

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

    print("nodes number:", G.number_of_nodes())
    print("edges number:", G.number_of_edges())

    # connected_components = list(nx.connected_components(G))
    connected_components = [c for c in nx.connected_components(G) if len(c) >= 100]
    print("graphs number:", len(connected_components))

    # colors = list(plt.cm.tab10.colors)
    # random.shuffle(colors)
    # plt.figure(figsize=(8, 8))

    # for i, component in enumerate(connected_components):
    #     nodes = list(component)
    #     color = colors[i % len(colors)]
    #     nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), nodelist=nodes, node_color=[color])
    #     nx.draw_networkx_edges(G, pos=nx.spring_layout(G), edgelist=G.subgraph(nodes).edges(), edge_color=color)

    # nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    # plt.axis("off")
    # plt.show()

