import os
import nrrd
import tifffile
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from skimage.morphology import skeletonize

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

    return endpoints, G_filtered

def find_distance(path_coords):
    distances = [0]
    for i in range(1, len(path_coords)):
        dist = euclidean(path_coords[i-1], path_coords[i])
        distances.append(distances[-1] + dist)
    return distances

# select points (interp via distant)
def find_points(path_coords, distances, interval, num_points=None):
    total_length = distances[-1]

    if not num_points:
        num_points = int(total_length // interval) + 1
    else:
        interval = total_length / float(num_points - 1)

    idx = 1
    current_distance = interval
    selected_points = [path_coords[0]]

    for i in range(1, num_points - 1):
        while idx < len(distances) and distances[idx] < current_distance:
            idx += 1
        if idx == len(distances):
            break
        if distances[idx] == current_distance:
            selected_points.append(path_coords[idx])
        else:
            prev_point = np.array(path_coords[idx -1])
            next_point = np.array(path_coords[idx])
            ratio = (current_distance - distances[idx -1]) / (distances[idx] - distances[idx -1])
            interp_point = prev_point + ratio * (next_point - prev_point)
            selected_points.append(tuple(interp_point))
        current_distance += interval

    selected_points.append(path_coords[-1])

    return selected_points

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

def generate_2d_points_array(points_top, points_bottom, num_points):
    points_top = np.array(points_top)
    points_bottom = np.array(points_bottom)

    # y, x, 2
    points_array = np.zeros((50, num_points, 2))

    # points_array[0] = points_top
    # points_array[-1] = points_bottom
    points_array = np.linspace(points_top, points_bottom, 50, axis=0)

    return points_array

def main(output_dir, mask_dir, layer, interval, plot, num_points):
    label_top, label_bottom = 3, 1

    # load mask
    data, header = nrrd.read(os.path.join(output_dir, mask_dir))
    data = np.asarray(data)

    # original
    image_top = np.zeros_like(data[layer], dtype=np.uint8)
    image_top[data[layer] == label_top] = 255

    image_bottom = np.zeros_like(data[layer], dtype=np.uint8)
    image_bottom[data[layer] == label_bottom] = 255

    # skeletonize
    mask_top = skeletonize(image_top)
    skeleton_image_top = np.zeros_like(mask_top)
    skeleton_image_top[mask_top] = 255

    mask_bottom = skeletonize(image_bottom)
    skeleton_image_bottom = np.zeros_like(mask_bottom)
    skeleton_image_bottom[mask_bottom] = 255

    # find endpoints & shortest path
    endpoints_top, G_top = find_endpoints(skeleton_image_top)
    endpoints_bottom, G_bottom = find_endpoints(skeleton_image_bottom)

    selected_points_top, start, end = process_slice_to_point(skeleton_image_top, interval, num_points)
    selected_points_bottom, _, _ = process_slice_to_point(skeleton_image_bottom, interval, num_points, start, end)

    points_array = generate_2d_points_array(selected_points_top, selected_points_bottom, num_points)

    # if (True):
    if (plot):
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton_image_top, cmap='gray')

        for i in range(points_array.shape[0]):
            plt.scatter(points_array[i, :, 0], points_array[i, :, 1], s=10, color=plt.cm.viridis(i / 50))

        x_coords, y_coords = zip(*selected_points_top)
        plt.scatter(x_coords, y_coords, c='red', s=int(interval // 3))
        x_coords, y_coords = zip(*selected_points_bottom)
        plt.scatter(x_coords, y_coords, c='green', s=int(interval // 3))

        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--plot', action='store_true', help='Plot the result')
    parser.add_argument('--d', type=int, default=5, help='Interval between each points or layers')
    args = parser.parse_args()

    num_points = 150
    z, y, x, layer = 3513, 1900, 3400, 0
    interval, plot = args.d, args.plot
    output_dir = '/Users/yao/Desktop/distort-space-test'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    main(output_dir, mask_dir, layer, interval, plot, num_points)
