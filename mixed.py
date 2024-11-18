import cv2
import nrrd
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap
from app import process_slice_to_point
from scipy.ndimage import convolve

def update_potential(potential, boundary):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

    binary_mask = (mask > 0).astype(int)
    sums = convolve(potential, kernel, mode='constant', cval=0)

    updated_potential = sums / 9
    updated_potential[ 0,  :] = potential[ 1,  :]
    updated_potential[-1,  :] = potential[-2,  :]
    updated_potential[ :,  0] = potential[ :,  1]
    updated_potential[ :, -1] = potential[ :, -2]

    updated_potential[boundary] = potential[boundary]
    return updated_potential

def update_mask(potential, mask, boundary, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size))


    binary_mask = (mask > 0).astype(int)
    counts = convolve(binary_mask, kernel, mode='constant', cval=0)
    counts[counts == 0] = 1

    sums = convolve(potential * binary_mask, kernel, mode='constant', cval=0)
    updated_potential = sums / counts

    updated_potential[mask <= 0] = potential[mask <= 0]
    updated_potential[boundary] = potential[boundary]
    return updated_potential

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
    z, y, x, layer = 3513, 1900, 3400, 0
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    # plot init
    fig, axes = plt.subplots(2, 3, figsize=(7, 3))

    axes = axes.ravel()
    for ax in axes: ax.axis('off')

    # load tif image
    tif_image = tifffile.imread(tif_dir)
    tif_image = tif_image[:, 140:600, :]
    tif_image = tif_image[layer]

    h, w = tif_image.shape
    axes[0].imshow(tif_image, cmap='gray')

    # load boundary
    boundary, header = nrrd.read(mask_dir)
    boundary = boundary[:, 140:600, :]
    boundary = boundary[layer]
    top_label, bot_label = 3, 1

    # blur
    blur_image = cv2.GaussianBlur(tif_image, (3, 3), 0)

    # edge
    edge_image = cv2.Canny(blur_image, threshold1=99, threshold2=100)
    axes[1].imshow(edge_image, cmap='gray')

    # mask
    mask = np.zeros_like(tif_image)
    mask[edge_image > 100] = 255
    axes[2].imshow(mask, cmap="nipy_spectral", origin="upper")

    # extract top, bottom boundary points
    num_points, interval = 500, None

    skeleton_top = np.zeros_like(boundary, dtype=bool)
    skeleton_bot = np.zeros_like(boundary, dtype=bool)
    skeleton_top[boundary == top_label] = True
    skeleton_bot[boundary == bot_label] = True

    skeleton_top = skeletonize(skeleton_top)
    skeleton_bot = skeletonize(skeleton_bot)

    selected_points_top, start, end = process_slice_to_point(skeleton_top, interval, num_points)
    selected_points_bottom, _, _ = process_slice_to_point(skeleton_bot, interval, num_points, start, end)

    axes[3].imshow(tif_image, cmap='gray')
    x_coords, y_coords = zip(*selected_points_top)
    axes[3].scatter(x_coords, y_coords, c='red', s=1)
    x_coords, y_coords = zip(*selected_points_bottom)
    axes[3].scatter(x_coords, y_coords, c='green', s=1)

    # potential (init)
    potential = np.ones_like(mask, dtype=float) * 128
    potential[boundary == top_label] = 255
    potential[boundary == bot_label] = 0

    boundary_mask = np.zeros_like(mask, dtype=bool)
    boundary_mask[boundary == top_label] = True
    boundary_mask[boundary == bot_label] = True
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True

    points_top = np.array(selected_points_top)
    points_bottom = np.array(selected_points_bottom)
    levels = np.linspace(255, 0, num=1000)

    for pt_top, pt_bottom in zip(points_top, points_bottom):
        # outside the boundary
        x_top, y_top = pt_top.astype(int)
        x_bot, y_bot = pt_bottom.astype(int)

        potential[:y_top, x_top: x_top + 10] = 255
        potential[y_bot:, x_bot: x_bot + 10] = 0
        boundary_mask[:y_top, x_top: x_top + 10] = True
        boundary_mask[y_bot:, x_bot: x_bot + 10] = True

        # inside the boundary
        line_points = np.linspace(pt_top, pt_bottom, num=1000).astype(int)
        for (x, y), level in zip(line_points, levels): potential[y, x] = level

    colors = ['#000000', '#ffffff'] * 20
    cmap = ListedColormap(colors)
    axes[4].imshow(potential, cmap=cmap)

    # dynamic plot (potential & flatten image)
    num_depth = 200
    counts = np.bincount(mask.flatten(), minlength=256)
    flatten_image = np.zeros((num_depth, num_points), dtype=np.uint8)

    plt.ion()
    for i in range(10000):
        potential = update_potential(potential, boundary_mask)

        potential = update_mask(potential, mask, boundary_mask, kernel_size=3)

        if (i%100 == 0):
            print(i)

            axes[4].imshow(potential, cmap=cmap)

            if (i%1000 == 0):
                points_array = generate_2d_points_array(potential, selected_points_top, selected_points_bottom, num_depth)

                for i in range(num_depth):
                    for j in range(num_points):
                        x, y = points_array[i, j]
                        flatten_image[i, j] = tif_image[int(y), int(x)]

                tifffile.imwrite("extracted_data.tif", flatten_image)
                axes[5].imshow(flatten_image, cmap='gray')

            plt.pause(0.01)
    plt.ioff()

    # show the plot
    plt.show()
