import os
import nrrd
import tifffile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def main(output_dir, mask_dir, layer, label, interval, plot):
    # load mask
    data, header = nrrd.read(os.path.join(output_dir, mask_dir))
    data = np.asarray(data)

    # original
    image = np.zeros_like(data[layer], dtype=np.uint8)
    image[data[layer] == label] = 255
    # tifffile.imwrite(os.path.join(output_dir, 'output.tif'), image)

    # skeletonize
    mask = skeletonize(image)
    skeleton_image = np.zeros_like(image)
    skeleton_image[mask] = 255
    # tifffile.imwrite(os.path.join(output_dir, 'skeleton.tif'), skeleton_image)

    if (True):
    # if (plot):
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton_image, cmap='gray')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--label', type=int, default=1, help='Selected label')
    parser.add_argument('--plot', action='store_true', help='Plot the result')
    parser.add_argument('--d', type=int, default=5, help='Interval between each points or layers')
    args = parser.parse_args()

    z, y, x, layer = 3513, 1900, 3400, 0
    label, interval, plot = args.label, args.d, args.plot
    output_dir = '/Users/yao/Desktop/distort-space-test'
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    main(output_dir, mask_dir, layer, label, interval, plot)