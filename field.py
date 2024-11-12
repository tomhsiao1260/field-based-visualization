import nrrd
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def torch_to_tif(tensor):
    volume = tensor.numpy()
    volume = np.abs(volume)
    volume = volume.astype(np.uint8)
    return volume

def resize(data, size=(100, 100)):
    h, w = data.shape[:2]
    rh, rw = size

    sth, stw = h // rh, w // rw

    resize_data = np.zeros((rh, rw), dtype=data.dtype)
    for i in range(rh):
        for j in range(rw):
            window = data[i*sth:(i+1)*sth, j*stw:(j+1)*stw]
            resize_data[i, j] = np.max(window)
    
    return resize_data

if __name__ == '__main__':
    z, y, x, layer = 3513, 1900, 3400, 0
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    mask, header = nrrd.read(mask_dir)
    mask = mask[layer]
    mask = resize(mask, (256, 256))

    mask_top = np.zeros_like(mask)
    mask_bot = np.zeros_like(mask)

    mask_top[mask == 3] = 255
    mask_bot[mask == 1] = 255

    potential = np.zeros_like(mask).astype(float)
    potential[mask_top == 255] = 255
    potential[mask_bot == 255] = 50
    boundary = potential.copy()

    for i in range(10000):
        pc = potential.copy()

        pc[1:-1, 1:-1]  = potential[1:-1, :-2]
        pc[1:-1, 1:-1] += potential[1:-1, 2:]
        pc[1:-1, 1:-1] += potential[:-2, 1:-1]
        pc[1:-1, 1:-1] += potential[2:, 1:-1]
        pc[1:-1, 1:-1] /= 4

        pc[boundary == 255] = 255
        pc[boundary == 50] = 50

        potential = pc

    # First Derivative (Green: > 0, Red: < 0)
    thres = 0.3 * 255
    fd = torch.load('first_derivative.pt') * 255
    tensor = torch.zeros(fd.shape + (3,))
    tensor[..., 0][fd < -thres] = fd[fd < -thres]
    tensor[..., 1][fd > thres] = fd[fd > thres]

    edge = torch_to_tif(tensor[1:-1, 1:-1, 1:-1])
    edge = np.pad(edge, pad_width=((1, 1), (1, 1), (1, 1), (0, 0)), mode='edge')

    # skeletonize
    mask_red = skeletonize(edge[0,:,:,0])
    mask_green = skeletonize(edge[0,:,:,1])
    skeleton_image_red = np.zeros_like(mask_red)
    skeleton_image_green = np.zeros_like(mask_green)
    skeleton_image_red[mask_red] = 255
    skeleton_image_green[mask_green] = 255

    skeleton_image_red = resize(skeleton_image_red, (256, 256))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(boundary + skeleton_image_red, cmap='gray', vmin=0, vmax=255)
    axes[1].imshow(skeleton_image_red, cmap='gray')
    axes[2].imshow(potential, cmap='gray', vmin=0, vmax=255)

    plt.axis("off")
    plt.show()