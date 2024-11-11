import nrrd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    z, y, x, layer = 3513, 1900, 3400, 0
    mask_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_mask.nrrd'

    mask, header = nrrd.read(mask_dir)
    mask = mask[layer]

    mask_top = np.zeros_like(mask)
    mask_bot = np.zeros_like(mask)

    mask_top[mask == 3] = 255
    mask_bot[mask == 1] = 255

    potential = np.zeros_like(mask).astype(float)
    potential[mask_top == 255] = 255
    potential[mask_bot == 255] = 50
    boundary = potential.copy()

    for i in range(1000):
        pc = potential.copy()

        pc[1:-1, 1:-1]  = potential[1:-1, :-2]
        pc[1:-1, 1:-1] += potential[1:-1, 2:]
        pc[1:-1, 1:-1] += potential[:-2, 1:-1]
        pc[1:-1, 1:-1] += potential[2:, 1:-1]
        pc[1:-1, 1:-1] /= 4

        pc[boundary == 255] = 255
        pc[boundary == 50] = 50

        potential = pc

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(boundary, cmap='gray', vmin=0, vmax=255)
    axes[1].imshow(potential, cmap='gray', vmin=0, vmax=255)

    plt.axis("off")
    plt.show()