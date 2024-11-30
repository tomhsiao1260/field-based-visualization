import tifffile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    z, y, x, layer = 3513, 1900, 3400, 0
    tif_dir = f'/Users/yao/Desktop/distort-space-test/{z:05}_{y:05}_{x:05}_volume.tif'

    image = tifffile.imread(tif_dir)
    image = image[layer]
    h, w = image.shape

    img = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            xn = float(x) / float(w)
            yn = float(y) / float(h)

            fn = ((xn-0.5) * (xn-0.5) + (yn-0.5) * (yn-0.5)) / 0.5
            img[y][x] = int(fn * 255)

    dy, dx = np.gradient(img) 
    print(dy.shape, dx.shape)
    print(dy[384][700], dx[384][700])
    print(dy[384][100], dx[384][100])
    print(dy[100][384], dx[100][384])
    print(dy[700][384], dx[700][384])

    fig, axes = plt.subplots(1, 2, figsize=(2*6, 6))
    # same level line
    axes[0].contour(img, levels=20, colors='black', linewidths=0.5)
    # gradient direction line
    axes[0].streamplot(np.arange(w), np.arange(h), dx, dy, color='blue', linewidth=0.5)
    axes[0].invert_yaxis()

    axes[1].imshow(image, cmap='gray')

    plt.show()
