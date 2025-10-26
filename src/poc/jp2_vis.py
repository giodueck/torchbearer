import matplotlib.pyplot as plt
import rasterio

from sys import argv

if __name__ == "__main__":
    with rasterio.open(argv[1]) as f:
        arr = f.read(1)

    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)

    print(arr.shape)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(arr, cmap='Grays_r')
    rect = plt.Rectangle((256, 256), 256, 256, ec='blue', fc='none')
    ax0.add_patch(rect)

    cropped_arr = arr[256:512, 256:512]
    ax1 = fig.add_subplot(gs[0, 1])
    print(cropped_arr.shape)
    ax1.imshow(cropped_arr, cmap='Grays_r')

    plt.show()
