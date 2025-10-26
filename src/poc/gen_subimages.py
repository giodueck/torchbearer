import matplotlib.pyplot as plt
import rasterio
import random
from sys import argv

if __name__ == "__main__":
    img_path = argv[1]
    width = int(argv[2])
    height = int(argv[3])
    subimg_count = int(argv[4])

    with rasterio.open(img_path) as f:
        arr = f.read(1)

    random.seed(123458)

    subarrs = []
    for i in range(subimg_count):
        x, y = (random.randint(0, arr.shape[0] - width - 1),
                random.randint(0, arr.shape[1] - height - 1))
        subarrs.append(arr[x:x+width, y:y+height])

    _, axes = plt.subplots(3, subimg_count // 3, figsize=(width, height))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(subarrs[i], cmap='Grays_r')

    plt.show()
