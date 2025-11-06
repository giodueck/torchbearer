import matplotlib.pyplot as plt
import rasterio
import random
import numpy as np
from sys import argv

if __name__ == "__main__":
    img_paths = [argv[1], argv[2], argv[3]]
    width = int(argv[4])
    height = int(argv[5])
    subimg_count = int(argv[6])

    with rasterio.open(img_paths[0]) as f:
        arr0 = f.read(1)
    with rasterio.open(img_paths[1]) as f:
        arr1 = f.read(1)
    with rasterio.open(img_paths[2]) as f:
        arr2 = f.read(1)

    arr = np.asarray([arr0, arr1, arr2])
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr/10000, 0, 1)
    arr = np.clip(arr, 0, 0.4)/0.4

    random.seed(123458)

    print(arr.shape)
    subarrs = []
    for i in range(subimg_count):
        x, y = (random.randint(0, arr.shape[0] - width - 1),
                random.randint(0, arr.shape[1] - height - 1))
        subarrs.append(arr[x:x+width, y:y+height, 0:])

    _, axes = plt.subplots(3, subimg_count // 3, figsize=(width, height))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.imshow(subarrs[i])

    plt.show()
