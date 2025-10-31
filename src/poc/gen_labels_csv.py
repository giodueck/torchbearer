import rasterio
import random
from sys import argv
import os
import numpy as np
import torch

if __name__ == "__main__":
    img_path = argv[1]
    width = int(argv[2])
    height = int(argv[3])
    subimg_count = int(argv[4])
    dest_dir = argv[5].rstrip("/")
    prefix = argv[6]

    with rasterio.open(img_path) as f:
        arr = f.read(1)

    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        pass

    random.seed(123458)

    with open(f"{dest_dir}/labels.csv", "w") as dst:
        dst.write("filename,label\n")
        for i in range(subimg_count):
            x, y = (random.randint(0, arr.shape[0] - width - 1),
                    random.randint(0, arr.shape[1] - height - 1))
            subimg = arr[x:x+width, y:y+height]

            # Assuming the image is a mask, where white marks the presence of
            # the feature looked for, set the label to true if the average
            # pixel has a value of at least 30, the maximum being 255
            if np.average(subimg) >= 30:
                dst.write(f"{prefix}-{i}.jp2,1\n")
            else:
                dst.write(f"{prefix}-{i}.jp2,0\n")
