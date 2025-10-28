import rasterio
import random
from sys import argv
import os

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

    subarrs = []
    subcoords = []
    for i in range(subimg_count):
        x, y = (random.randint(0, arr.shape[0] - width - 1),
                random.randint(0, arr.shape[1] - height - 1))
        subcoords.append((y, x))

    with rasterio.open(img_path) as src:
        for i, coord in enumerate(subcoords):
            # Create a window and calculate the transform from the source offset
            x, y = coord
            window = rasterio.windows.Window(x, y, width, height)
            transform = src.window_transform(window)

            # Create a new cropped raster to write to
            profile = src.profile
            profile.update({
                'height': height,
                'width': width,
                'transform': transform,
            })

            # Read the data from the window into the new raster
            with rasterio.open(f"{dest_dir}/{prefix}-{i}.jp2", "w", **profile) as f:
                f.write(src.read(window=window))
