import rasterio
import numpy as np
from sys import argv

if __name__ == '__main__':
    crs_src = argv[1]
    arr_src = argv[2]
    dst_name = argv[3]

    with rasterio.open(crs_src) as src:
        crs = src.crs
        transform = src.transform

    with rasterio.open(arr_src) as src:
        arr = src.read([1,2,3])

    profile = {
        'driver': 'GTiff',
        'dtype': np.float32,
        'nodata': 0,
        'width': arr.shape[1],
        'height': arr.shape[2],
        'count': 3,
        'crs': crs,
        'transform': transform,
        'tiled': True,
        'compress': 'lzw'
    }

    with rasterio.open(dst_name, 'w', **profile) as dst:
        dst.write(arr)
