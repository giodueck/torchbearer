import torch
from torchgeo.datasets import RasterDataset
import matplotlib.pyplot as plt

from collections.abc import Callable, Iterable, Sequence
from torchgeo.datasets.utils import Path
from typing import Any
from rasterio.crs import CRS
import os
import requests
import zipfile
from tqdm.auto import tqdm
import functools
import shutil
import pathlib

from sys import argv
from .label_dataset import LabelDataset
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import stack_samples, unbind_samples
from torch.utils.data import DataLoader
from ..config.products import PRODUCTS


class Sentinel2_60m(RasterDataset):
    """
    Initialize with a path or list of paths to a directory containing the
    individual band images.

    If products is a dictionary with product names and download links,
    downloading of those products will be attempted.
    """

    filename_glob = "**/T*_B01_60m.jp2"
    filename_regex = r"^.{6}_(?P<date>\d{8}T\d{6})_(?P<band>B[0-9A]{2})"
    date_format = '%Y%m%dT%H%M%S'
    is_image = True
    separate_files = True
    all_bands = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06',
                 'B07', 'B8A', 'B09', 'B11', 'B12')
    rgb_bands = ('B04', 'B03', 'B02')

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        products: dict = None,
    ):
        if products is not None:
            if type(paths) is Path:
                raise Exception(
                    'If products is defined, paths must be a string')

            ls = os.listdir(paths)
            for key in products.keys():
                if len([f for f in ls if key in f]) == 0:
                    self._download(paths, products[key], key)

        super().__init__(paths, crs, res, bands, transforms, cache)

    def _copernicus_authenticate(self):
        """
        Generate the token needed to download tiles from Copernicus
        """
        try:
            if self.access_token is not None:
                return
        except AttributeError:
            pass

        # Assume env is set with COPERNICUS_LOGIN and COPERNICUS_PASS
        login = os.getenv('COPERNICUS_LOGIN')
        if login is None:
            raise Exception('COPERNICUS_LOGIN environment variable not set')
        password = os.getenv('COPERNICUS_PASS')
        if password is None:
            raise Exception('COPERNICUS_PASS environment variable not set')

        session = requests.Session()
        data = [('client_id', 'cdse-public'), ('username', login),
                ('password', password), ('grant_type', 'password')]
        url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        resp = session.post(url, data=data, stream=True)
        content = resp.json()
        if resp.status_code != 200:
            resp.raise_for_status()
            raise Exception(f'Error authenticating with Copernicus: {
                            content['error']}: {content['error_description']}')

        self.access_token = content['access_token']

    def _download(self, path: Path, url: str, filename: str):
        cwd = os.getcwd()
        os.chdir(path)

        self._copernicus_authenticate()
        session = requests.Session()
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }
        session.headers.update(headers)

        resp = session.get(url, stream=True)

        if resp.status_code != 200:
            resp.raise_for_status()
            raise Exception(f'Download failed: {resp.text}')

        filesize = int(resp.headers.get('Content-Length', 0))
        path = pathlib.Path(f'{filename}.zip').expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        resp.raw.read = functools.partial(resp.raw.read)
        with tqdm.wrapattr(resp.raw, "read", total=filesize, desc=filename) as raw:
            with path.open('wb') as f:
                shutil.copyfileobj(raw, f)

        with zipfile.ZipFile(f'{filename}.zip', 'r') as zip_src:
            zip_src.extractall('.')

        os.chdir(cwd)

    def plot(self, sample, ax=None):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        image = sample['image'][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 10000, min=0, max=1).numpy()

        # Plot the image
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(image)

        if ax is None:
            return fig


# # How to use (from project root): $ python -m package.datasets.sentinel2_60m data masks
# if __name__ == "__main__":
#     img = Sentinel2_60m(argv[1], products=PRODUCTS)
#     mask = LabelDataset(argv[2])
#     ds = img & mask
#     g = torch.Generator().manual_seed(3)
#     sampler = RandomGeoSampler(ds, size=256, length=3, generator=g)
#     dataloader = DataLoader(ds, sampler=sampler, collate_fn=stack_samples)
#
#     for batch in dataloader:
#         sample = unbind_samples(batch)[0]
#         _, axes = plt.subplots(ncols=2)
#         imgfig = img.plot(sample, axes[0])
#         maskfig = mask.plot(sample, axes[1])
#         plt.axis('off')
#         plt.show()
