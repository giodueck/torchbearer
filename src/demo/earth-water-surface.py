from collections.abc import Callable, Iterable
from pathlib import Path
import os

import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader

from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, utils
from torchgeo.samplers import RandomGeoSampler, Units
from torchgeo.transforms import indices


model_filename = 'model.pth'


# Download and extract dataset to a temp folder
tmp_path = Path(os.path.curdir) / 'surface_water/'
utils.download_and_extract_archive(
    'https://hf.co/datasets/cordmaur/earth_surface_water/resolve/main/earth_surface_water.zip',
    tmp_path,
)

# Set the root to the extracted folder
root = tmp_path / 'dset-s2'


def scale(item: dict):
    item['image'] = item['image'] / 10000
    return item

train_imgs = RasterDataset(
    paths=(root / 'tra_scene').as_posix(), crs='epsg:3395', res=10, transforms=scale
)
train_msks = RasterDataset(
    paths=(root / 'tra_truth').as_posix(), crs='epsg:3395', res=10
)

valid_imgs = RasterDataset(
    paths=(root / 'val_scene').as_posix(), crs='epsg:3395', res=10, transforms=scale
)
valid_msks = RasterDataset(
    paths=(root / 'val_truth').as_posix(), crs='epsg:3395', res=10
)

# IMPORTANT
train_msks.is_image = False
valid_msks.is_image = False

train_dset = train_imgs & train_msks
valid_dset = valid_imgs & valid_msks

# Create the samplers

train_sampler = RandomGeoSampler(train_imgs, size=512, length=130, units=Units.PIXELS)
valid_sampler = RandomGeoSampler(valid_imgs, size=512, length=64, units=Units.PIXELS)


# Adjust the batch size according to your GPU memory
train_dataloader = DataLoader(
    train_dset, sampler=train_sampler, batch_size=2, collate_fn=stack_samples
)
valid_dataloader = DataLoader(
    valid_dset, sampler=valid_sampler, batch_size=2, collate_fn=stack_samples
)

train_batch = next(iter(train_dataloader))
valid_batch = next(iter(valid_dataloader))


# Visualization
def plot_imgs(
    images: Iterable, axs: Iterable, chnls: list[int] = [2, 1, 0], bright: float = 3.0
):
    for img, ax in zip(images, axs):
        arr = torch.clamp(bright * img, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis('off')


def plot_msks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.squeeze().numpy(), cmap='Blues')
        ax.axis('off')


def plot_batch(
    batch: dict,
    bright: float = 3.0,
    cols: int = 4,
    width: int = 5,
    chnls: list[int] = [2, 1, 0],
):
    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())

    # if batch contains images and masks, the number of images will be doubled
    n = 2 * len(samples) if ('image' in batch) and ('mask' in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ('image' in batch) and ('mask' in batch):
        # plot the images on the even axis
        plot_imgs(
            images=map(lambda x: x['image'], samples),
            axs=axs.reshape(-1)[::2],
            chnls=chnls,
            bright=bright,
        )

        # plot the masks on the odd axis
        plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1)[1::2])

    else:
        if 'image' in batch:
            plot_imgs(
                images=map(lambda x: x['image'], samples),
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )

        elif 'mask' in batch:
            plot_msks(masks=map(lambda x: x['mask'], samples), axs=axs.reshape(-1))

plot_batch(train_batch)
plt.show()


# Data Standardization and Spectral Indices
def calc_statistics(dset: RasterDataset):
    """
    Calculate the statistics (mean and std) for the entire dataset
    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.
    For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index
    files = [
        item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
    ]

    # Resetting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rio.open(file).read() / 10000  # type: ignore
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with length n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)

# Calculate the statistics (Mean and std) for the dataset
mean, std = calc_statistics(train_imgs)

# Please, note that we will create spectral indices using the raw (non-normalized) data. Then, when normalizing, the sensors will have more channels (the indices) that should not be normalized.
# To solve this, we will add the indices to the 0's to the mean vector and 1's to the std vectors
mean = np.concat([mean, [0, 0, 0]])
std = np.concat([std, [1, 1, 1]])

norm = K.Normalize(mean=mean, std=std)

tfms = torch.nn.Sequential(
    indices.AppendNDWI(index_green=1, index_nir=3),
    indices.AppendNDWI(index_green=1, index_nir=5),
    indices.AppendNDVI(index_nir=3, index_red=2),
    norm,
)

# We add 3 layers and now have 9 instead of 6
transformed_img = tfms(train_batch['image'])
# print(transformed_img.shape)


# Segmentation Model
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(weights=None, num_classes=2)
# print(model)

# Since the model is prepared for RGB images, it expects 3 channels in the first convolution. We modify this layer to accept our 9 layers instead.
# Before: (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
backbone = model.get_submodule('backbone')

conv = torch.nn.modules.conv.Conv2d(
    in_channels=9,
    out_channels=64,
    kernel_size=(7, 7),
    stride=(2, 2),
    padding=(3, 3),
    bias=False,
)
backbone.register_module('conv1', conv)


pred = model(torch.randn(3, 9, 512, 512))
pred['out'].shape


# Training Loop

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

def train_loop(
    epochs: int,
    train_dl: DataLoader,
    val_dl: DataLoader | None,
    model: torch.nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    acc_fns: list | None = None,
    batch_tfms: Callable | None = None,
):
    # size = len(dataloader.dataset)
    cuda_model = model.to(device)

    for epoch in range(epochs):
        accum_loss = 0
        for batch in train_dl:
            if batch_tfms is not None:
                X = batch_tfms(batch['image']).to(device)
            else:
                X = batch['image'].to(device)

            y = batch['mask'].type(torch.long).to(device)
            pred = cuda_model(X)['out']
            loss = loss_fn(pred, y)

            # BackProp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the accum loss
            accum_loss += float(loss) / len(train_dl)

        # Testing against the validation dataset
        if acc_fns is not None and val_dl is not None:
            # reset the accuracies metrics
            acc = [0.0] * len(acc_fns)

            with torch.no_grad():
                for batch in val_dl:
                    if batch_tfms is not None:
                        X = batch_tfms(batch['image']).to(device)
                    else:
                        X = batch['image'].type(torch.float32).to(device)

                    y = batch['mask'].type(torch.long).to(device)

                    pred = cuda_model(X)['out']

                    for i, acc_fn in enumerate(acc_fns):
                        acc[i] = float(acc[i] + acc_fn(pred, y) / len(val_dl))

            # at the end of the epoch, print the errors, etc.
            print(
                f'Epoch {epoch}: Train Loss={accum_loss:.5f} - Accs={[round(a, 3) for a in acc]}'
            )
        else:
            print(f'Epoch {epoch}: Train Loss={accum_loss:.5f}')


# Loss and Accuracy Functions

def oa(pred, y):
    flat_y = y.squeeze()
    flat_pred = pred.argmax(dim=1)
    acc = torch.count_nonzero(flat_y == flat_pred) / torch.numel(flat_y)
    return acc


def iou(pred, y):
    flat_y = y.cpu().numpy().squeeze()
    flat_pred = pred.argmax(dim=1).detach().cpu().numpy()
    return jaccard_score(flat_y.reshape(-1), flat_pred.reshape(-1), zero_division=1.0)


def loss(p, t):
    return torch.nn.functional.cross_entropy(p, t.squeeze())


# Training

# adjust number of epochs depending on the device
if torch.cuda.is_available():
    num_epochs = 8
else:
    # if GPU is not available, just make 1 pass and limit the size of the datasets
    num_epochs = 2

    # by limiting the length of the sampler we limit the iterations in each epoch
    train_dataloader.sampler.length = 8
    valid_dataloader.sampler.length = 8

# train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
train_loop(
    num_epochs,
    train_dataloader,
    valid_dataloader,
    model,
    loss,
    optimizer,
    acc_fns=[oa, iou],
    batch_tfms=tfms,
)

torch.save(model.state_dict(), model_filename)
