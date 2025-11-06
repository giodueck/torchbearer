# Taken from https://torchgeo.readthedocs.io/en/stable/tutorials/trainers.html
# Short demo of pytorch lightning with torchgeo. This runs flawlessly in the docker container built with
# the Dockerfile in the root directory.
#
# Notice how this is so much shorter and more modularized than unstructured pytorch, e.g.
# https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html

import os
import tempfile

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask

## Lightning modules

batch_size = 10
num_workers = 4
max_epochs = 16
fast_dev_run = False

root = os.path.join(tempfile.gettempdir(), 'eurosat100')
datamodule = EuroSAT100DataModule(
    root=root, batch_size=batch_size, num_workers=num_workers, download=True
)

task = ClassificationTask(
    loss='ce',
    model='resnet18',
    weights=ResNet18_Weights.SENTINEL2_ALL_MOCO,
    in_channels=13,
    num_classes=10,
    lr=0.1,
    patience=5,
)

## Training

default_root_dir = os.path.join(tempfile.gettempdir(), 'experiments')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10)
logger = TensorBoardLogger(save_dir=default_root_dir, name='tutorial_logs')

trainer = Trainer(
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=1,
    max_epochs=max_epochs,
)

trainer.fit(model=task, datamodule=datamodule)

trainer.test(model=task, datamodule=datamodule)
