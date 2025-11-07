import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchgeo.datasets import stack_samples, unbind_samples

from .datamodules import Sentinel2_60mDataModule
from .config.products import PRODUCTS
from .models import UNet

if __name__ == "__main__":
    pl.seed_everything(3)
    datamodule = Sentinel2_60mDataModule(batch_size=6, patch_size=96, length=None,
                                         num_workers=4, sentinel_path='data',
                                         sentinel_products=PRODUCTS, mask_path='masks')

    # # Plot sample training images
    # datamodule.setup('fit')
    # print(len(datamodule.train_dataloader()))
    # for batch in datamodule.train_dataloader():
    #     print(len(unbind_samples(batch)))
    #     sample = unbind_samples(batch)[0]
    #     datamodule.plot(sample)
    # exit(0)

    model = UNet(in_channels=11, out_channels=1, lr=1e-4)

    logger = TensorBoardLogger('data/logs')
    early_stopping = EarlyStopping('val_loss', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint('val_loss', save_top_k=1, mode='min')

    trainer = pl.Trainer(min_epochs=1, max_epochs=10, log_every_n_steps=8,
                         logger=logger, callbacks=[early_stopping, model_checkpoint])

    trainer.fit(model, datamodule=datamodule)
