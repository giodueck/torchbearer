import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .datamodules import Sentinel2_60mDataModule
from .config.products import PRODUCTS
from .models import UNet

if __name__ == "__main__":
    pl.seed_everything(3)
    datamodule = Sentinel2_60mDataModule(batch_size=3, patch_size=128, length=900,
                                         num_workers=6, sentinel_path='data',
                                         sentinel_products=PRODUCTS, mask_path='masks')

    model = UNet(in_channels=11, out_channels=1, lr=1e-4)

    logger = TensorBoardLogger('data/logs')
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='min')
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    trainer = pl.Trainer(min_epochs=1, max_epochs=50, log_every_n_steps=8,
                         logger=logger, callbacks=[early_stopping, model_checkpoint])

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
