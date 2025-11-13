import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from . import datamodules
from . import models

if __name__ == "__main__":
    pl.seed_everything(3)
    datamodule = datamodules.createSentinel2_60mDataModule()

    model = models.createUnet()

    logger = TensorBoardLogger('data/logs')
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, mode='min')
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')

    trainer = pl.Trainer(min_epochs=1, max_epochs=50, log_every_n_steps=8,
                         logger=logger, callbacks=[early_stopping, model_checkpoint])

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
