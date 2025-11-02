import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from custom_lightning_module import CustomLightningModule
from custom_datamodule import CustomDataModule
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    datamodule = CustomDataModule(batch_size=4)

    model = CustomLightningModule(11, 128, 128, 2)

    logger = TensorBoardLogger("data/logs")
    early_stopping = EarlyStopping("val_loss", patience=5, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=32,
                         log_every_n_steps=8, logger=logger, callbacks=[early_stopping, model_checkpoint],
                         )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    preds = trainer.predict(model, dataloaders=test_dataloader)
