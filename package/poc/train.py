import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from custom_lightning_module import CustomLightningModule
from custom_datamodule import CustomDataModule
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == "__main__":
    # Polygons as masks are less learnable, I can't seem to get over 65% accuracy, while the hand-drawn one gets 82%
    datamodule = CustomDataModule(batch_size=2)

    model = CustomLightningModule(11, 192, 192, 2)

    logger = TensorBoardLogger("data/logs")
    early_stopping = EarlyStopping("val_loss", patience=7, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=50,
                         log_every_n_steps=8, logger=logger, callbacks=[early_stopping, model_checkpoint],
                         )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    preds = trainer.predict(model, dataloaders=test_dataloader)
