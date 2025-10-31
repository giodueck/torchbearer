import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from custom_lightning_module import CustomLightningModule
from custom_datamodule import CustomDataModule
from pytorch_lightning.loggers import CSVLogger

if __name__ == "__main__":
    datamodule = CustomDataModule(batch_size=32)

    model = CustomLightningModule(64*64, 2)

    logger = CSVLogger("data", name="logs")
    early_stopping = EarlyStopping("train_loss", patience=5)
    model_checkpoint = ModelCheckpoint(
        monitor="train_acc",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=32,
                         log_every_n_steps=3, logger=logger, callbacks=[early_stopping, model_checkpoint],
                         )

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    for batch in test_dataloader._get_iterator():
        print(batch[1])
    preds = trainer.predict(model, dataloaders=test_dataloader)
    for p in preds:
        print(p)
