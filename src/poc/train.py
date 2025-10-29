import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from custom_lightning_module import CustomLightningModule
from custom_datamodule import CustomDataModule
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datamodule = CustomDataModule()

    model = CustomLightningModule(64*64, 2)

    logger = CSVLogger("data", name="logs")
    # early_stopping = EarlyStopping("train_loss")
    trainer = pl.Trainer(min_epochs=1, max_epochs=16,
                         log_every_n_steps=3, logger=logger)

    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule)

    test_dataloader = datamodule.test_dataloader()
    for batch in test_dataloader._get_iterator():
        print(batch[1])
    preds = trainer.predict(model, dataloaders=test_dataloader)
    print(preds[0])
