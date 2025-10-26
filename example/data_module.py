import pytorch_lightning as pl


class ExampleDataModule(pl.LightningDataModule):
    def __init__(self):
        pass

    def prepare_data(self):
        """
        Data setup, done once for all GPUs.
        Can be downloading data or instead call a custom dataset from setup, skipping this step.
        """
        pass

    def setup(self, stage: str = None):
        """
        Data setup, done once per GPU.
        """
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
