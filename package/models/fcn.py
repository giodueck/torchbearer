from torchgeo.models import FCN
import lightning.pytorch as pl
import torch
import torch.nn as nn


class FCNLightningModule(pl.LightningModule):
    def __init__(self, in_channels=11, num_filters=64, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = FCN(in_channels=in_channels,
                         classes=1, num_filters=num_filters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model.forward(x)

    def _common_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        outputs = self.forward(x)
        y = y.unsqueeze(1).float()
        loss = self.loss_fn(outputs, y)
        return loss, outputs, y

    def training_step(self, batch, batch_idx):
        loss, outputs, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        loss, outputs, y = self._common_step(batch, batch_idx)
        preds = (outputs > 0.5).float()
        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def createFCN(params: dict):
    model = FCNLightningModule(
        in_channels=params.get('in_channels', 11),
        num_filters=params.get('num_filters', 64),
        lr=params.get('lr', 1e-4),
        weight_decay=params.get('weight_decay', 1e-4),
    )
    return model
