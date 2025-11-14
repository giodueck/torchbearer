from torchgeo.models import FarSeg
import lightning.pytorch as pl
import torch
import torch.nn as nn


class FarSegLightningModule(pl.LightningModule):
    def __init__(self, backbone='resnet50', backbone_pretrained=True, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = FarSeg(backbone=backbone, classes=1,
                            backbone_pretrained=backbone_pretrained)
        self.lr = lr
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)


def createFarSeg(params: dict):
    model = FarSegLightningModule(
        backbone=params.get('backbone', 'resnet50'),
        backbone_pretrained=params.get('backbone_pretrained', True),
        lr=params.get('lr', 1e-4),
    )
    return model
