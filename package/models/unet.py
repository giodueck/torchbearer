import torch
import torch.nn as nn
import lightning.pytorch as pl


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

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


def createUnet(params: dict):
    """
    Returns a configured UNet.

    If params contains a parameter for UNet, it is used, otherwise it is set to
    a default value. Inexistant parameters are ignored.
    """
    model = UNet(
        in_channels=params.get('in_channels', 11),
        out_channels=params.get('out_channels', 1),
        features=params.get('features', [64, 128, 256, 512]),
        lr=params.get('lr', 1e-4),
    )
    return model
