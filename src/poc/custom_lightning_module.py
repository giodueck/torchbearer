import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class CustomLightningModule(pl.LightningModule):
    def __init__(self, in_channels, img_height, img_width, num_classes):
        super().__init__()
        # Initialize layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # halves height and width
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = torch.nn.Dropout(0.2)

        self.fc1 = torch.nn.Linear(64*(img_height//(2*2*2))*(img_width//(2*2*2)), 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

        class_counts = torch.tensor([0.60, 0.40])
        weights = 1.0 / class_counts  # inverse frequency weighting
        weights = weights / weights.sum()  # normalize if needed
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))  # (11x128x128) => (16x64x64)
        x = self.pool(F.relu(self.conv2(x)))  # (16x64x64) => (32x32x32)
        x = self.pool(F.relu(self.conv3(x)))  # (32x32x32) => (64x16x16)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log('train_loss', loss)
        preds = torch.argmax(scores, 1)
        self.log('train_acc', (preds == y).sum().item() / y.size(0))

        # accuracy metrics can also be added here, like F1 score and similar
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log('val_loss', loss)
        preds = torch.argmax(scores, 1)
        self.log('val_acc', (preds == y).sum().item() / y.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)

        self.log('test_loss', loss)
        preds = torch.argmax(scores, 1)
        self.log('test_acc', (preds == y).sum().item() / y.size(0))
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        # Flatten image to fit fully connected NN
        # x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        # If using a scheduler, also do the setup here
        # return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.8)
        return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0)

    # Many more possible steps or ends of steps, in which stuff like logging could be done

    def predict_step(self, batch):
        x, y = batch
        # x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, 1)
        return preds
