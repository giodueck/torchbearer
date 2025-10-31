import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class CustomLightningModule(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Initialize layers
        # self.fc1 = torch.nn.Linear(input_size, 8000)
        # self.fc2 = torch.nn.Linear(8000, 500)
        # self.fc3 = torch.nn.Linear(500, 500)
        # self.fc4 = torch.nn.Linear(500, num_classes)

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # halves height and width (64x64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = torch.nn.Dropout(0.25)

        self.fc1 = torch.nn.Linear(32*8*8, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

        class_counts = torch.tensor([0.60, 0.40])
        weights = 1.0 / class_counts  # inverse frequency weighting
        weights = weights / weights.sum()  # normalize if needed
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        x = x.to(torch.float32)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        x = self.pool(F.relu(self.conv1(x)))  # (1x64x64) => (8x32x32)
        x = self.pool(F.relu(self.conv2(x)))  # (8x32x32) => (16x16x16)
        x = self.pool(F.relu(self.conv3(x)))  # (16x16x16) => (32x8x8)
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
        loss, _, _ = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        # Flatten image to fit fully connected NN
        # x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)  # from [batch, H, W] to [batch, 1, H, W]
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def configure_optimizers(self):
        # If using a scheduler, also do the setup here
        return torch.optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        # return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0)

    # Many more possible steps or ends of steps, in which stuff like logging could be done

    def predict_step(self, batch):
        x, y = batch
        # x = x.reshape(x.size(0), -1)
        x = x.unsqueeze(1)  # from [batch, H, W] to [batch, 1, H, W]
        scores = self.forward(x)
        preds = torch.argmax(scores, 1)
        return preds
