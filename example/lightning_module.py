import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class ExampleLightningModule(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Initialize layers
        self.fc1 = torch.nn.Linear(input_size, 50)
        self.fc2 = torch.nn.Linear(50, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        # accuracy metrics can also be added here, like F1 score and similar
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        # Flatten image to fit fully connected NN
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss  # , scores, y # in case we need them we can just return them too

    def configure_optimizers(self):
        # If using a scheduler, also do the setup here
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # Many more possible steps or ends of steps, in which stuff like logging could be done

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
