# Proof of Concept

- `jp2_vis.py`: visualize and annotate a .jp2 image
- `gen_subimages.py`: crop .jp2 image into many smaller ones
- `gen_multispectral_subimages.py`: combine several spectra and crop into subimages
- `gen_dataset.py`: save subimages as new .jp2 images, conserving their geotags
- `gen_labels_csv.py`: generate `labels.csv` with filename and True/False label for images based on how much of the mask
is painted. If a very small area of the image fits the label, it is true, otherwise it is false. This file can be used
in a custom dataset.

## TODO
- add truth layer, i.e. paint areas to label
- binary label generation from truth layer, e.g. if coverage > some percentage then yes else no
- train small classification model based on that generated data, from just one or a small number of images
    - [x] create dataset on hardcoded downloaded images
    - [ ] create dataset downloading images

## Notes
### Control
Predicting `False` for all test images (also known as most experiments before the first converging one):
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.5400000214576721     │
│         test_loss         │    0.7007813453674316     │
└───────────────────────────┴───────────────────────────┘
```

Any accuracy better than a coin flip would be progress.

### First converging run
#### Results
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.7599999904632568     │
│         test_loss         │    0.5119308233261108     │
└───────────────────────────┴───────────────────────────┘
```

#### Dataset
Downloaded and hand-masked based on a false-color image of bands B12, B8A, B04 with 60m resolution and a study of paleochannels around the Pilcomayo.

Dataset samples generated with a combined image of all bands available for 60m resolution (11 samples per pixel), of 128x128 size:
```bash
# Download link with product id: https://download.dataspace.copernicus.eu/odata/v1/Products(c9c0a608-1ed5-41ee-98bb-2346ff1502d2)/$value
# Generate combined image
magick /mnt/data/copernicus-data/S2B_MSIL2A_20251003T141709_N0511_R010_T20KNA_20251003T174955.SAFE/GRANULE/L2A_T20KNA_A044795_20251003T142621/IMG_DATA/R60m/T20KNA_20251003T141709_B{01,02,03,04,05,06,07,8A,09,11,12}_60m.jp2 -combine /mnt/data/copernicus-data/S2B_MSIL2A_20251003T141709_N0511_R010_T20KNA_20251003T174955.SAFE/GRANULE/L2A_T20KNA_A044795_20251003T142621/IMG_DATA/R60m/combined.jp2
# Generate 1000 chunks of 128x128
python src/poc/gen_dataset.py /mnt/data/copernicus-data/S2B_MSIL2A_20251003T141709_N0511_R010_T20KNA_20251003T174955.SAFE/GRANULE/L2A_T20KNA_A044795_20251003T142621/IMG_DATA/R60m/combined.tiff 128 128 1000 data image
# Label all images with at least 10% of coverage as having palleochannels
python src/poc/gen_labels_csv.py src/poc/masks/T20KNA.jpg 128 128 1000 data image
```

This produces 392 images tagged `True`, and the rest tagged `False`. This is compensated by punishing loss for the `False` more in the Cross Entropy Loss calculation.

#### Model

```python
    def __init__(self, in_channels, img_height, img_width, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)

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

    # --snip--

    def configure_optimizers(self):
        # If using a scheduler, also do the setup here
        # return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.8)
        return torch.optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0)

```

#### Trainer

```python
    datamodule = CustomDataModule(batch_size=4)

    model = CustomLightningModule(11, 128, 128, 2)

    logger = TensorBoardLogger("data/logs")
    early_stopping = EarlyStopping("val_loss", patience=5, mode="min")
    model_checkpoint = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=32,
                         log_every_n_steps=8, logger=logger, callbacks=[early_stopping, model_checkpoint],
                         )
```

#### Some tweaks
With a learning rate of 0.00005 instead of 0.0001 (halved), a patience of 7 instead of 5, and max_epochs of 50 instead of 32, a run that stopped at epoch=47 resulted in even better accuracy:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │     0.800000011920929     │
│         test_loss         │    0.4062427580356598     │
└───────────────────────────┴───────────────────────────┘
```

Additionally, a weight_decay of 0.01 produced a validation loss graph that consistently trailed the one before, but achieved better test results anyways:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8199999928474426     │
│         test_loss         │    0.37320077419281006    │
└───────────────────────────┴───────────────────────────┘
```

### Combined datasets
Combining 1000 images from each of the tiles T20KNA and T20KNB, the accuracy stayed at >80%, even when the proportion of images containing a paleochannel dropped to around 29% of the dataset:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8349999785423279     │
│         test_loss         │    0.3819182515144348     │
└───────────────────────────┴───────────────────────────┘
```

The loss weights had to be adjusted, in this run they were `[0.75, 0.25]`.
