# Best models so far
## Version 61

```yaml
- model: unet
  model_params:
    in_channels: 11
    features:
      - 64
      - 128
      - 256
      - 512
      - 1024
    lr: 0.0001
  datamodule: sentinel2_60m
  datamodule_params:
    length: 900
  seed: 3
  trainer_params:
    patience: 7
    save_top_k: 1
    max_epochs: 50
```

Test loss: 0.3550446033477783

## Version 74

```yaml
- model: unet
  model_params:
    in_channels: 11
    features:
      - 32
      - 64
      - 128
      - 256
      - 512
    lr: 0.0001
  datamodule: sentinel2_60m
  datamodule_params:
    length: 1200
    batch_size: 3
  seed: 42
  trainer_params:
    patience: 7
    save_top_k: 1
    max_epochs: 50
```

Test loss: 0.3845057189464569

## Version 88

```yaml
- model: farseg
  model_params:
    backbone: resnet18
    backbone_pretrained: false
  datamodule: sentinel2_60m
  datamodule_params:
    length: 1200
    batch_size: 3
    bands:
      - B04
      - B03
      - B02
  seed: 42
  trainer_params:
    patience: 7
    save_top_k: 1
    max_epochs: 50
```

Test loss: 0.3218611776828766

However, the results look blurrier and less precise than what the UNets produce

## Comparison between band usages
Compare versions 101 and 102

## Comparison between UNet feature sets
Compare versions 106, 107, 114

## Refined masks (paying more attention to fields with paleochannels)
### Version 115
This one only contains the updated version of the T20KNA mask.

```yaml
datamodule: sentinel2_60m
datamodule_params:
  batch_size: 3
  length: 900
model: unet
model_params:
  features:
  - 32
  - 64
  - 128
  - 256
  - 512
  in_channels: 11
  lr: 0.0001
seed: 42
trainer_params:
  log_every_n_steps: 8
  max_epochs: 50
  min_epochs: 1
  patience: 7
  save_top_k: 1
```

Test loss: 0.4122890532016754

### Version 116
This one contains slightly tweaked masks for T20KNB and T20KPC as well.

Same config as 115

Test loss: 0.6524485349655151

This looks bad, but the predictions are IMO better than 115.
