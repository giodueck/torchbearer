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
