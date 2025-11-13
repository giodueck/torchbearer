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
