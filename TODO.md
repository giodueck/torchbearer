# TODO list

- [x] Proofs of concept
    - Partition images into datasets and binary label them
    - Train small classification models with lightning
        - Results: >80% correct predictions for datasets with 50-80% of images containing paleochannels
    - Test torchgeo datasets and datamodules
- [x] Sentinel-2 dataset
    - Custom dataset class since the copernicus browser downloads don't work for the current iteration of torchgeo's Sentinel2 dataset
    - [x] Local files support
    - [x] Download support
    - [x] Normalization and data augmentation
- [x] Sentinel-2 label dataset
    - [x] Convert PNG data to geotagged GeoTIFF
    - Tiles:
        - [x] T20KNA
        - [x] T20KNB
        - [ ] T20KPC
        - expand
    - [ ] Revisit tiles with results of previous experiments to refine masks
- [x] Sentinel-2 datamodule
    - [x] Dataset configurations, e.g. number and selection of bands
- Segmentation models
    - [x] UNet
    - Try torchgeo modules
        - [x] Fully convolutional network
- [x] Generic trainer:
    Should be able to take in a lightning module and training hyperparameters
    - [x] Set job with a config file
    - [ ] Output summary of all models trained and test performance
    - [ ] Output model config into logs directory
- [x] Training queue
    - Pass a list of settings and run training loops in sequence
- [ ] Add segmentation accuracy metric
- [ ] Add a profiler
- [ ] Generate predicted masks for entire tiles by collaging predictions
- Citations:
    - U Net paper: https://arxiv.org/abs/1505.04597
    - Pytorch
    - Pytorch lightning
    - Torchgeo
    - FarSeg: https://arxiv.org/pdf/2011.09766

## Next
### Need
- Generic trainer
    - [x] Set job with a config file
- Training queue
    - [x] Set a number of training rounds to run in sequence
    - [x] Properly log which versions correspond to which job
- Sentinel-2 datamodule
    - [x] Dataset configurations, e.g. number and selection of bands
### Want
- [ ] Add a profiler
- [ ] Generate predicted masks for entire tiles by collaging predictions
- [ ] Revisit tiles with results of previous experiments to refine masks
- [ ] Add segmentation accuracy metric
- Generic trainer
    - [ ] Output summary of all models trained and test performance
