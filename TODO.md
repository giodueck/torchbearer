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
        - [x] T20KPC
        - expand number of tiles
        - expand in times of year for the same tiles
    - [x] Revisit tiles with results of previous experiments to refine masks
- [x] Sentinel-2 datamodule
    - [x] Dataset configurations, e.g. number and selection of bands
- Segmentation models
    - [x] UNet
    - Try torchgeo modules
        - [x] Fully convolutional network
- [x] Generic trainer:
    Should be able to take in a lightning module and training hyperparameters
    - [x] Set job with a config file
    - [x] Output model config into logs directory
    - [ ] Output summary of all models trained and test performance
- [x] Training queue
    - Pass a list of settings and run training loops in sequence
- [ ] Add segmentation accuracy metric
- [ ] Add a profiler
- [x] Generate predicted masks for entire tiles by collaging predictions
- Citations:
    - U Net paper: https://arxiv.org/abs/1505.04597
    - Pytorch
    - Pytorch lightning
    - Torchgeo
    - FarSeg: https://arxiv.org/pdf/2011.09766

## Next
### Need
- [x] Generate predicted masks for entire tiles by collaging predictions
    - [ ] Generate using an ansemble of models
- [x] Run inference on tiles that were not used in training.
    Different products, configurable with a job config override
- Tiles:
    - [ ] expand in times of year for the same tiles

### Want
- [ ] Add a profiler
- [ ] Revisit tiles with results of previous experiments to refine masks
- [ ] Add segmentation accuracy metric
- Generic trainer
    - [ ] Output summary of all models trained and test performance
        Optional since this can also be viewed in TensorBoard
