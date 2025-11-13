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
- Segmentation models
    - [x] UNet
        Validation loss seems to behave, check result with plot of output compared to image and mask
    - [ ] Fully convolutional network
    - [ ] Try torchgeo modules
- [x] Generic trainer:
    - Should be able to take in a lightning module and training hyperparameters
- [x] Training queue
    - Pass a list of settings and run training loops in sequence
- [ ] Add segmentation accuracy metric
- [ ] Add a profiler
- [ ] Generate predicted masks for entire tiles by collaging predictions

## Next
### Need
- Generic trainer
    - [x] Set job with a config file
- Training queue
    - [x] Set a number of training rounds to run in sequence
    - [ ] Properly log which versions correspond to which job
### Want
- [ ] Add a profiler
- [ ] Generate predicted masks for entire tiles by collaging predictions
- [ ] Revisit tiles with results of previous experiments to refine masks
