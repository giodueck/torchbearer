# Changelog

## Unreleased

### Added

- Inferrer script that collages a large mask from model outputs
- Ensemble inferrer script that creates enferrence masks like the inferrer script, then combines them into a single mask

### Changed

- Products split into more categories to simplify testing with the inferrer script

### Fixed

- Sentinel_60m dataset now respects the products passed and only loads them instead of all found products

## [0.2.0] - 2025-11-14

### Added

- Configuration parser to run sequential training jobs
- Some logging for each trainer run
- Trainer now takes a job config as an argument
- Add Fully-Convolutional Network as a model option
- Add FarSeg as a model option
- The used job config is now output to the logs directory of a version

### Changed

- Moved DataModule and LightningModule creation out of the trainer script
- Simplify plotter arguments

### Fixed

- Plotter indexing fixed for some cases in which not all bands are used
- Free up memory after each training round to avoid CUDA out of memory errors

## [0.1.0] - 2025-11-09

### Added

- Sentinel2 dataset and dataloader
- Label dataset for masks
- UNet module
- Trainer for UNet module and Sentinel2 datamodule
- Plotter for showing results of predictions by checkpoint models

### Changed

- Moved masks from `package/poc/masks` to `masks`
- Changed mask format from JPG to PNG for lossless compression when converting to GeoTIFF

## [0.0.0]

### Added

- Basic Dockerfile and attempt to convert PyTorch tutorial into a Lightning example.
- Basic examples showing how to create a LightningModule and a LightningDataModule.
- Visualizer for large `jp2` (JPEG2000) files
- Small scripts to demonstrate creating cropped images in either grayscale or with multiple spectra
