# Changelog

## [0.2.0] - 2025-11-12

### Changed

- Moved DataModule and LightningModule creation out of the trainer script

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
