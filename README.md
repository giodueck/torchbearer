# torchbearer
Scheduler for running Pytorch model training on Satellite images unattended.

This project makes use of [Pytorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) and [TorchGeo](https://github.com/torchgeo/torchgeo) and allows for using simple configuration files to queue up model configurations to train with one command. It supports using different models and dataloaders in a modular fashion, automatically downloading the needed images from [Copernicus](https://browser.dataspace.copernicus.eu).

## Features

- Modular design using Lightning's Trainer, LightningDataModule and LightningModule
- Several CNN model architectures:
    - FCN: a simple 5-layer fully-connected network
    - FarSeg: foreground-aware relation network, designed for geospatial object segmentation
    - U-Net: simple-yet-powerful well known architecture
- Several Data Modules:
    - 60m resolution provides 11 bands
    - 20m resolution provides 10 bands
- Automatic image data downloads from Copernicus (requires an account)
- Automatic loading of georeferenced training objective masks
    - Includes a series of masks created for detection of [Palaeochannels](https://en.wikipedia.org/wiki/Palaeochannel) in the Chaco region of South America
- Training of models on satellite images
- Inference on satellite images
    - Ensemble inference allows for a set of models to combine their outputs into one result image

## Motive

This project was developed as part of my undergraduate thesis (in spanish): "Uso de Redes Neuronales Convolucionales para Interpretación de Imágenes Satelitales"; or Use of Convolutional Neural Networks for Interpretation of Satellite Images.

## Disclaimers

> [!NOTE]
> **About hardware dependencies**\
> This project was initially developed for use with CUDA capable GPUs (i.e. Nvidia), but as Pytorch supports AMD's ROCm,
> it is possible to run training and inference on AMD GPUs as well. For this, a `Dockerfile-rocm` is provided.
> The setup for Nvidia GPU use with Docker is simpler, as it is more established, and is what is assumed here.
> To set up the same for AMD, build and install the binaries for the [AMD container toolkit](https://github.com/ROCm/container-toolkit),
> then set it up as described in [the guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html).\
> This setup was not thoroughly tested, but a local installation of the needed libraries and Python requirements works, replacing the lengthy docker command with an equivalent local command.\
> As an example, installing the python requirements in a virtual environment: `sudo su -c "source venv/bin/activate; export $(cat env | xargs) && python -m package.trainer configs/jobs.yaml"`

> [!NOTE]
> **About old hardware**\
> This project was developed on a GTX 1000 series GPU, right around when CUDA and later official driver support was
> dropped. Accordingly, some software versions were pinned to old versions in the `Dockerfile`.

## Dependencies
- `docker` and `docker-buildx`
- `nvidia-container-toolkit` for Nvidia GPU support in the container

## Building the container
```sh
docker build . -t torchbearer:1.0.0
```

## Running the training
In the `PWD`:
- data/: contains source images
- masks/: contains ground truth images (masks or labels)
- package/: contains the Python code
- configs/: job parameter configuration files
- env: environment variables for Sentinel2 product download credentials

Run the package by doing:
```sh
docker run -ti \
    --env-file=env \
    -v $PWD/data:/workdir/data \
    -v $PWD/masks:/workdir/masks \
    -v $PWD/package:/workdir/package \
    -v $PWD/configs:/workdir/configs \
    --gpus all \
    torchbearer:1.0.0 \
    python -m package.trainer configs/jobs.yaml
```

Or locally with
```sh
export $(cat env | xargs) && python -m package.trainer configs/jobs.yaml
```

## Plotting results

```sh
docker run -ti \
    --env-file=env \
    -v $PWD/data:/workdir/data \
    -v $PWD/masks:/workdir/masks \
    -v $PWD/package:/workdir/package \
    -v $PWD/configs:/workdir/configs \
    --gpus all \
    torchbearer:1.0.0 \
    python -m package.plotter path/to/version/checkpoint.ckpt path/to/version <number of plots to generate>
```

This command will generate plots under `data/plots` with filenames like `plot_{i}.png`.

## Creating prediction tiles
To create a grayscale prediction map ready to import into a GIS program, configure the model checkpoints and prediction
parameters in `configs/ensemble.yaml` and run the following command:

```sh
docker run -ti \
    --env-file=env \
    -v $PWD/data:/workdir/data \
    -v $PWD/masks:/workdir/masks \
    -v $PWD/package:/workdir/package \
    -v $PWD/configs:/workdir/configs \
    --gpus all \
    torchbearer:1.0.0 \
    python -m package.ensemble_inferrer configs/ensemble.yaml
```

Or locally with
```sh
export $(cat env | xargs) && python -m package.ensemble_inferrer configs/ensemble.yaml
```

This allows for using multiple models at once to stack their outputs.
