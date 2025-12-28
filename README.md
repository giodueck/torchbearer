# torchbearer
Scheduler/task manager for running Pytorch model training unattended

> [!NOTE]
> **About hardware dependencies**\
> This project was initially developed for use with CUDA capable GPUs (i.e. Nvidia), but as Pytorch supports AMD's ROCm,
> it is possible to run training and inference on AMD GPUs as well.
> The setup for Nvidia GPU use with Docker is simpler, as it is more established, and is what is assumed here.
> To set up the same for AMD, build and install the binaries for the [AMD container toolkit](https://github.com/ROCm/container-toolkit),
> then set it up as described in [the guide](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/quick-start-guide.html).

> [!NOTE]
> **About old hardware**\
> This project was developed on a GTX 1000 series GPU, right around when CUDA and later official driver support was
> dropped. Accordingly, some software versions were pinned to old versions.

## Dependencies
- `docker` and `docker-buildx`
- `nvidia-container-toolkit` for Nvidia GPU support in the container

## Building the container
```sh
docker build . -t torchbearer:0.2.0
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
    torchbearer:0.2.0 \
    python -m package.trainer configs/jobs.yaml
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
    torchbearer:0.2.0 \
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
    torchbearer:0.2.0 \
    python -m package.ensemble_inferrer configs/ensemble.yaml
```

This allows for using multiple models at once to stack their outputs.
