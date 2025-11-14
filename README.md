# torchbearer
Scheduler/task manager for running Pytorch model training unattended

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
