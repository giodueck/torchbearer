# torchbearer
Scheduler/task manager for running Pytorch model training unattended

## Dependencies
- `docker` and `docker-buildx`
- `nvidia-container-toolkit` for Nvidia GPU support in the container

## Building the container
```sh
docker build . -t torchbearer:0.0.0
```

## Running the training
In the `PWD`:
- data/: contains source images
- masks/: contains ground truth images (masks or labels)
- package/: contains the Python code

Run the package by doing:
```sh
docker run -ti -v $PWD/data:/workdir/data -v $PWD/masks:/workdir/masks -v $PWD/package:/workdir/package --gpus all torchbearer:0.0.0 python -m package.trainer
```
