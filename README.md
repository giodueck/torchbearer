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
```sh
mkdir .cache
mkdir data
docker run -ti -v $PWD/.cache:/root/.cache -v $PWD/data:/root/data -v $PWD/src:/workdir/src --gpus all torchbearer:0.0.0 python src/demo/earth-water-surface-lightning.py
```
