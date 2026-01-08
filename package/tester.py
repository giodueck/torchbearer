import lightning.pytorch as pl
from sys import argv
import pathlib
import torch

from . import datamodules
from . import models
from .config import configparser


# Args:
# 1. path/to/checkpoint
# 2. path/to/version/logs which contains the hparams.yaml and config.yaml files
if __name__ == "__main__":
    # For cards with Tensor cores
    torch.set_float32_matmul_precision('medium')

    default_config = configparser.defaultConfig()[0]
    checkpoint = argv[1]
    path = pathlib.PosixPath(argv[2])
    hparams_file = path / "hparams.yaml"
    config_file = path / "config.yaml"
    conf = default_config | configparser.parseConfig(config_file)[0]
    model = models.model_classes[conf['model']].load_from_checkpoint(
        argv[1], hparams_file=hparams_file)

    pl.seed_everything(conf['seed'])
    datamodule = datamodules.datamodules[conf['datamodule']](conf['datamodule_params'])

    trainer = pl.Trainer()

    trainer.test(model, datamodule=datamodule)
