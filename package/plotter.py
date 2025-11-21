import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from sys import argv
import pathlib

from . import datamodules
from . import models
from .config import configparser


if __name__ == "__main__":
    if (len(argv) > 1):
        do_predict = True
    else:
        do_predict = False

    # Also only relevant when do_predict == True
    default_config = configparser.defaultConfig()[0]

    if do_predict:
        checkpoint = argv[1]
        path = pathlib.PosixPath(argv[2])
        hparams_file = path / "hparams.yaml"
        config_file = path / "config.yaml"
        conf = default_config | configparser.parseConfig(config_file)[0]
        plot_count = int(argv[3])
        model = models.model_classes[conf['model']].load_from_checkpoint(
            argv[1], hparams_file=hparams_file)

        # For overrides of e.g. the dataset to plot
        # Should be compatible with original model, of course
        if len(argv) > 4:
            conf |= configparser.parseConfig(argv[4])[0]
    else:
        conf = default_config

    pl.seed_everything(conf['seed'])
    datamodule = datamodules.datamodules[conf['datamodule']](conf['datamodule_params'])

    # Plot sample testing images
    datamodule.prepare_data()
    datamodule.setup('test')

    for i, batch in enumerate(datamodule.test_dataloader()):
        if i == 0:
            batch_len = len(batch['image'])
        if do_predict:
            with torch.no_grad():
                image_tensor = (torch.FloatTensor(batch['image'])).to(
                    torch.device('cuda'))
                batch['output'] = model(image_tensor)

        if i*batch_len >= plot_count:
            break
        for j, sample in enumerate(unbind_samples(batch)):
            if i*batch_len + j >= plot_count:
                break
            datamodule.plot(
                sample, filename=f'data/plots/plot_{i*batch_len+j}')
    exit(0)
