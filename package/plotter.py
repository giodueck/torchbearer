import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from sys import argv

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
    if len(argv) >= 5:
        configs = configparser.parseConfig(argv[4])
    else:
        configs = configparser.defaultConfig()

    # This is meant as a quick plotting/debugging script, so only do the first config
    conf = configs[0]

    pl.seed_everything(conf.get('seed', default_config['seed']))
    datamodule = datamodules.datamodules[conf.get(
        'datamodule', default_config['datamodule'])](conf.get('datamodule_params', {}))

    # Plot sample testing images
    datamodule.prepare_data()
    datamodule.setup('test')

    if do_predict:
        checkpoint = argv[1]
        hparams_file = argv[2]
        plot_count = int(argv[3])
        model = models.UNet.load_from_checkpoint(argv[1], hparams_file=hparams_file)

    for i, batch in enumerate(datamodule.test_dataloader()):
        if i == 0:
            batch_len = len(batch['image'])
        if do_predict:
            with torch.no_grad():
                image_tensor = (torch.FloatTensor(batch['image'])).to(
                    torch.device('cuda'))
                batch['output'] = model(image_tensor)
        # sample = unbind_samples(batch)[0]

        if i*batch_len >= plot_count:
            break
        for j, sample in enumerate(unbind_samples(batch)):
            if i*batch_len + j >= plot_count:
                break
            datamodule.plot(
                sample, filename=f'data/plots/plot_{i*batch_len+j}')
    exit(0)
