import torch
import lightning.pytorch as pl
from torchgeo.datasets import unbind_samples
from .datamodules import Sentinel2_60mDataModule
from .models import UNet
from .config.products import PRODUCTS
from sys import argv


if __name__ == "__main__":
    if (len(argv) > 1):
        do_predict = True
    else:
        do_predict = False

    pl.seed_everything(3)
    datamodule = Sentinel2_60mDataModule(batch_size=3, patch_size=128, length=900,
                                         num_workers=6, sentinel_path='data',
                                         sentinel_products=PRODUCTS, mask_path='masks')

    # Plot sample testing images
    datamodule.prepare_data()
    datamodule.setup('test')

    if do_predict:
        checkpoint = argv[1]
        hparams_file = argv[2]
        plot_count = int(argv[3])
        model = UNet.load_from_checkpoint(argv[1], hparams_file=hparams_file)

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
