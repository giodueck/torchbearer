import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sys import argv
import yaml
import time

from . import datamodules
from . import models
from .config import configparser

if __name__ == "__main__":
    default_config = configparser.defaultConfig()[0]
    if len(argv) == 1:
        configs = configparser.defaultConfig()
    else:
        configs = configparser.parseConfig(argv[1])

    # Merge loaded configs into the default config
    # Last dict merged in gets priority
    for i, config in enumerate(configs):
        configs[i] = default_config | config
        configs[i]['trainer_params'] = default_config['trainer_params'] | configs[i]['trainer_params']

    global_start = time.perf_counter()

    for conf in configs:
        logger = TensorBoardLogger('data/logs')

        print(f'=> Starting experiment version: {logger.version}')
        print(f'==> Config:\n{yaml.dump(conf, None)}<==\n')
        start_time = time.perf_counter()

        try:
            pl.seed_everything(conf['seed'])

            datamodule = datamodules.datamodules[conf['datamodule']](
                conf['datamodule_params'])

            model = models.models[conf['model']](conf['model_params'])

            trainer_params = conf['trainer_params']

            early_stopping = EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=trainer_params['patience'],
            )
            model_checkpoint = ModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_top_k=trainer_params['save_top_k'],
            )

            trainer = pl.Trainer(
                min_epochs=trainer_params['min_epochs'],
                max_epochs=trainer_params['max_epochs'],
                log_every_n_steps=trainer_params['log_every_n_steps'],
                logger=logger,
                callbacks=[early_stopping, model_checkpoint]
            )

            trainer.fit(model, datamodule=datamodule)
            trainer.test(model, datamodule=datamodule)

            with open(f'data/logs/lightning_logs/version_{logger.version}/config.yaml', 'wt') as cfg:
                yaml.safe_dump(conf, cfg)

        except Exception as e:
            print(f'==> Exception when running experiment version {
                  logger.version}: {e}')

        duration = time.perf_counter() - start_time
        print(f'==> Experiment {logger.version} took {duration}s')

        # Clean up and free CUDA memory
        del trainer
        del model
        del datamodule.dataset
        del datamodule
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        pl.utilities.memory.garbage_collection_cuda()

    global_duration = time.perf_counter() - global_start
    print(f'=> Total runtime: {global_duration}s')
