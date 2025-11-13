import yaml
from sys import argv

"""
Config structure:
    [
        {
            'model': 'name',
            'model_params': {'p1':'v1', 'p2':'v2'}, # empty for default
            'datamodule': 'dm name',
            'datamodule_params': {}
            'seed': 42
            'trainer_params': { patience: 5, save_top_k: 1, max_epochs: 10 }
        }
    ]

Single yaml config example:

model: unet
model_params:
  in_channels: 11
datamodule: sentinel2_60m
datamodule_params: {}
seed: 42
trainer_params:
  patience: 5
  save_top_k: 1
  max_epochs: 10

Multiple yaml config example (repeat this block for multiple configs)

- model: unet
  model_params:
    in_channels: 11
  datamodule: sentinel2_60m
  datamodule_params: {}
  seed: 42
  trainer_params:
    patience: 5
    save_top_k: 1
    max_epochs: 10
"""


def defaultConfig():
    """
    Returns a list of one config with the first model and the first datamodule,
    with default configs.

    Use this as a reference of the available general and trainer parameters.
    For a list of model and datamodule parameters, the available parameters are
    the same as for the constructor of the respective class.
    """
    return [{
        'model': 'unet',
        'model_params': {},
        'datamodule': 'sentinel2_60m',
        'datamodule_params': {},
        'seed': 3,
        'trainer_params': {
            'patience': 5,
            'save_top_k': 1,
            'min_epochs': 1,
            'max_epochs': 10,
            'log_every_n_steps': 8,
        },
    }]


def parseConfig(filename):
    with open(filename) as stream:
        config = yaml.safe_load(stream)
    if type(config) is dict:
        return [config]
    return config


if __name__ == '__main__':
    # Test: call this script with module syntax and add the path to a yaml file as an argument
    # python -m package.config.configparser package/config/example.yaml
    print(defaultConfig())
    print(parseConfig(argv[1]))
