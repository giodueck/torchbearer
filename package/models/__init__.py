from .unet import UNet, createUnet
from .fcn import FCNLightningModule, createFCN
from .farseg import FarSegLightningModule, createFarSeg


models = {
    'unet': createUnet,
    'fcn': createFCN,
    'farseg': createFarSeg,
}

model_classes = {
    'unet': UNet,
    'fcn': FCNLightningModule,
    'farseg': FarSegLightningModule,
}
