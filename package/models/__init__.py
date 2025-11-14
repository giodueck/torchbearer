from .unet import UNet, createUnet
from .fcn import FCNLightningModule, createFCN


models = {
    'unet': createUnet,
    'fcn': createFCN,
}

model_classes = {
    'unet': UNet,
    'fcn': FCNLightningModule,
}
