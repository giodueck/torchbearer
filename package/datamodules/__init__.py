from .sentinel2_20m_datamodule import Sentinel2_20mDataModule, createSentinel2_20mDataModule
from .sentinel2_60m_datamodule import Sentinel2_60mDataModule, createSentinel2_60mDataModule

datamodules = {
    'sentinel2_20m': createSentinel2_20mDataModule,
    'sentinel2_60m': createSentinel2_60mDataModule,
}
