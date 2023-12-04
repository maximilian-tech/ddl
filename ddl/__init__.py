"""
ddl
-----------------
This is the distributed dataloader package.
"""

__all__ = [
    "ProducerFunctionSkeleton",
    "DataProducerOnInitReturn",
    "distributed_dataloader",
    "DistributedDataLoader",
    "Marker",
]

from .mpi_dataloader import (
    ProducerFunctionSkeleton,
    DistributedDataLoader,
    Marker,
)
from .ddl_env import distributed_dataloader
from .datapusher import DataProducerOnInitReturn
