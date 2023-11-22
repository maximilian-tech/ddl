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
    DataProducerOnInitReturn,
    distributed_dataloader,
    DistributedDataLoader,
    Marker,
)
