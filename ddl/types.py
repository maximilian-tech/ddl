import dataclasses
from mpi4py import MPI
from enum import Enum, auto
from .datasetwrapper import ProducerFunctionSkeleton


@dataclasses.dataclass
class MetaData_Consumer_To_Producer:
    producer_function: ProducerFunctionSkeleton
    global_shuffle_fraction_exchange: float
    global_shuffle_exchange_method: str
    batch_size: int


@dataclasses.dataclass
class MetaData_Producer_To_Consumer:
    nData: int  # number of data points (e.g. images,
    nValues: int
    shape: tuple[int, ...]
    splits: tuple[int, ...]
    batches_per_window: int


@dataclasses.dataclass
class MPI_Env:
    comm_global: MPI.Intracomm
    comm_per_gpu: MPI.Intracomm
    comm_per_gpu_shm: MPI.Intracomm
    comm_nth_pusher: MPI.Intracomm
    color: int
    color_nth_pusher: int
    n_instances: int


class Marker(Enum):
    END_OF_BATCH = auto()
    END_OF_EPOCH = auto()
