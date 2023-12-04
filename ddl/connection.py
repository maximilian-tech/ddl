import math
from enum import Enum, auto
from typing import List, cast
from mpi4py import MPI
import numpy as np
from mpi4py.util import dtlib

from .utils import for_all_methods, with_logging
from .types import MetaData_Producer_To_Consumer, MetaData_Consumer_To_Producer


class WorkerInfo(Enum):
    CONTINUE = auto()
    STOP = auto()


@for_all_methods(with_logging)
class Connection:
    """
    # this class takes care of the connection between
    # the data pusher(s) and the data consumer
    # it is responsible for the creation of the windows and the access to the data
    """

    # this care
    def __init__(self, comm_per_gpu_shm: MPI.Intracomm):
        self.wins: List[MPI.Win | None] = [None]
        self.comm_per_gpu_shm = comm_per_gpu_shm
        assert self.comm_per_gpu_shm.Get_size() > 1
        self.n_producers = self.comm_per_gpu_shm.Get_size() - 1

        self.shutdown_comm = self.comm_per_gpu_shm.Dup()
        self.shutdown_comm.Set_name(f"shutdown_comm_of_{self.comm_per_gpu_shm.Get_name()}")

        self.shutdown_request = MPI.Request()
        if self.shutdown_comm.Get_rank() > 0:
            self.shutdown_request = self.shutdown_comm.Ibarrier()

    def lock_windows(self) -> None:
        assert self.wins is not None
        for w in self.wins:
            if w is not None:
                w.Lock_all()

    def unlock_windows(self) -> None:
        assert self.wins is not None
        for w in self.wins:
            if w is not None:
                w.Unlock_all()

    def finalize(self, arys: List[np.ndarray | None]) -> None:
        assert len(self.wins) == len(arys)
        for i, (win, ary) in enumerate(zip(self.wins, arys)):
            if win is None or ary is None:
                continue
            win.Fence()
            win.Free()
            self.wins[i] = None
            arys[i] = None

    def sync(self, target) -> None:
        # -1 because the first rank has no window
        self.wins[target - 1].Sync()

    @with_logging
    def send_metadata(
        self,
        metadata: MetaData_Producer_To_Consumer | MetaData_Consumer_To_Producer,
        called_from: str,
    ) -> None:
        if called_from == "consumer":
            for t in range(1, self.comm_per_gpu_shm.Get_size()):
                self.comm_per_gpu_shm.ssend(metadata, dest=t, tag=0)
        elif called_from == "producer":
            self.comm_per_gpu_shm.ssend(metadata, dest=0, tag=0)
        else:
            raise ValueError(f"{called_from=} is not valid")

    def recv_metadata_as_producer(self) -> MetaData_Consumer_To_Producer:
        return self.comm_per_gpu_shm.recv(source=0, tag=0)

    def recv_metadata_as_consumer(self) -> list[MetaData_Producer_To_Consumer]:
        data = []
        for t in range(1, self.comm_per_gpu_shm.Get_size()):
            data.append(self.comm_per_gpu_shm.recv(source=t, tag=0))
        return data

    def init_windows(self, shapes: tuple[int, ...] | list[tuple[int, ...]]) -> list[np.ndarray | None]:
        # ToDo: Support only tensors of PyTorch and
        #       a) use pinned memory
        #       b) push them to the gpu already
        #       Problems: Typing of MPI + PyTorch

        shm_comm = self.comm_per_gpu_shm
        rank: int = shm_comm.Get_rank()
        comm_shm_size: int = shm_comm.Get_size()

        # consumer has already data from every producer as a list
        # producer has only own array. Mimic list to enable same code
        if not isinstance(shapes, list):
            shapes = [shapes] * (shm_comm.Get_size() - 1)

        # ToDo: Keep in mind compatibility to Tensors of PyTorch
        # get datatype from array with infodict?
        datatype = MPI.FLOAT
        np_dtype = dtlib.to_numpy_dtype(datatype)
        itemsize = datatype.Get_size()

        mpi_info = MPI.Info.Create()
        mpi_info.Set("alloc_shared_noncont", "true")

        wins: list[MPI.Win | None] = []
        arys: list[np.ndarray | None] = []

        for target in range(1, comm_shm_size):
            shape = shapes[target - 1]

            size = math.prod(shape) * itemsize if target == rank else 0

            win = MPI.Win.Allocate_shared(size=size, disp_unit=itemsize, comm=shm_comm, info=mpi_info)

            shm_comm.Barrier()
            win.Fence()

            if target == rank:
                buf = win.tomemory()
            else:
                buf, buf_size = win.Shared_query(rank=target)
                assert buf_size == itemsize

            ary: np.ndarray = np.ndarray(buffer=cast(memoryview, buf), dtype=np_dtype, shape=shape)

            win.Fence()

            wins.append(win)
            arys.append(ary)
        # Keep windows in class to ease access
        self.wins = wins
        return arys

    def Barrier(self):
        self.comm_per_gpu_shm.Barrier()

    def _sync(self, target_rank: int) -> None:
        if target_rank == 0:
            # if target_rank == 0, then sync my window, because rank 0 has no window
            if window := self.wins[self.comm_per_gpu_shm.Get_rank() - 1]:
                window.Sync()
        else:
            if window := self.wins[target_rank - 1]:
                window.Sync()

    def start_access_epoch(self, target_rank: int) -> None:
        self.comm_per_gpu_shm.Recv([None, 0, MPI.INT], source=target_rank, tag=7)
        self._sync(target_rank)

    def end_access_epoch(self, target_rank: int) -> None:
        self._sync(target_rank)
        self.comm_per_gpu_shm.Ssend([None, 0, MPI.INT], dest=target_rank, tag=7)

    def Istart_access_epoch(self, target_rank: int) -> WorkerInfo:
        req = self.comm_per_gpu_shm.Irecv([None, 0, MPI.INT], source=target_rank, tag=7)
        req_idx = MPI.Request.Waitany([req, self.shutdown_request])
        if req_idx == 0:
            info = WorkerInfo.CONTINUE
        else:
            req.Cancel()
            info = WorkerInfo.STOP

        self._sync(target_rank)
        return info

    def Iend_access_epoch(self, target_rank: int) -> WorkerInfo:
        self._sync(target_rank)
        req = self.comm_per_gpu_shm.Issend([None, 0, MPI.INT], dest=target_rank, tag=7)
        req_idx = MPI.Request.Waitany([req, self.shutdown_request])
        if req_idx == 0:
            info = WorkerInfo.CONTINUE
        else:
            req.Cancel()
            info = WorkerInfo.STOP
        return info

    def shutdown_operation(self) -> None:
        assert self.comm_per_gpu_shm.Get_rank() == 0
        self.shutdown_request = self.shutdown_comm.Ibarrier()
        self.shutdown_request.Wait()
