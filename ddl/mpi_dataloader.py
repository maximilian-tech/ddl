import dataclasses
import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
import functools
import os

import mpi4py.MPI
import torch

import numpy as np

from typing import List, Callable, Protocol, Any, cast, Iterable

from mpi4py import MPI
from mpi4py.util import dtlib
import math

# ToDo: Check what happens if epochs < workers , etc ...

# ToDo: add second window -> window can be filled while the other one is used
# Problem -> how to regain access while still filling the second window? Nonblocking?
# pusher[0]: use window[0]: fill window[1]
# pusher[1]: use window[0]: fill window[1]
# pusher[0]: use window[1]: fill window[0]
# pusher[1]: use window[1]: fill window[0]
# pusher[0]: use window[0]: fill window[1]
# ...


class CallbackProtocol(Protocol):
    def on_push_begin(self, **kwargs) -> Any:
        ...

    def global_shuffle(self, **kwargs) -> Any:
        ...

    def exec_function(self, **kwargs) -> Any:
        ...

    def on_push_end(self, **kwargs) -> Any:
        ...

    def on_shuffle_end(self, **kwargs) -> Any:
        ...


def execute_callbacks(position, callbacks: Iterable[CallbackProtocol], **kwargs) -> Any:
    for callback in callbacks:
        method: Callable = getattr(callback, position, lambda **args: None)
        if method is not None:
            logging.debug(
                f"[{MPI.COMM_WORLD.Get_rank():03d}] --> '{position}' "
                f"callback '{callback.__class__.__name__}' with {kwargs=}"
            )
            ret = method(**kwargs)
            logging.debug(
                f"[{MPI.COMM_WORLD.Get_rank():03d}] <-- '{position}'"
                f" callback '{callback.__class__.__name__}' with {kwargs=}"
            )
            return ret


def with_logging(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)

        try:
            logging.debug(f"[{MPI.COMM_WORLD.Get_rank():03d}] --> {func.__name__}: {signature}")
            result = func(*args, **kwargs)
            logging.debug(f"[{MPI.COMM_WORLD.Get_rank():03d}] <-- {func.__name__}: {result}")
        except Exception as e:
            logging.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e

        return result

    return wrapper


def for_all_methods(decorator: Callable[..., Any], exclude: str | list[str] | None = None):
    if exclude is None:
        exclude = []
    elif not isinstance(exclude, list):
        exclude = [exclude]

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr not in exclude:
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


class ProducerFunctionSkeleton:
    def __init__(self, *args, **kwargs):
        self.my_ary = None
        self.rank_global = None

    def on_init(self, *args, **kwargs):
        try:
            self.rank_global = kwargs["rank_global"]
        except KeyError:
            self.rank_global = MPI.COMM_WORLD.Get_rank()

    def post_init(self, *args, **kwargs):
        self.my_ary = kwargs["my_ary"]

    def execute_function(self, *args, **kwargs):
        raise NotImplementedError


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


def init_mpi_error_handling(comm_global, n_instances):
    rank_global = comm_global.Get_rank()
    size_global = comm_global.Get_size()

    if size_global < 2:
        if rank_global == 0:
            logging.warning("Using only a single rank!")
    #         logging.error("At least 2 MPI processes are required.")
    #     comm_global.Barrier()
    #     comm_global.Abort(2)

    print(f"{rank_global=} {size_global=} {n_instances=}")

    if size_global % n_instances != 0:
        if rank_global == 0:
            logging.error(f"Error! Number of MPI processes must be a multiple of {n_instances}")

        comm_global.Barrier()
        comm_global.Abort(1)


class DoesNotMatchError(Exception):
    def __init(self, value, message):
        self.value = value
        self.message = message
        super().__init__(message)


def init_mpi(n_instances: int = 1) -> MPI_Env:
    comm_global = MPI.COMM_WORLD
    comm_global.Set_name("comm_global")
    rank_global = comm_global.Get_rank()
    size_global = comm_global.Get_size()

    init_mpi_error_handling(comm_global, n_instances)

    # create sub communicator for shared memory / distribution
    # n_instances   = 3
    # rank          = 0 1 2 3 4 5 6 7 8 9 10 11
    # color_per_gpu = 0 0 0 0 1 1 1 1 2 2 2 2
    color_per_gpu = rank_global // (size_global // n_instances)

    logging.info("Creating sub communicator")
    logging.debug(f"{rank_global=} {size_global=} {n_instances=} {color_per_gpu=}")
    comm_per_gpu = cast(MPI.Intracomm, comm_global.Split(color=color_per_gpu, key=rank_global))
    comm_per_gpu.Set_name(f"comm_per_gpu_{color_per_gpu}")

    rank_per_gpu = comm_per_gpu.Get_rank()
    size_per_gpu = comm_per_gpu.Get_size()

    logging.debug(f"{rank_global}: {rank_per_gpu=} {size_per_gpu=}")

    # create Shared Memory (SHM) Communicators
    comm_per_gpu_shm = cast(
        MPI.Intracomm,
        comm_per_gpu.Split_type(
            split_type=MPI.WIN_FLAVOR_SHARED,
            key=rank_per_gpu,
            info=MPI.INFO_NULL,
            # split_type=MPI.WIN_FLAVOR_SHARED, key=rank_per_gpu, info=MPI.INFO_NULL
        ),
    )
    comm_per_gpu_shm.Set_name(f"comm_per_gpu_shm_{color_per_gpu}")
    rank_per_gpu_shm = comm_per_gpu_shm.Get_rank()
    size_per_gpu_shm = comm_per_gpu_shm.Get_size()

    logging.debug(f"{rank_global}: {rank_per_gpu_shm=} {size_per_gpu_shm=}")
    if size_per_gpu_shm != size_per_gpu:
        raise DoesNotMatchError((size_per_gpu_shm, size_per_gpu), f"{size_per_gpu_shm=} != {size_per_gpu=}")
    # this communicator connects the n-th datapushers to each other
    # n_instances      = 3
    # rank             = 0 1 2 3 4 5 6 7 8 9 10 11
    # color_nth_pusher = 0 1 2 0 1 2 0 1 2 0 1 2
    color_nth_pusher = rank_global % size_per_gpu

    comm_nth_pusher = cast(MPI.Intracomm, comm_global.Split(color=color_nth_pusher, key=rank_global))
    comm_nth_pusher.Set_name(f"comm_nth_pusher{color_nth_pusher}")

    logging.info("Creating sub communicator 1")
    logging.debug(f"{rank_global=:2d} {size_global=} {n_instances=}" f"{color_per_gpu=} {color_nth_pusher=}")

    comm_global.Barrier()

    env = MPI_Env(
        comm_global=comm_global,
        comm_per_gpu=comm_per_gpu,
        comm_per_gpu_shm=comm_per_gpu_shm,
        comm_nth_pusher=comm_nth_pusher,
        color=color_per_gpu,
        color_nth_pusher=color_nth_pusher,
        n_instances=n_instances,
    )
    return env


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


class GlobalShuffler:
    """
    This class takes care of the global shuffling of the data
    Only the "Pusher" ranks will use this class
    """

    def __init__(
        self,
        my_ary: np.ndarray | None,
        num_exchange: int,
        comm_nth_pusher: MPI.Intracomm,
        comm_global: MPI.Intracomm,
    ):
        self.my_ary = my_ary
        self.num_exchange = num_exchange

        self.comm_nth_pusher: MPI.Intracomm = comm_nth_pusher
        self.comm_global: MPI.Intracomm = comm_global

        self.rank: int = self.comm_global.Get_rank()

        self.n_instance: int = self.comm_nth_pusher.Get_size()

        seed = self.rank % (self.comm_global.Get_size() // self.n_instance)

        self.comm_shuffle_rng = np.random.default_rng(seed=seed)

    def _calculate_comm_partner(self) -> tuple[int, int]:  # type: ignore[return]
        """ "
        This function calculates the receiver and the sender of the data
        """
        row_idx = np.arange(self.comm_nth_pusher.Get_size(), dtype=np.int32)
        col_idx = np.arange(self.comm_nth_pusher.Get_size(), dtype=np.int32)

        # col
        #              | 0 1 0
        # row  | 0 0 1
        #      | 1 0 0

        if self.n_instance == 1:
            return 0, 0

        elif self.n_instance == 2:
            return (1, 1) if self.comm_nth_pusher.Get_rank() == 0 else (0, 0)

        cnt = 0
        reshuffle = True
        while reshuffle:
            self.comm_shuffle_rng.shuffle(col_idx)

            # bruteforce check, that the communication pattern is valid for every rank
            for r in range(0, len(row_idx)):
                send_to = col_idx[row_idx == r][0]
                recv_from = row_idx[col_idx == r][0]

                # prevents split graph    # prevent comm with self
                if (send_to == recv_from) or (row_idx == col_idx).any():
                    # invalid: try again if comm pattern is invalid on any rank
                    reshuffle = True
                    break
                else:
                    # valid
                    reshuffle = False

            if not reshuffle:
                send_to = col_idx[row_idx == self.comm_nth_pusher.Get_rank()][0]
                recv_from = row_idx[col_idx == self.comm_nth_pusher.Get_rank()][0]
                return send_to, recv_from

            cnt += 1
            if cnt > 1000:
                raise SystemExit(
                    f"[{MPI.COMM_WORLD.Get_rank():03d}] calculate_comm:"
                    f" Could not find a valid communication pattern after {cnt} tries "
                )


class GlobalShuffler_SendRecvReplace(GlobalShuffler):
    def __init__(
        self,
        my_ary: np.ndarray | None,
        num_exchange: int,
        comm_nth_pusher: MPI.Intracomm,
        comm_global: MPI.Intracomm,
    ):
        super().__init__(my_ary, num_exchange, comm_nth_pusher, comm_global)

    def global_shuffle(self):
        send_to, recv_from = self._calculate_comm_partner()

        self.comm_nth_pusher.Sendrecv_replace(
            self.my_ary[: self.num_exchange // 2],
            dest=send_to,
            sendtag=50,
            source=recv_from,
            recvtag=50,
        )
        self.comm_nth_pusher.Sendrecv_replace(
            self.my_ary[self.num_exchange // 2 : self.num_exchange],
            dest=recv_from,
            sendtag=51,
            source=send_to,
            recvtag=51,
        )


"""        
class GlobalShuffle_Bsend(GlobalShuffle):
    # this class takes care of the global shuffle
    # this is only done within the data pusher
    def __init__(self, exchange_method, ):
        super().__init__()
        self.info_dict = info_dict
        self.comm = comm
        self.num_exchange = num_exchange
        self.comm_shuffle_rng = comm_shuffle_rng
        self.my_ary = my_ary
        self.rank = rank
        pass

    def on_push_begin(self) -> None:
        datatype = MPI.FLOAT
        np_dtype = dtlib.to_numpy_dtype(datatype)
        self.bsend_buf = MPI.Alloc_mem(2 * MPI.BSEND_OVERHEAD
                                       + (self.num_exchange * self.info_dict['nFeatures']) * np_dtype.itemsize)

    def global_shuffle(self) -> None:
        send_to, recv_from = self._calculate_comm_partner2(self.comm_shuffle_rng)
        # send/recv data from other processes

        logging.info(f'[{self.rank:03d}]: will bsend ...')

        MPI.Attach_buffer(self.bsend_buf)

        self.comm.Bsend(self.my_ary[:self.num_exchange // 2], dest=send_to)
        self.comm.Bsend(self.my_ary[self.num_exchange // 2:self.num_exchange], dest=recv_from)

        req1 = self.comm.Irecv(self.my_ary[:self.num_exchange // 2], source=recv_from)
        req2 = self.comm.Irecv(self.my_ary[self.num_exchange // 2:self.num_exchange], source=send_to)
        MPI.Request.Waitall([req1, req2])

        logging.info(f'[{self.rank:03d}]: bsend done ...')

    def on_shuffle_end(self):
        # to be sure it could be used next time
        MPI.Detach_buffer()

    def on_push_end(self):
        MPI.Free_mem(self.bsend_buf)
"""


class Marker(Enum):
    END_OF_BATCH = auto()
    END_OF_EPOCH = auto()


class DistributedDataloaderABC(ABC):
    @abstractmethod
    def __len__(self):
        """
        :return: the number of data in the current access epoch (i.e. the number of batches)
        """
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        :param index: the index of the batch to be returned
        :return: the batch at the given index
        """
        pass

    @abstractmethod
    def _advance_to_next_producer(self):
        """
        Advance to the next target rank where to read data from

        """
        pass

    @abstractmethod
    def _finalize(self):
        """
        Finalize this dataloader and its worker. This is called when the dataloader is not needed anymore
        The data producer will be notified that it can stop producing data and can also shut down

        """
        pass

    @abstractmethod
    def _start_access_epoch(self, target_rank: int) -> None:
        """
        Acquires the access rights for the correct array if needed

        """
        pass

    @abstractmethod
    def _end_access_epoch(self, target_rank: int) -> None:
        """
        Releases the access rights

        """
        pass

    @abstractmethod
    def _can_continue(self):
        """
        :return: True if the dataloader can continue to produce data

        return self.epoch < self.n_epochs
        """

    @abstractmethod
    def mark(self, mark: Marker):
        """
        Mark the current event as finished. This method is called from the outside to indicate this.

        for epoch in range(n_epochs):
            for batch, data in enumerate(dataloader):
                # do something with data

                train(data)

                dataloader.mark(Marker.END_OF_BATCH)
            dataloader.mark(Marker.END_OF_EPOCH)

        """
        pass


@for_all_methods(with_logging, exclude="__getitem__")
class DistributedDataLoader(DistributedDataloaderABC):
    def __init__(
        self,
        producer_function: ProducerFunctionSkeleton,
        batch_size: int,
        connection: Connection,
        n_epochs: int,
        fraction_exchange: float,
        exchange_method: str,
        instance_idx: int,
        n_instances: int,
    ):
        self.epoch = 0
        self.batch = 0
        self.target_rank = 1

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.connection = connection

        if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
            # Send Metadata to the data producer
            # ToDo: define what metadata is needed
            self.metadata_to_producer = MetaData_Consumer_To_Producer(
                producer_function=producer_function,
                global_shuffle_fraction_exchange=fraction_exchange,
                global_shuffle_exchange_method=exchange_method,
                batch_size=batch_size,
            )
            self.connection.send_metadata(self.metadata_to_producer, "consumer")

            # Receive Metadata from the data producer
            self.metadata_from_producer: list[
                MetaData_Producer_To_Consumer
            ] = self.connection.recv_metadata_as_consumer()

            # ToDo:
            # 1. It depends on the "mode" is the batches per window are accumulated or not!
            # Do not mix up the two types of parallelism!!!!!
            self.splits: list[tuple[int, ...]] = [x.splits for x in self.metadata_from_producer]
            self.batches_per_window: list[int] = [x.batches_per_window for x in self.metadata_from_producer]

            mode = ["split_along_epoch", "do_not_split_along_epoch"]
            self.mode = mode[1]  # Only Mode 1 is logically correct
            self._len: int
            if self.mode == "do_not_split_along_epoch":
                self._len = self.batches_per_window[0]  # Total number of batches in an epoch
            elif self.mode == "split_along_epoch":
                self._len = sum(self.batches_per_window)  # Total number of batches in an epoch
            else:
                raise ValueError(f"Unknown mode {mode}")

            assert self._len > 0
            shape: list[tuple[int, ...]] = [x.shape for x in self.metadata_from_producer]

            self.arys: list[np.ndarray | None] = self.connection.init_windows(shape)

            self.connection.lock_windows()
            # Data producer will start producing data

            # Sync with Producers
            self.connection.Barrier()

            logging.debug(f"[{MPI.COMM_WORLD.Get_rank():03d}]: *** Barrier {self._len=} ***")

            self._start_access_epoch()
        else:
            self._len = 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if idx >= self._len:
            raise IndexError
        if idx < 0:
            raise ValueError

        logging.debug(
            f"[{MPI.COMM_WORLD.Get_rank():03d}] __getitem__({idx}) "
            f"{self.target_rank=}, {idx % self.batches_per_window[self.target_rank-1]=}"
        )
        # ToDo: What to do with the ugly 'self.target-1'?
        start: int = self.batch_size * (idx % self.batches_per_window[self.target_rank - 1])
        end: int = start + self.batch_size
        tmp: np.ndarray = cast(np.ndarray, self.arys[self.target_rank - 1])
        data_tensor = torch.from_numpy(tmp[start:end])

        splits: list[int] = list(self.splits[self.target_rank - 1])
        split_view = torch.split(data_tensor, splits, dim=1)

        return split_view

    def _start_access_epoch(self, target_rank: int = 0) -> None:
        if target_rank == 0:
            target_rank = self.target_rank
        self.connection.start_access_epoch(target_rank)

    def _end_access_epoch(self, target_rank: int = 0) -> None:
        if target_rank == 0:
            target_rank = self.target_rank
        self.connection.end_access_epoch(target_rank)

    def _can_continue(self) -> bool:
        return self.epoch < self.n_epochs

    def _advance_to_next_producer(self) -> None:
        self.target_rank += 1
        if self.target_rank > self.connection.n_producers:
            self.target_rank = 1
        if self.mode == "do_not_split_along_epoch":
            self._len = self.batches_per_window[self.target_rank - 1]

    def _on_batch_end(self) -> None:
        self.batch += 1

        # ToDo: this might lead to deadlock if batches_per_window is not the same for all producers ?
        if self.batch % self.batches_per_window[self.target_rank - 1] == 0:
            self._end_access_epoch()
            self._advance_to_next_producer()
            self._start_access_epoch()

    def _on_epoch_end(self) -> None:
        self.batch = 0
        self.epoch += 1
        if not self._can_continue():
            self.connection.shutdown_operation()
            self._finalize()

    def mark(self, mark: Marker) -> None:
        if mark == Marker.END_OF_BATCH:
            self._on_batch_end()

        elif mark == Marker.END_OF_EPOCH:
            self._on_epoch_end()

        else:
            raise ValueError(f"Unknown mark {mark}")

    def _finalize(self):
        self.connection.Barrier()
        self.connection.unlock_windows()
        self.connection.finalize(self.arys)


class DataPusherABC(ABC):
    @abstractmethod
    def push_data(self) -> None:
        pass

    @abstractmethod
    def _start_access_epoch(self, target_rank: int = -1) -> None:
        pass

    @abstractmethod
    def _end_access_epoch(self, target_rank: int = -1) -> None:
        pass

    @abstractmethod
    def sync(self) -> None:
        pass

    @abstractmethod
    def _finalize(self) -> None:
        pass


@dataclasses.dataclass
class DataProducerOnInitReturn:
    nData: int
    nValues: int
    shape: tuple[int, ...]
    splits: tuple[int, ...]


@for_all_methods(with_logging)
class DataPusher(DataPusherABC):
    def __init__(
        self,
        connection: Connection,
        comm_per_gpu_shm: MPI.Intracomm,
        comm_nth_pusher: MPI.Intracomm,
        comm_global: MPI.Intracomm,
    ) -> None:
        self.callbacks: list[CallbackProtocol] = []

        self.connection = connection
        self.comm_per_gpu_shm = comm_per_gpu_shm
        self.rank_per_gpu_shm: int = comm_per_gpu_shm.Get_rank()
        self.rank_global: int = comm_global.Get_rank()
        self.n_instance: int = comm_nth_pusher.Get_size()

        # Get Information from Consumer (which also has User defined information
        self.metadata_from_consumer: MetaData_Consumer_To_Producer = self.connection.recv_metadata_as_producer()

        self.callbacks.append(cast(CallbackProtocol, self.metadata_from_consumer.producer_function))
        # Execute the 'on_init' function to get information of the data (see DataProducerOnInit object)
        on_init_return: DataProducerOnInitReturn = execute_callbacks(
            "on_init", callbacks=self.callbacks, rank_global=self.rank_global
        )
        # Todo: how does this behave if different for different producer?
        batches_per_window: int = on_init_return.nData // self.metadata_from_consumer.batch_size
        assert batches_per_window > 0

        self.metadata_to_consumer = MetaData_Producer_To_Consumer(
            batches_per_window=batches_per_window,
            nData=on_init_return.nData,
            nValues=on_init_return.nValues,
            shape=on_init_return.shape,
            splits=on_init_return.splits,
        )
        # Send Information to the Consumer
        self.connection.send_metadata(self.metadata_to_consumer, "producer")

        self.arys = self.connection.init_windows(
            shapes=self.metadata_to_consumer.shape,
        )
        self.my_ary = self.arys[self.rank_per_gpu_shm - 1]  # rank -1 because rank 0 is the master and has no ary
        self.connection.lock_windows()

        num_exchange = int(on_init_return.nData * self.metadata_from_consumer.global_shuffle_fraction_exchange)
        # ToDo: When does the global shuffler shuffle data, when indices?
        # ToDo: how to indicate, that no global shuffle is wanted? What is affected?
        #  1. No split of data across nth_pusher dimensions
        #  2. ???
        # ToDo: Probably pass GlobalShuffeler object via the DistributedDataloader to here
        if (self.n_instance > 1) and (num_exchange > 0):
            if self.metadata_from_consumer.global_shuffle_exchange_method == "sendrecv_replace":
                global_shuffler = GlobalShuffler_SendRecvReplace(
                    self.my_ary,
                    num_exchange,
                    comm_nth_pusher,
                    comm_global,  # This is not implemented yet. This is for data exchange along the shuffle pipeline
                )
            else:
                raise NotImplementedError(
                    f"{self.metadata_from_consumer.global_shuffle_exchange_method}" f" is not implemented"
                )

            self.callbacks.append(cast(CallbackProtocol, global_shuffler))

        self.connection.sync(self.comm_per_gpu_shm.Get_rank())

        # fill the arrays with the data
        execute_callbacks(
            "post_init",
            callbacks=self.callbacks,
            rank_global=self.rank_global,
            my_ary=self.my_ary,
            n_instance=self.n_instance,
        )
        # sync the windows
        self.connection.sync(self.comm_per_gpu_shm.Get_rank())
        # sync with master
        logging.debug(f"[{self.rank_global:03d}]: *** Barrier")
        self.connection.Barrier()

    def sync(self) -> None:
        self.connection.sync(self.rank_per_gpu_shm)

    def _start_access_epoch(self, target_rank: int = 0) -> None:
        return None

    def _end_access_epoch(self, target_rank: int = 0) -> None:
        return None

    def _istart_access_epoch(self, target_rank: int = 0) -> WorkerInfo:
        return self.connection.Istart_access_epoch(target_rank)

    def _iend_access_epoch(self, target_rank: int = 0) -> WorkerInfo:
        return self.connection.Iend_access_epoch(target_rank)

    def _finalize(self) -> None:
        self.connection.Barrier()
        self.connection.unlock_windows()
        self.connection.finalize(self.arys)

    # def push_data(self, data: ArrayLike, target_rank: int) -> None:
    def push_data(self) -> None:
        execute_callbacks("on_push_begin", callbacks=self.callbacks)

        do_work = True
        while do_work:
            execute_callbacks("global_shuffle", callbacks=self.callbacks)
            self.sync()
            execute_callbacks("execute_function", callbacks=self.callbacks)

            info = self._iend_access_epoch()
            if info is WorkerInfo.STOP:
                break

            # MASTER READS WINDOW DATA HERE

            info = self._istart_access_epoch()
            if info is WorkerInfo.STOP:
                break

            execute_callbacks("on_shuffle_end", callbacks=self.callbacks)

        execute_callbacks("on_push_end", callbacks=self.callbacks)

        self._finalize()


def distributed_dataloader(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            _ = os.environ["SLURM_JOBID"]
            n_instances = int(os.environ["SLURM_GPUS_PER_NODE"]) * int(os.environ["SLURM_NNODES"])
        except KeyError:
            n_instances = 1  # default for local testing

        mpi_env: MPI_Env = init_mpi(n_instances=n_instances)
        if mpi4py.MPI.COMM_WORLD.Get_size() > 1:
            conn = Connection(mpi_env.comm_per_gpu_shm)
        else:
            conn = None

        if mpi_env.comm_per_gpu_shm.Get_rank() == 0:
            func(*args, mpi_env, conn)
        else:
            data_pusher = DataPusher(
                conn,
                mpi_env.comm_per_gpu_shm,
                mpi_env.comm_nth_pusher,
                mpi_env.comm_global,
            )
            data_pusher.push_data()
        print(f"[{mpi_env.comm_global.Get_rank():03d}]: finished")
        mpi_env.comm_global.Barrier()

    return wrapper
