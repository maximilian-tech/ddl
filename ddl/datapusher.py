from abc import ABC, abstractmethod
from typing import cast
import dataclasses
from mpi4py import MPI
import logging

from .utils import for_all_methods, with_logging, execute_callbacks
from .connection import Connection, WorkerInfo
from .protocols import CallbackProtocol
from .types import MetaData_Consumer_To_Producer, MetaData_Producer_To_Consumer
from .shuffle import SendRecvReplaceGlobalShuffler


@dataclasses.dataclass
class DataProducerOnInitReturn:
    nData: int
    nValues: int
    shape: tuple[int, ...]
    splits: tuple[int, ...]


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
                global_shuffler = SendRecvReplaceGlobalShuffler(
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
