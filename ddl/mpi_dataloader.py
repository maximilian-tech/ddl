import logging
from abc import ABC, abstractmethod

import mpi4py.MPI
import torch

import numpy as np

from typing import cast

from mpi4py import MPI

from .types import Marker, MetaData_Consumer_To_Producer, MetaData_Producer_To_Consumer
from .datasetwrapper import ProducerFunctionSkeleton
from .connection import Connection
from .utils import with_logging, for_all_methods


# ToDo: Check what happens if epochs < workers , etc ...

# ToDo: add second window -> window can be filled while the other one is used
# Problem -> how to regain access while still filling the second window? Nonblocking?
# pusher[0]: use window[0]: fill window[1]
# pusher[1]: use window[0]: fill window[1]
# pusher[0]: use window[1]: fill window[0]
# pusher[1]: use window[1]: fill window[0]
# pusher[0]: use window[0]: fill window[1]
# ...


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
