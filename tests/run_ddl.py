import mpi4py.MPI
import pytest
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset

import os
from socket import gethostname

from dataclasses import dataclass, asdict
import ddl

import numpy as np


class PointWiseData(object):
    def __init__(self, parameter_data, x_data, u_data, sample_weight=None):
        if sample_weight is not None:
            self.data_raw = np.hstack([parameter_data, x_data, u_data, sample_weight])
        else:
            self.data_raw = np.hstack([parameter_data, x_data, u_data])
        self.data = None
        self.sample_weight = None
        self.n_p = parameter_data.shape[-1]
        self.n_x = x_data.shape[-1]
        self.n_o = u_data.shape[-1]

    @property
    def parameter(self):
        return self.data[:, : self.n_p]

    @property
    def x(self):
        return self.data[:, self.n_p: self.n_p + self.n_x]

    @property
    def u(self):
        return self.data[:, self.n_p + self.n_x: self.n_p + self.n_x + self.n_o]

    @staticmethod
    def standard_normalize(raw_data, area_weighted=False):
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        if area_weighted:
            mean[-1] = 0.0
            std[-1] = np.mean(raw_data[:, -1])
            normalized_data = (raw_data - mean) / std
            return normalized_data[:, :-1], mean, std, normalized_data[:, -1]
        else:
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std

    @staticmethod
    def minmax_normalize(raw_data, n_para, n_x, n_target, area_weighted=False):
        mean = raw_data.mean(axis=0)
        std = raw_data.std(axis=0)
        for i in range(n_para + n_x):
            mean[i] = 0.5 * (np.min(raw_data[:, i]) + np.max(raw_data[:, i]))
            std[i] = 0.5 * (-np.min(raw_data[:, i]) + np.max(raw_data[:, i]))

        # also we normalize the output target to make sure the maximal is most 1
        for j in range(n_para + n_x, n_para + n_x + n_target):
            std[j] = np.max(np.abs(raw_data[:, j]))

        if area_weighted:
            # for area, simply take the mean as std for normalize
            mean[-1] = 0.0
            std[-1] = np.mean(raw_data[:, -1])
            normalized_data = (raw_data - mean) / std
            return normalized_data[:, :-1], mean, std, normalized_data[:, -1]
        else:
            normalized_data = (raw_data - mean) / std
            return normalized_data, mean, std


class DummyDataset(PointWiseData):
    def __init__(
            self, path: str, file: str, nTimesteps: int, idx: int, n_instances: int
    ):
        nData = nTimesteps * 10052

        start = (
                -(idx + 1) * nData // n_instances - 1
        )  # this cannot be 0, but has to be -1 because of inverted slicing
        end = -idx * nData // n_instances - 1
        # print(f"CombusterTurbineInterface {start=}, {end=}")
        data = np.random.rand(end - start, 10).astype("float32")

        print(
            f"[{mpi4py.MPI.COMM_WORLD.Get_rank():03d}]: {data.shape=} {start=}, {end=}"
        )

        #        print(f'{data.shape=}') # [10052*1000,10]
        parameter_data = data[:, [0]]
        x_data = data[:, [2, 3]]
        u_data = data[:, [4, 5, 6, 7, 8]]
        _sample_weight = (
            data[:, [-1]]
        )  # increase due to fp32 floatingpoint accuracy
        super().__init__(parameter_data, x_data, u_data, _sample_weight)
        self.data, self.mean, self.std, self.sample_weight = self.minmax_normalize(
            self.data_raw,
            n_para=self.n_p,
            n_x=self.n_x,
            n_target=self.n_o,
            area_weighted=True,
        )
        print("Done with init")


class Data_Producer(ddl.ProducerFunctionSkeleton):
    def __init__(self, cfg, data, idx: int, n_instances: int):
        super().__init__()
        self.rng = None
        self.cfg = cfg
        self.idx = idx
        self.n_instances = n_instances
        self.data = data
        self.producer = None
        self.training_data = None
        self.nData = None

    def on_init(self, *args, **kwargs):
        super().on_init(*args, **kwargs)

        self.rng = np.random.default_rng(seed=self.rank_global)

        print(f"PusherFunction: {self.rank_global}", flush=True)
        data = self.data(
            path=self.cfg.paths.data,
            file=self.cfg.files.dataset,
            idx=self.idx,  # which part of the data to load
            n_instances=self.n_instances,  # how many instances to load?
            nTimesteps=self.cfg.params.nData,
        )
        train_data = data.data
        sample_weight = data.sample_weight

        self.training_data = TensorDataset(
            torch.from_numpy(train_data[:, :3]),
            torch.from_numpy(train_data[:, 3:8]),
            torch.from_numpy(sample_weight).reshape(-1, 1),
        )

        nData = self.training_data.tensors[0].shape[0]
        splits: tuple[int, ...] = tuple(
            [x.shape[1] for x in self.training_data.tensors]
        )
        nValues = sum(splits)
        shape: tuple[int, ...] = (nData, nValues)

        self.nData = nData
        assert nData > 0
        assert nValues > 0

        return ddl.DataProducerOnInitReturn(nData, nValues, shape, splits)

    def post_init(self, *args, **kwargs):
        # fill the array
        assert self.nData is not None
        self.my_ary = kwargs["my_ary"]
        self.my_ary[...] = np.concatenate(
            (
                [
                    x.numpy().copy().reshape(self.nData, -1)
                    for x in self.training_data.tensors
                ]
            ),
            axis=-1,
        )

        self.training_data = None

    def execute_function(self, *args, **kwargs):
        assert self.my_ary is not None
        assert self.rng is not None
        # ToDo: Think about the distribution patterns... how is the data distributed?
        self.rng.shuffle(self.my_ary)


@ddl.distributed_dataloader
def main(cfg, mpi_env, conn):
    device, backend = (
        ("cpu", "gloo") if not torch.cuda.is_available() else ("cuda", "nccl")
    )

    try:
        gpus_per_node = int(os.environ["SLURM_GPUS_PER_NODE"])
    except KeyError:
        gpus_per_node = (
            mpi_env.n_instances
        )  # ToDo: not clean, only works for 1 node (e.g. local testing)
        # raise ValueError("SLURM_GPUS_PER_NODE not set (should be same as n_instances")

    if device == "cuda":
        pass
        #assert gpus_per_node == torch.cuda.device_count()

    try:
        _ = os.environ["MASTER_ADDR"]
        _ = os.environ["MASTER_PORT"]
    except KeyError:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    rank = mpi_env.comm_nth_pusher.Get_rank()
    world_size = mpi_env.comm_nth_pusher.Get_size()

    print(
        f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
        f" {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )
    if mpi_env.n_instances > 1:
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    # ToDo: autodetect local rank (create new shm of comm_nth_pusher?

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    print(f"Rank {rank} has local rank {local_rank} on {gethostname()}", flush=True)
    if device == "cuda":
        torch.device(f"{device}:{local_rank}")
        device = f"{device}:{local_rank}"
        print(f"USING GPU!", flush=True)

    producer_function = Data_Producer(
        cfg=cfg,
        data=DummyDataset,
        idx=rank,
        n_instances=world_size
    )

    train_dataloader = ddl.DistributedDataLoader(
        producer_function=producer_function,
        batch_size=cfg.params.batch_size,
        connection=conn,
        n_epochs=cfg.params.nepoch,
        fraction_exchange=0.5,
        instance_idx=rank,
        n_instances=world_size,
        exchange_method="sendrecv_replace",
    )

    for epoch in range(cfg.params.nepoch):
        print(f"Epoch {epoch + 1} of {cfg.params.nepoch}")

        logging.debug(
            f"[{mpi_env.comm_global.Get_rank():03d}]: {len(train_dataloader)=}"
        )
        for i, (pos, target, sample_weight) in enumerate(train_dataloader):
            # pos = pos.to(device)
            # target = target.to(device)
            # sample_weight = sample_weight.to(device)

            train_dataloader.mark(ddl.Marker.END_OF_BATCH)
        train_dataloader.mark(ddl.Marker.END_OF_EPOCH)

    print("Training finished")


@dataclass
class Paths:
    data: str
    log: str
    save_dir: str


@dataclass
class Files:
    dataset: str


@dataclass
class Params:
    nepoch: int
    lr: float
    batch_size: int
    checkpt_epoch: int
    display_epoch: int
    print_figure_epoch: int
    nData: int


@dataclass
class ShapeNet:
    use_resblock: bool
    use_multiscale: bool
    connectivity: str
    input_dim: int
    output_dim: int
    units: int
    nlayers: int
    weight_init_factor: float
    omega_0: int
    activation: str


@dataclass
class ParameterNet:
    use_resblock: bool
    use_multiscale: bool
    input_dim: int
    latent_dim: int
    units: int
    omega_0: int
    nlayers: int
    activation: str


@dataclass
class Config:
    paths: Paths
    files: Files
    params: Params
    shape_net: ShapeNet
    parameter_net: ParameterNet


def main_proxy():
    home = os.path.expanduser("~")
    cfg = Config(
        paths=Paths(
            data='unused-dummy', log="./runs", save_dir="../../saved_models/"
        ),
        files=Files(dataset="unused-dummy"),
        params=Params(
            nepoch=3,  # increase for more 'outer' iterations, increases total work
            lr=7e-4,
            batch_size=64 * 64,
            # Increase for larger batch size (needs more memory, longer kernels), decreases 'inner' iterations, does not change total work
            checkpt_epoch=2,
            display_epoch=100,
            print_figure_epoch=1,
            nData=10,  # Increase for more data, increases total work
        ),
        shape_net=ShapeNet(
            use_resblock=True,
            use_multiscale=True,
            connectivity="full",
            input_dim=2,
            output_dim=5,
            units=30,
            nlayers=3,
            weight_init_factor=0.01,
            omega_0=30,
            activation="SiLU",
        ),
        parameter_net=ParameterNet(
            use_resblock=True,
            use_multiscale=True,
            input_dim=1,
            latent_dim=9,
            units=30,
            omega_0=30,
            nlayers=3,
            activation="sine",
        ),
    )

    main(cfg)


if __name__ == "__main__":
    import logging

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    main_proxy()

