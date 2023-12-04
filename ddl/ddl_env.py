from mpi4py import MPI
import os
import logging
import functools
from typing import cast
from .types import MPI_Env
from .connection import Connection
from .datapusher import DataPusher
from .exceptions import DoesNotMatchError


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
            split_type=MPI.COMM_TYPE_SHARED,
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


def distributed_dataloader(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            _ = os.environ["SLURM_JOBID"]
            n_instances = int(os.environ["SLURM_GPUS_PER_NODE"]) * int(os.environ["SLURM_NNODES"])
        except KeyError:
            n_instances = 1  # default for local testing

        mpi_env: MPI_Env = init_mpi(n_instances=n_instances)
        if MPI.COMM_WORLD.Get_size() > 1:
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
