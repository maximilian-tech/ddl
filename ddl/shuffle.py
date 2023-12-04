import numpy as np
from mpi4py import MPI


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


class SendRecvReplaceGlobalShuffler(GlobalShuffler):
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
class BsendGlobalShuffle(GlobalShuffle):
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
