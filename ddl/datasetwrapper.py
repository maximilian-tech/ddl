from mpi4py import MPI


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
