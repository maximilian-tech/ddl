from typing import Callable, Any, Iterable
import functools
import logging
from mpi4py import MPI

from .protocols import CallbackProtocol


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
