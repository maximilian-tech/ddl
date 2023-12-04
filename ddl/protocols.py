from typing import Any, Protocol


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
