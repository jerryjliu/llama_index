from typing import Any, TypeVar, Union

from .events import StopEvent

StopEventT = TypeVar("StopEventT", bound=StopEvent)
# TODO: When releasing 1.0, remove support for Any
# and enforce usage of StopEventT
RunResultT = Union[StopEventT, Any]
