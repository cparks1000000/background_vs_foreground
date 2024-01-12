from typing import TypeVar, Callable, Protocol, NewType, NamedTuple

from torch import nn, Tensor
from torch.utils.data import DataLoader

from our_code.modules.base import OneModule

_T = TypeVar("_T", covariant=True)
_TT = TypeVar("_TT", covariant=True)
_S = TypeVar("_S", contravariant=True)

ModelType = TypeVar("ModelType", bound=tuple[nn.Module, ...])
DataType = TypeVar("DataType", bound=tuple)
LogType = TypeVar("LogType", bound=tuple)
ScoreType = TypeVar("ScoreType", bound=tuple)
OutputType = TypeVar("OutputType")

Device = NewType("Device", str)
DataFactory = Callable[[int], DataLoader[DataType]]


# noinspection PyPropertyDefinition
class LabeledImage(Protocol):
    @property
    def image(self) -> Tensor: ...

    @property
    def label(self) -> Tensor: ...


# noinspection PyPropertyDefinition
class ISimpleClassifierModel(Protocol[OutputType]):
    @property
    def classifier(self) -> OneModule[Tensor, OutputType]: ...


class SimpleClassifierModel(NamedTuple):
    classifier: nn.Module


