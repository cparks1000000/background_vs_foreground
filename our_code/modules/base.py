from abc import abstractmethod
from typing import Generic, cast

from torch import nn

from our_code.typing import _T, _S, _TT


class OneModule(nn.Module, Generic[_T, _S]):
    def __call__(self, x: _T) -> _S: return cast(_S, super().__call__(x))

    @abstractmethod
    def forward(self, x: _T) -> _S: ...


class TwoModule(nn.Module, Generic[_T, _TT, _S]):
    def __call__(self, x: _T, y: _TT) -> _S: return cast(_S, super().__call__(x,y))

    @abstractmethod
    def forward(self, x: _T, y: _TT) -> _S: ...
