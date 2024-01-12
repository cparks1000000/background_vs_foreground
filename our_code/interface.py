from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, Tuple, Sequence, List
from torch import Tensor


# does the background augmentation.
class ShuffleBackgrounds(Protocol):
    def __call__(self, images: Tensor, bounding_boxes: Sequence[BoundingBox], labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns: image, foreground_label, background_label"""


# Output of the network's forward pass.
class MultiClassification(NamedTuple):
    classification: Tensor
    foreground: Tensor
    background: Tensor


# Holds bounding-box data.
class BoundingBox(NamedTuple):
    h_min: int
    h_max: int
    w_min: int
    w_max: int


def bounding_box_from_tuple(x: Tuple[int, int, int, int]): return BoundingBox(x[0], x[1], x[2], x[3])


# noinspection PyPropertyDefinition
class ILabeledBoundingBoxImage(Protocol):
    @property
    def image(self) -> Tensor: ...
    @property
    def label(self) -> Tensor: ...
    @property
    def bounding_box(self) -> List[BoundingBox]: ...


# output of the dataset.
class LabeledBoundingBoxImage(NamedTuple):
    image: Tensor
    label: Tensor
    bounding_box: List[BoundingBox]
