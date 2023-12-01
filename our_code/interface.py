from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol, Tuple, Sequence
from torch import Tensor


# does the background augmentation.
class ShuffleBackgrounds(Protocol):
    def __call__(self, images: Tensor, bounding_boxes: Sequence[BoundingBox], labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns: image, foreground_label, background_label"""


# Output of the network's forward pass.
class FBClassification(NamedTuple):
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


# output of the dataset.
class LBBImage(NamedTuple):
    image: Tensor
    label: Tensor
    bounding_box: BoundingBox
