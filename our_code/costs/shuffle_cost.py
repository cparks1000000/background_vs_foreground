from typing import Tuple, NamedTuple

import torch
from torch import nn, Tensor

from our_code.costs.base import BaseCost
from our_code.data.shuffle_objects import rotate, shuffle
from our_code.interface import ILabeledBoundingBoxImage, MultiClassification
from our_code.loops.train_loop import LossLog
from our_code.typing import ISimpleClassifierModel


class ShuffleCostLog(NamedTuple):
    loss: float
    foreground_loss: float
    background_loss: float
    classification_loss: float
    fuzzy_loss: float


class ShuffleCost(BaseCost[ISimpleClassifierModel[MultiClassification], ILabeledBoundingBoxImage, LossLog]):
    def __init__(self, loss: nn.Module):
        self._loss = loss

    def __call__(self, model: ISimpleClassifierModel[MultiClassification], data: ILabeledBoundingBoxImage) -> Tuple[float, LossLog]:
        images = data.image
        mixed_images = shuffle(images, data.bounding_box)
        prediction = model.classifier(mixed_images)
        foreground_labels = data.label
        background_labels = rotate(foreground_labels)
        foreground_loss = self._loss(prediction.foreground, foreground_labels)
        background_loss = self._loss(prediction.background, background_labels)
        loss = foreground_loss + background_loss
