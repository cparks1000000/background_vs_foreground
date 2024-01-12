from typing import Generic, Tuple

from our_code.typing import ModelType, DataType, LogType


class BaseCost(Generic[ModelType, DataType, LogType]):
    """
    In the cost function, you should calculate all derivatives necessary for the model to be updated.
    For optimizer purposes, the model must return the loss.
    """
    def __call__(self, model: ModelType, data: DataType) -> Tuple[float, LogType]: ...
