from typing import NamedTuple, Generic, Callable, Sequence

from our_code.loops.test_loop import BaseTestLoop
from our_code.loops.train_loop import BaseTrainLoop
from our_code.typing import DataFactory, DataType, LabeledImage, ModelType, Device, ScoreType


class DataInfo(NamedTuple):
    channels: int
    height: int
    width: int
    class_count: int


class DataGroup(NamedTuple, Generic[DataType]):
    info: DataInfo
    train: Sequence[DataFactory[DataType]]
    test: Sequence[DataFactory[LabeledImage]]


def run(
    device: str, model_factory: Callable[[Device, DataInfo], ModelType],
    data_factory: Callable[[Device], DataGroup[DataType]], train_loop: BaseTrainLoop[ModelType, DataType],
    test_loop: BaseTestLoop[ModelType, LabeledImage, ScoreType], show_score: Callable[[Sequence[ScoreType]], str]
) -> str:
    device = Device(device)
    data = data_factory(device)
    model = model_factory(device, data.info)
    trained_model = train_loop(model, data.train)
    # noinspection PyTypeChecker
    return show_score(test_loop(trained_model, data.test))
