from abc import abstractmethod
from typing import Generic, Callable, List, Sequence

from our_code.typing import ModelType, DataType, ScoreType, DataFactory


class BaseTestLoop(Generic[ModelType, DataType, ScoreType]):
    def __call__(self, model: ModelType, data_factory: Sequence[DataFactory[DataType]]) -> Sequence[ScoreType]: ...


class DefaultTestLoop(BaseTestLoop[ModelType, DataType, ScoreType]):
    def __init__(self, batch_size: int,
                 metric: Callable[[ModelType, DataType], ScoreType],
                 aggregation: Callable[[List[ScoreType]], ScoreType]):
        self._batch_size = batch_size
        self._metric = metric
        self._aggregation = aggregation

    def __call__(self, model: ModelType, data_factory: Sequence[DataFactory[DataType]]) -> Sequence[ScoreType]:
        data = data_factory[0](self._batch_size)
        scores: List[ScoreType] = []
        for batch in data: scores.append(self._metric(model, batch))
        return self._aggregation(scores),


class BaseValidateLoop(Generic[ModelType, ScoreType]):
    @abstractmethod
    def __call__(self, model: ModelType) -> ScoreType: ...


class ValidateLoop(BaseValidateLoop[ModelType, ScoreType]):
    def __init__(self, test_loop: BaseTestLoop[ModelType, DataType, ScoreType], data_factory: DataFactory[DataType]):
        self._test_loop = test_loop
        self._data_factory = data_factory

    def __call__(self, model: ModelType) -> ScoreType: return self._test_loop(model, self._data_factory)
