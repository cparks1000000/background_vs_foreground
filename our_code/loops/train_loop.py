from abc import abstractmethod
from typing import Generic, Callable, Any, Dict, Optional, Protocol, Sequence, NamedTuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from our_code.costs.base import BaseCost
from our_code.typing import ModelType, DataType, LogType, DataFactory


class BaseTrainLoop(Generic[ModelType, DataType]):
    @abstractmethod
    def __call__(self, model: ModelType, data:  Sequence[DataFactory[DataType]]) -> ModelType: ...


class BaseLogger(Generic[LogType]):
    @abstractmethod
    def __call__(self, log: LogType) -> None: ...

    @abstractmethod
    def batch_end(self, epoch_number: int, batch_number: int) -> None: ...

    @abstractmethod
    def epoch_end(self, epoch_number: int) -> None: ...


class NoLogger(BaseLogger[Any]):
    def __call__(self, log: Any): ...
    def batch_end(self, epoch_number: int, batch_number: int) -> None: ...
    def epoch_end(self, epoch_number: int) -> None: ...


class ILossLog(Protocol):
    # noinspection PyPropertyDefinition
    @property
    def loss(self) -> float: ...


class LossLog(NamedTuple):
    loss: float


class LossLogger(BaseLogger[ILossLog]):
    def __init__(self, batch_print: Callable[[int], bool] = lambda x: False, do_print: Callable[[str], None] = print):
        self._batch_print = batch_print
        self._total: float = 0
        self._count: int = 0
        self._last_print_total: float = 0
        self._last_print_count: int = 0
        self._do_print = do_print

    def __call__(self, log: LogType):
        self._total += log.loss
        self._count += 1

    def batch_end(self, epoch_number: int, batch_number: int) -> None:
        if self._batch_print(batch_number):
            self._do_print(str((self._total-self._last_print_total)/(self._count-self._last_print_count)))
            self._last_print_total = self._total
            self._last_print_count = self._count

    def epoch_end(self, epoch_number: int) -> None:
        self._do_print(str(self._total/self._count))
        self._total = 0.0
        self._count = 0


class NoScheduler(LRScheduler):
    # noinspection PyMissingConstructor
    def __init__(self): ...
    def state_dict(self): return {}
    def load_state_dict(self, state_dict: Dict[str, Any]): ...
    def get_last_lr(self) -> float: return 0.0
    def get_lr(self) -> float: return 0.0
    def print_lr(self, is_verbose: bool, group: Dict[str, Any], lr: float, epoch: Optional[int] = None): ...
    def step(self, epoch: Optional[int] = None): ...


class DefaultTrainLoop(BaseTrainLoop[ModelType, DataType, LogType]):
    def __init__(self, epoch_count: int, batch_size: int,
                 cost_factory: Callable[[], BaseCost[ModelType, DataType, LogType]],
                 optimizer_factory: Callable[[ModelType], Optimizer],
                 scheduler_factory: Callable[[Optimizer], LRScheduler] = lambda x: NoScheduler(),
                 logger_factory: Callable[[], BaseLogger[LogType]] = lambda: NoLogger()):
        self._epoch_count = epoch_count
        self._batch_size = batch_size
        self._cost_factory = cost_factory
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._logger_factory = logger_factory

    def __call__(self, model: ModelType, data_factory: Sequence[DataFactory[DataType]]):
        data = data_factory[0](self._batch_size)
        cost = self._cost_factory()
        optimizer = self._optimizer_factory(model)
        scheduler = self._scheduler_factory(optimizer)
        logger = self._logger_factory()
        for epoch_number in range(1, self._epoch_count + 1):
            # batch_number is intentionally here for debugging purposes.
            for batch_number, batch in enumerate(data):
                optimizer.zero_grad()
                _, log = cost(model, batch)
                logger(log)
                logger.batch_end(epoch_number, batch_number)
                optimizer.step(lambda: cost(model, batch)[0])
            logger.epoch_end(epoch_number)
            scheduler.step(epoch_number)
