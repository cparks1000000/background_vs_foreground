from typing import overload, TypeVar, Tuple as T, Iterable as I, Any, List, Tuple, Sequence, Iterable

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")
T7 = TypeVar("T7")
T8 = TypeVar("T8")
T9 = TypeVar("T9")


@overload
def typed_zip(x1: I[T1]) -> I[T1]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2]) -> I[T[T1, T2]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3]) -> I[T[T1, T2, T3]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4]) -> I[T[T1, T2, T3, T4]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4], x5: I[T5]) -> I[T[T1, T2, T3, T4, T5]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4], x5: I[T5], x6: I[T6], ) -> I[T[T1, T2, T3, T4, T5, T6]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4], x5: I[T5], x6: I[T6], x7: I[T7]) -> I[T[T1, T2, T3, T4, T5, T6, T7]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4], x5: I[T5], x6: I[T6], x7: I[T7], x8: I[T8]) -> I[T[T1, T2, T3, T4, T5, T6, T7, T8]]: ...
@overload
def typed_zip(x1: I[T1], x2: I[T2], x3: I[T3], x4: I[T4], x5: I[T5], x6: I[T6], x7: I[T7], x8: I[T8], x9: I[T9]) -> I[T[T1, T2, T3, T4, T5, T6, T7, T8, T9]]: ...


def typed_zip(*x: I[Any]) -> I[Any]: return zip(*x)  # type: ignore[misc]


@overload
def flip_flop(x: List[Tuple[T1, T2]]) -> Tuple[List[T1], List[T2]]: ...
@overload
def flip_flop(x: List[Tuple[T1, T2, T3]]) -> Tuple[List[T1], List[T2], List[T3]]: ...
@overload
def flip_flop(x: List[Tuple[T1, T2, T3, T4]]) -> Tuple[List[T1], List[T2], List[T3], List[T4]]: ...
@overload
def flip_flop(x: List[Tuple[T1, T2, T3, T4, T5]]) -> Tuple[List[T1], List[T2], List[T3], List[T4], List[T5]]: ...
@overload
def flip_flop(x: Tuple[List[T1], List[T2]]) -> List[Tuple[T1, T2]]: ...
@overload
def flip_flop(x: Tuple[List[T1], List[T2], List[T3]]) -> List[Tuple[T1, T2, T3]]: ...
@overload
def flip_flop(x: Tuple[List[T1], List[T2], List[T3], List[T4]]) -> List[Tuple[T1, T2, T3, T4]]: ...
@overload
def flip_flop(x: Tuple[List[T1], List[T2], List[T3], List[T4], List[T5]]) -> List[Tuple[T1, T2, T3, T4, T5]]: ...


def flip_flop(x: Any) -> Any:
    y = zip(*x)
    if isinstance(x, list): return tuple(y)
    if isinstance(x, tuple): return list(x)
    assert False, f"Expected `x` to be a tuple or list. Got {type(x)}."


def sum_sequences(vals: Iterable[Sequence[T1]]) -> List[T1]: return sum((list(val) for val in vals), [])
def cat(vals: Iterable[Sequence[T1]]) -> List[T1]: return sum_sequences(vals)
