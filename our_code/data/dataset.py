from functools import reduce
from pathlib import Path
from typing import Callable, Sequence, NewType, Dict, Optional, List, cast

import torch
from numpy import ndarray
from pandas import DataFrame, read_csv
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image as readim

from our_code.interface import LabeledBoundingBoxImage, BoundingBox

#Should include keys "id", "path", "label", x", "y", "width", "height"
BirdsDataFrame = NewType("BirdsDataFrame", DataFrame)


def to_birds_frame(x: DataFrame) -> BirdsDataFrame:
    for key in ("id", "path", "label", "x", "y", "width", "height"): assert key in x, f"{key} is missing."
    return BirdsDataFrame(x)


def merge_all(frames: Sequence[DataFrame], merge_key: str) -> DataFrame:
    return reduce(lambda x, y: x.merge(y, on=merge_key), frames)


def get_data_info(root: Path | str, train: bool) -> BirdsDataFrame:
    if isinstance(root, str): root = Path(root)
    path = read_csv(root/"images.txt", delim_whitespace=True, names=["id", "path"])
    # noinspection PyTypeChecker
    path["path"] = str(root) + "/images/" + path["path"]
    label = read_csv(root/"image_class_labels.txt", delim_whitespace=True, names=["id", "label"])
    split = read_csv(root/"train_test_split.txt", delim_whitespace=True, names=["id", "split"])
    boxes = read_csv(root/"bounding_boxes.txt", delim_whitespace=True, names=["id", "x", "y", "width", "height"])
    out = merge_all([path, label, split, boxes], merge_key="id")
    return to_birds_frame(out[out["split"] == (0 if train else 1)])


def get_data(i: int, read_image: Callable[[str], Tensor], frame: BirdsDataFrame, device: Optional[str]) -> LabeledBoundingBoxImage:
    row = frame.iloc[i]
    bounding_box = BoundingBox(
        h_min=int(row["y"]), h_max=int(row["y"] + row["height"]),
        w_min=int(row["x"]), w_max=int(row["x"] + row["width"])
    )
    image = read_image(row["path"])
    if device is not None: image = image.to(device)
    return LabeledBoundingBoxImage(image=image, label=Tensor(row["label"]), bounding_box=cast(List[BoundingBox], bounding_box))


class LoadWithCache:
    def __init__(self, max_size: Optional[int] = None):
        self._cache: Dict[str, ndarray] = {}
        self._max_size = max_size

    def __call__(self, path: str) -> Tensor:
        if path in self._cache: return torch.from_numpy(self._cache[path])
        temp = readim(path)
        if self._max_size is not None and len(self._cache) < self._max_size: self._cache[path] = temp.numpy()
        return temp


class BirdsDataset(Dataset[LabeledBoundingBoxImage]):
    def __init__(self, root: str, train: bool, transform: Callable[[Tensor], Tensor], device: Optional[str] = None,
                 load_function: Callable[[str], Tensor] = LoadWithCache()):
        self._data_info = get_data_info(Path(root), train)
        self._length = len(self._data_info["id"])
        self._read_image: Callable[[str], Tensor] = lambda x: transform(load_function(x))
        self._device = device

    def __len__(self) -> int: return self._length

    def __getitem__(self, index: int) -> LabeledBoundingBoxImage: return get_data(index, self._read_image, self._data_info, self._device)
