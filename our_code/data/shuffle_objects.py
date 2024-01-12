from math import ceil
from typing import Tuple, List

import torch
from torch import Tensor
from torchvision.transforms.v2.functional import resize

from our_code.interface import BoundingBox


def get_height(box: BoundingBox) -> int: return box.h_max - box.h_min


def get_width(box: BoundingBox) -> int: return box.w_max - box.w_min


def get_target_size(foreground_shape: Tuple[int, int], foreground_box: BoundingBox,
                    background_box: BoundingBox, preserve_ratio: bool = True) -> Tuple[int, int]:
    """Gets the scaling ratio so that the background box fits completely over the foreground box."""
    height_ratio = get_height(foreground_box)/get_height(background_box)
    width_ratio = get_width(foreground_box)/get_width(background_box)
    if preserve_ratio:
        width_ratio = max(height_ratio, width_ratio)
        height_ratio = max(width_ratio, height_ratio)
    height = ceil(get_height(background_box)*height_ratio)
    width = ceil(get_width(background_box)*width_ratio)
    return min(height, foreground_shape[0]), min(width, foreground_shape[1])


def get_delta(x_min: int, x_max: int, length: int) -> int:
    return -x_min if x_min < 0 else length-x_max if x_max > length else 0


def get_paste_box(foreground_size: Tuple[int, int], target_size: Tuple[int, int], foreground_box: BoundingBox) -> BoundingBox:
    height_difference = target_size[0] - get_height(foreground_box)
    width_difference = target_size[1] - get_width(foreground_box)
    h_min = foreground_box.h_min - height_difference // 2
    w_min = foreground_box.w_min - width_difference // 2
    out = BoundingBox(h_min=h_min, h_max=h_min + target_size[0], w_min=w_min, w_max=w_min + target_size[1])
    delta_h = get_delta(out.h_min, out.h_max, foreground_size[0])
    delta_w = get_delta(out.w_min, out.w_max, foreground_size[1])
    return shift_box(out, delta_h, delta_w)


def shift_box(box: BoundingBox, h: int, w: int) -> BoundingBox:
    return BoundingBox(h_min=box.h_min+h, h_max=box.h_max+h, w_min=box.w_min+w, w_max=box.w_max+w)


def crop_background(image: Tensor, box: BoundingBox): return image[:, box.h_min:box.h_max, box.w_min:box.w_max]


def paste(foreground_image: Tensor, background_image: Tensor, box: BoundingBox):
    foreground_clone = foreground_image.clone()
    foreground_clone[:, box.h_min:box.h_max, box.w_min:box.w_max] = background_image
    return foreground_clone


def crop_and_paste(foreground_image: Tensor, foreground_box: BoundingBox,
                   background_image: Tensor, background_box: BoundingBox) -> Tensor:
    foreground_size: Tuple[int, int] = foreground_image.shape[1:]
    target_size = get_target_size(foreground_size, foreground_box, background_box)
    background_crop = resize(crop_background(background_image, background_box), target_size)
    paste_box = get_paste_box(foreground_size, target_size, foreground_box)
    return paste(foreground_image, background_crop, paste_box)


def rotate(x: Tensor, distance: int = 1) -> Tensor: return torch.cat([x[distance:], x[0:distance]])


def shuffle(xs: Tensor, x_boxes: List[BoundingBox]) -> Tensor:
    ys = rotate(xs)
    y_boxes = x_boxes[1:] + x_boxes[0:1]
    return torch.stack([crop_and_paste(x, x_box, y, y_box) for (x, x_box, y, y_box) in zip(xs, x_boxes, ys, y_boxes)])
