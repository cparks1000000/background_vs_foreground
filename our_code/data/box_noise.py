from functools import cache
from typing import Sequence

import cv2
import torch
from torch import Tensor, nn
from torchvision.transforms import Compose, CenterCrop

from our_code.data.dataset import BirdsDataset
from our_code.data.shuffle_objects import get_width, get_height
from our_code.interface import BoundingBox
from our_code.typed_zip import typed_zip


def clamp(x: Tensor, low: float = -1, high: float = 1.0) -> Tensor:
    return torch.where(x < low, low, torch.where(x > high, high, x))


def fast_crop_and_paste(images: Tensor, noised_images: Tensor, boxes: Sequence[BoundingBox]):
    clone = images.clone()
    for b, box in typed_zip(range(len(images)), boxes):
        clone[b, :, box.h_min:box.h_max, box.w_min:box.w_max] = noised_images[b, :, box.h_min:box.h_max, box.w_min:box.w_max]
    return clone


def box_noise(images: Tensor, boxes: Sequence[BoundingBox], weight: float):
    scaled_weight = weight ** (1/4)
    noise = clamp(1.5 * torch.rand_like(images) - 0.75 + 0.25 * torch.randn_like(images))
    noised_image = (1-scaled_weight)*images + scaled_weight*noise
    return fast_crop_and_paste(images, noised_image, boxes)


def make_odd(x: int) -> int: return x + 1 if x % 2 == 0 else x


def _box_blur_help(image: Tensor, box: BoundingBox, weight: float) -> Tensor:
    kernel_size = (1+round(weight*get_height(box)), 1+round(weight*get_width(box)))
    temp = cv2.blur((image.numpy() + 1) / 2, kernel_size)
    return 2 * torch.from_numpy(temp) - 1


def box_blur(images: Tensor, boxes: Sequence[BoundingBox], weight: float):
    blurred_list = [_box_blur_help(image, box, weight/3) for image, box in zip(images.permute(0, 2, 3, 1), boxes)]
    blurred_images = torch.stack(blurred_list).permute(0, 3, 1, 2)
    return fast_crop_and_paste(images, blurred_images, boxes)


from torchshow import show
data = BirdsDataset(
    "C:/Users/cgparks/PycharmProjects/background_vs_foreground/_data/CUB_200_2011",
    True, Compose([lambda x: x/127.5-1])
)
