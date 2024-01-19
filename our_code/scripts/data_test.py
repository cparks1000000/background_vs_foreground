from typing import Tuple

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from our_code.data.dataset import BirdsDataset


def try_expanding(image_size: Tuple[int, int], crop_size: Tuple[int, int]) -> Tuple[int, int]:
    if crop_size[0] <= crop_size[1]:
        return min(crop_size[0], image_size[0]), crop_size[1]



def main():
    from torchshow import show
    dataset = BirdsDataset(
        "C:/Users/cgparks/PycharmProjects/background_vs_foreground/_data/CUB_200_2011",
        True, Compose([lambda x: x / 127.5 - 1])
    )
    loader = DataLoader(dataset, batch_size=4)
    datapoint = dataset[1]
    image = dataset[1].image
    box = datapoint.bounding_box

    show(image)
    # scale
    # _, h, w = image.shape
    # tensor = resize(tensor)
    # crop = crop(tensor, cords)
    # crop = resize(crop)
    # torchshow.show(crop)
