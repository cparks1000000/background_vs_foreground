import torch
import torchvision

from typing import TypedDict, Optional

from ..interface import FBClassification



class FCSettings(TypedDict):
    dim_list: list[int]
    activation_list: list[torch.nn.Module]


def get_linear(dim_list, a_list) -> torch.nn.Module:
    if len(a_list) == 1: a_list = a_list * (len(dim_list) - 1)
    return _get_linear(dim_list, a_list)


def _get_linear(dim_list, a_list) -> torch.nn.Module:
    out_dims = dim_list[1:]
    temp_list = []
    for in_dim, out_dim, act in zip(dim_list, out_dims, a_list):
        temp_list.append(torch.nn.Linear(in_dim, out_dim))
        temp_list.append(act)
    return torch.nn.Sequential(*temp_list)


class BigClass(torch.nn.Module):
    #TODO add checker on class list
    #
    def __init__(self,
                 class_list: tuple[int, int, int],
                 linear_settings_list: list[FCSettings],
                 down_sampler: torch.nn.Module = torchvision.models.resnet18('DEFAULT'),
                 image_size: Optional[tuple[int, int, int]] = None,
                 unknown_dim: bool = False):
        super().__init__()
        self.down_sampler = torch.nn.Sequential(*[down_sampler, torch.nn.Flatten()])
        dim = self._get_dim(image_size) if (image_size and unknown_dim) else []
        self._check_settings(class_list, linear_settings_list, )
        self.splitter = self._build_splitter(linear_settings_list, class_list, dim)

    @staticmethod
    def _check_settings(class_list, linear_settings_list, ):
        assert (len(class_list) == len(linear_settings_list),
                r"Class list and linear settings list must have the same length. " +
                f"class_list length is {len(class_list)} " +
                f"while linear_settings_list length is {len(linear_settings_list)}"
                )

    @staticmethod
    def _build_splitter(linear_settings_list: list[FCSettings],
                        class_list: list[int],
                        dim: list[int]):
        temp_splitter_list: list[torch.nn.Module] = []
        for linear_setting, c_number in zip(linear_settings_list, class_list):
            dim_list = dim + linear_setting['dim_list'] + [c_number]
            a_list = linear_setting['activation_list']
            temp_splitter_list.append(get_linear(dim_list, a_list))
        return Splitter(temp_splitter_list)

    def _get_dim(self, image_size: tuple[int, int, int]) -> list[int]:
        return [self.down_sampler(torch.zeros((1, *image_size))).shape[1]]

    def forward(self, inputs: torch.Tensor) -> FBClassification:
        feature = self.down_sampler(inputs)
        final_features = self.splitter(feature)
        return FBClassifcatioin(*final_features)


class Splitter(torch.nn.Module):
    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self._models = torch.nn.ModuleList(models)

    def forward(self, inputs: torch.Tensor) -> list[torch.nn.Module]:
        feature_list = []
        for model in self._models:
            feature_list.append(model(inputs))
        return feature_list


def main():
    background_l_settings = {'dim_list': [10, 10, 10],
                             'activation_list': [torch.nn.LeakyReLU()]
                             }
    class_l_settings = {'dim_list': [6, 50, 3, 8],
                        'activation_list': [torch.nn.ReLU()]
                        }

    class_list = [2, 4, ]
    linear_settings = [background_l_settings, class_l_settings]
    n = BigClass(class_list=class_list,
                 linear_settings_list=linear_settings,
                 image_size=(3, 640, 640),
                 unknown_dim=True
                 )
    test = torch.zeros(1, 3, 640, 640)



if __name__ == "__main__": main()
