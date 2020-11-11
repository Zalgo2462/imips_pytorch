import torch

from .preprocess import PreprocessModule


class PreprocessNormalize(PreprocessModule):
    def __init__(self):
        super(PreprocessModule, self).__init__()

    def preprocess(self, image: torch.Tensor):
        image = (image / 127.5) - 1.0  # scale to [-1, 1]
        return image

    def output_channels(self, input_channels: int) -> int:
        return input_channels
