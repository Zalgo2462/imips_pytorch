import torch

from .preprocess import PreprocessModule


class PreprocessIMIPCenter(PreprocessModule):
    def __init__(self):
        super(PreprocessModule, self).__init__()

    def preprocess(self, image: torch.Tensor):
        image = image - 127  # the original IMIP implementation centers data between [-127, 128]
        return image

    def output_channels(self, input_channels: int) -> int:
        return input_channels
