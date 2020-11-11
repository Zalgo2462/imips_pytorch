import torch


class PreprocessModule(torch.nn.Module):
    def __init__(self):
        super(PreprocessModule, self).__init__()

    @torch.autograd.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.preprocess(image)

    def preprocess(self, image: torch.Tensor):
        raise NotImplementedError()

    def output_channels(self, input_channels: int) -> int:
        raise NotImplementedError()


class PreprocessIdentity(PreprocessModule):
    def __init__(self):
        super(PreprocessIdentity, self).__init__()

    def preprocess(self, image: torch.Tensor):
        return image

    def output_channels(self, input_channels: int) -> int:
        return input_channels
