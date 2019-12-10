from typing import Union, Iterable

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

import epipolar_nn.dataloaders.tum
import epipolar_nn.imips_pytorch.networks.convnet
import epipolar_nn.imips_pytorch.networks.imips
import epipolar_nn.imips_pytorch.train


def train_net():
    data_root = "./data"
    iterations = 100000  # default in imips. TUM dataset has 2336234 pairs ...
    learning_rate = 10e-6
    tum_dataset = epipolar_nn.dataloaders.tum.TUMMonocularStereoPairs(root=data_root, train=True, download=True)

    def adam_optimizer_factory(parameters: Union[Iterable[torch.Tensor], dict]) -> TorchOptimizer:
        return torch.optim.Adam(parameters, learning_rate)

    # TODO: load from checkpoint
    network: epipolar_nn.imips_pytorch.networks.imips.ImipsNet = epipolar_nn.imips_pytorch.networks.convnet.SimpleConv(
        num_convolutions=14,
        input_channels=1,
        output_channels=128,
    )
    network = network.cuda()

    trainer = epipolar_nn.imips_pytorch.train.ImipsTrainer(network, tum_dataset, adam_optimizer_factory)
    trainer.train(iterations)


if __name__ == "__main__":
    # todo handle parameters
    train_net()
