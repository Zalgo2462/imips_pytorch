import os
import socket
from datetime import datetime
from typing import Union, Iterable

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

import epipolar_nn.dataloaders.hpatches
import epipolar_nn.dataloaders.tum
import epipolar_nn.imips_pytorch.networks.convnet
import epipolar_nn.imips_pytorch.networks.imips
import epipolar_nn.imips_pytorch.train


def train_net():
    data_root = "./data"
    checkpoints_root = "./checkpoints/imips"
    iterations = 100000  # 100000  # default in imips. TUM train_dataset has 2336234 pairs ...
    num_eval_samples = 50
    validation_frequency = 250  # 250  # default in imips
    learning_rate = 10e-6
    seed = 1

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    checkpoints_dir = os.path.join(checkpoints_root, current_time + '_' + socket.gethostname())

    os.makedirs(checkpoints_dir, exist_ok=True)

    tum_dataset = epipolar_nn.dataloaders.tum.TUMMonocularStereoPairs(root=data_root, train=True, download=True)
    hpatches_dataset = epipolar_nn.dataloaders.hpatches.HPatchesSequencesStereoPairs(
        root=data_root, train=False, download=True
    )

    def adam_optimizer_factory(parameters: Union[Iterable[torch.Tensor], dict]) -> TorchOptimizer:
        return torch.optim.Adam(parameters, learning_rate)

    network: epipolar_nn.imips_pytorch.networks.imips.ImipsNet = epipolar_nn.imips_pytorch.networks.convnet.SimpleConv(
        num_convolutions=14,
        input_channels=1,
        output_channels=128,
    )
    network = network.cuda()

    trainer = epipolar_nn.imips_pytorch.train.ImipsTrainer(
        network,
        adam_optimizer_factory,
        tum_dataset,
        hpatches_dataset,
        num_eval_samples,
        checkpoints_dir,
        inlier_radius=3,
        seed=seed
    )
    trainer.train(iterations, validation_frequency)
    print("done")


if __name__ == "__main__":
    # todo handle parameters
    train_net()
