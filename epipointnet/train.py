import os
import socket
from datetime import datetime
from typing import Union, Iterable

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

import epipointnet.datasets.hpatches
import epipointnet.datasets.kitti
import epipointnet.imips_pytorch.losses.classic
import epipointnet.imips_pytorch.models.convnet
import epipointnet.imips_pytorch.models.imips
import epipointnet.imips_pytorch.trainer
import epipointnet.model
import epipointnet.trainer


def train_net():
    data_root = "./data"
    checkpoints_root = "./checkpoints/epinet"
    iterations = 100000  # 100000  # default in imips. TUM train_dataset has 2336234 pairs ...
    num_eval_samples = 50
    validation_frequency = 250  # 250  # default in imips
    learning_rate = 10e-6
    seed = 1

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    checkpoints_dir = os.path.join(checkpoints_root, current_time + '_' + socket.gethostname())

    os.makedirs(checkpoints_dir, exist_ok=True)

    kitti_dataset_train = epipointnet.datasets.kitti.KITTIMonocularStereoPairsSequence(data_root, "00")
    kitti_dataset_test = epipointnet.datasets.kitti.KITTIMonocularStereoPairsSequence(data_root, "05")

    def adam_optimizer_factory(parameters: Union[Iterable[torch.Tensor], dict]) -> TorchOptimizer:
        return torch.optim.Adam(parameters, learning_rate)

    imip_net = epipointnet.imips_pytorch.models.convnet.SimpleConv(
        num_convolutions=14,
        input_channels=3,
        output_channels=128,
    )
    imip_loss = epipointnet.imips_pytorch.losses.classic.ClassicImipLoss()

    epi_net = epipointnet.model.PatchBatchEpiPointNet(imip_net, side_info_size=0)

    trainer = epipointnet.trainer.EpiPointNetTrainer(
        epi_net,
        imip_loss,
        adam_optimizer_factory,
        kitti_dataset_train,
        kitti_dataset_test,
        num_eval_samples,
        checkpoints_dir,
        inlier_radius=3,
        seed=seed
    )

    with torch.autograd.detect_anomaly():
        trainer.train(iterations, validation_frequency)
    print("done")


if __name__ == "__main__":
    # todo handle parameters
    train_net()
