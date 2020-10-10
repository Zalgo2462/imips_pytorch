import os
import socket
from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from epipointnet2.imips_unet.UNet import UNet

seed_everything(0)

parser = ArgumentParser()
# parser = Trainer.add_argparse_args(parser)
parser = UNet.add_model_specific_args(parser)
args = parser.parse_args("")

name = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()
logger = TensorBoardLogger("./runs", name)
os.makedirs('./checkpoints/' + name + '/', exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/' + name + '/',
    save_last=True,
    verbose=True,
    monitor='val_true_inliers',
    mode='max'
)
trainer = Trainer(logger=logger, gpus=[0], val_check_interval=250, max_epochs=20,
                  checkpoint_callback=checkpoint_callback)
unet = UNet(args)
trainer.fit(unet)
