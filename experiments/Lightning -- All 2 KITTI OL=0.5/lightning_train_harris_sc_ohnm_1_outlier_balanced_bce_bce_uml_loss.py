import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from imipnet.lightning_module import IMIPLightning

parser = ArgumentParser()
parser = IMIPLightning.add_model_specific_args(parser)
args = parser.parse_args([
    "--loss", "outlier-balanced-bce-bce-uml", "--n_top_patches", "1",
    "--train_set", "tum-megadepth-blender-gray", "--eval_set", "tum-megadepth-blender-gray",
    "--test_set", "tum-megadepth-blender-gray",
    "--preprocess", "harris"
])

imip_module = IMIPLightning(args)
name = imip_module.get_new_run_name()

logger = TensorBoardLogger("./runs", name)

checkpoint_dir = os.path.join(".", "checkpoints", "simple-conv", name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_dir,
    save_last=True,
    verbose=True,
    monitor="eval_true_inliers",
    mode='max',
    period=0  # don't wait for a new epoch to save a better model
)

overfit_val = args.overfit_n

trainer = Trainer(logger=logger, gpus=[0], val_check_interval=250 if overfit_val == 0 else overfit_val,
                  max_steps=20000 * 5, limit_train_batches=20000 if overfit_val == 0 else 1.0,
                  max_epochs=2000, reload_dataloaders_every_epoch=False,
                  checkpoint_callback=checkpoint_callback)
trainer.fit(imip_module)
