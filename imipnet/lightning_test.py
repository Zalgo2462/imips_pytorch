import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import move_data_to_device

from imipnet.datasets.shuffle import ShuffledDataset
from imipnet.lightning_module import IMIPLightning, test_dataset_registry

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument('test_set', choices=test_dataset_registry.keys())
parser.add_argument('--data_root', default="./data")
parser.add_argument('--n_eval_samples', type=int, default=-1)
parser.add_argument("--output_dir", type=str, default="./test_results")

params = parser.parse_args()
run_name = os.path.basename(os.path.dirname(params.checkpoint))

checkpoint_net = IMIPLightning.load_from_checkpoint(params.checkpoint, strict=False)  # calls seed everything
checkpoint_net.freeze()

# override test set
if params.test_set is not None:
    checkpoint_net.hparams.test_set = params.test_set
    checkpoint_net.hparams.n_eval_samples = params.n_eval_samples
    checkpoint_net.test_set = ShuffledDataset(
        test_dataset_registry[params.test_set](params.data_root)
    )

print("Evaluating {} on {}".format(checkpoint_net.get_name(), checkpoint_net.hparams.test_set))

# TODO: load device from params
if checkpoint_net.hparams.n_eval_samples > 0:
    print("Number of samples: {}".format(checkpoint_net.hparams.n_eval_samples))
    trainer = Trainer(gpus=[0], limit_test_batches=checkpoint_net.hparams.n_eval_samples)
else:
    print("Number of samples: {}".format(len(checkpoint_net.test_set)))
    trainer = Trainer(gpus=[0], limit_test_batches=1.0)

results = move_data_to_device(trainer.test(checkpoint_net)[0], torch.device("cpu"))


def save_results(result_dict: dict, output_dir, test_set, n_eval_samples, name):
    if n_eval_samples < 1:
        n_eval_samples = "all"
    else:
        n_eval_samples = str(n_eval_samples)

    output_subdir = os.path.join(output_dir, test_set, n_eval_samples)
    os.makedirs(output_subdir, exist_ok=True)

    torch.save(result_dict, os.path.join(output_subdir, name + ".pt"))


save_results(
    results, params.output_dir,
    checkpoint_net.hparams.test_set,
    checkpoint_net.hparams.n_eval_samples,
    run_name,
)

"""
apparent_matching_scores = results["matching_scores"]["apparent"].cpu()
true_matching_scores = results["matching_scores"]["true"].cpu()


def draw_matching_scores(title, run_name, matching_scores):
    import matplotlib.pyplot as plt
    import numpy as np
    frac_gt = 1 - (np.arange(1, len(matching_scores) + 1) / len(matching_scores))

    plt.title(title)
    plt.step(matching_scores, frac_gt, linestyle="-", label=run_name)
    plt.axvline(x=10. / 128., label='10 inliers at 128', color='black')
    plt.grid()
    plt.legend()
    plt.xlabel('Matching score')
    plt.ylabel('Fraction of pairs with higher matching score')
    plt.ylim([0, 1])
    plt.show()


draw_matching_scores(
    "Train: " + checkpoint_net.hparams.train_set + ", Test: " + checkpoint_net.hparams.test_set,
    checkpoint_net.get_name(),
    true_matching_scores
)
"""
