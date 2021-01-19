import multiprocessing
import os.path
import socket
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Tuple, List, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import imipnet.losses.ohnm_1_classic
import imipnet.losses.ohnm_outlier_balanced_bce
import imipnet.losses.ohnm_outlier_balanced_classic
import imipnet.models.convnet
import imipnet.models.preprocess.center
import imipnet.models.preprocess.harris
import imipnet.models.preprocess.hessian
import imipnet.models.preprocess.normalize
import imipnet.models.preprocess.preprocess
import imipnet.models.strided_conv
from imipnet.data.pairs import CorrespondencePair
from imipnet.datasets.blender import BlenderStereoPairs
from imipnet.datasets.colmap import COLMAPStereoPairs
from imipnet.datasets.kitti import KITTIMonocularStereoPairs
from imipnet.datasets.shuffle import ShuffledDataset
from imipnet.datasets.tum_mono import TUMMonocularStereoPairs

colmap_max_image_bytes = 1750000

preprocess_registry = {
    "harris": imipnet.models.preprocess.harris.PreprocessHarris,
    "hessian": imipnet.models.preprocess.hessian.PreprocessHessian,
    "normalize": imipnet.models.preprocess.normalize.PreprocessNormalize,
    "center": imipnet.models.preprocess.center.PreprocessIMIPCenter,
    "": imipnet.models.preprocess.preprocess.PreprocessIdentity
}

model_registry = {
    "simple-conv": imipnet.models.convnet.SimpleConv,
    "strided-simple-conv": imipnet.models.strided_conv.StridedConv,
}

loss_registry = {
    "1-maxima-patch-classic": imipnet.losses.ohnm_1_classic.OHNM1ClassicImipLoss,
    "outlier-balanced-classic": imipnet.losses.ohnm_outlier_balanced_classic.OHNMClassicImipLoss,
    "outlier-balanced-bce-bce-uml": imipnet.losses.ohnm_outlier_balanced_bce.OHNMBCELoss
}

train_dataset_registry = {
    "tum-mono": lambda data_root: TUMMonocularStereoPairs(data_root, "train", True, 0.3),
    "kitti-gray": lambda data_root: KITTIMonocularStereoPairs(data_root, "train", True, False, 0.3),
    "kitti-color": lambda data_root: KITTIMonocularStereoPairs(data_root, "train", True, False, 0.3),
    "megadepth-gray": lambda data_root: torch.utils.data.ConcatDataset((
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0035"), False,
                          max_image_bytes=colmap_max_image_bytes),
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0036"), False,
                          max_image_bytes=colmap_max_image_bytes),
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0039"), False,
                          max_image_bytes=colmap_max_image_bytes),
    )),
    "megadepth-color": lambda data_root: torch.utils.data.ConcatDataset((
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0035"), True,
                          max_image_bytes=colmap_max_image_bytes),
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0036"), True,
                          max_image_bytes=colmap_max_image_bytes),
        COLMAPStereoPairs(data_root, os.path.join("MegaDepth-Pairs", "train", "0039"), True,
                          max_image_bytes=colmap_max_image_bytes),
    )),
    "blender-livingroom-color": lambda data_root: BlenderStereoPairs(data_root, "livingroom_1", True, True),
    "blender-livingroom-gray": lambda data_root: BlenderStereoPairs(data_root, "livingroom_1", True, False),
}

test_dataset_registry = {
    "tum-mono": lambda data_root: TUMMonocularStereoPairs(data_root, "test", True, 0.3),
    "kitti-gray": lambda data_root: KITTIMonocularStereoPairs(data_root, "test", True, False, 0.3),
    "kitti-gray-0.5": lambda data_root: KITTIMonocularStereoPairs(data_root, "test", True, False, 0.5),
    "kitti-color": lambda data_root: KITTIMonocularStereoPairs(data_root, "test", True, False, 0.3),
    "megadepth-gray": lambda data_root: COLMAPStereoPairs(
        data_root, os.path.join("MegaDepth-Pairs", "test", "0032"), False,
        max_image_bytes=colmap_max_image_bytes),
    "megadepth-color": lambda data_root: COLMAPStereoPairs(
        data_root, os.path.join("MegaDepth-Pairs", "test", "0032"), True,
        max_image_bytes=colmap_max_image_bytes),
    "blender-livingroom-color": lambda data_root: BlenderStereoPairs(data_root, "livingroom_3", True, True),
    "blender-livingroom-gray": lambda data_root: BlenderStereoPairs(data_root, "livingroom_3", True, False),
}

validation_dataset_registry = {
    "tum-mono": lambda data_root: TUMMonocularStereoPairs(data_root, "validation", True, 0.3),
    "kitti-gray": lambda data_root: KITTIMonocularStereoPairs(data_root, "validation", True, False, 0.3),
    "kitti-gray-0.5": lambda data_root: KITTIMonocularStereoPairs(data_root, "validation", True, False, 0.5),
    "kitti-color": lambda data_root: KITTIMonocularStereoPairs(data_root, "validation", True, False, 0.3),
    "megadepth-gray": lambda data_root: COLMAPStereoPairs(
        data_root, os.path.join("MegaDepth-Pairs", "validation", "0008"), False,
        max_image_bytes=colmap_max_image_bytes),
    "megadepth-color": lambda data_root: COLMAPStereoPairs(
        data_root, os.path.join("MegaDepth-Pairs", "validation", "0008"), True,
        max_image_bytes=colmap_max_image_bytes),
    "blender-livingroom-color": lambda data_root: BlenderStereoPairs(data_root, "livingroom_2", True, True),
    "blender-livingroom-gray": lambda data_root: BlenderStereoPairs(data_root, "livingroom_2", True, False),
}


class IMIPLightning(pl.LightningModule):

    def __init__(self, hparams):

        if isinstance(hparams, dict):  # when loading chkpts, pytorch-lightning is restoring hparams as an dict
            hparams = Namespace(**hparams)

        super(IMIPLightning, self).__init__()
        self.save_hyperparameters(hparams)

        pl.seed_everything(hparams.seed)

        self.train_set = train_dataset_registry[hparams.train_set](hparams.data_root)

        if hparams.overfit_n > 0:
            self.train_set = ShuffledDataset(self.train_set, hparams.overfit_n)
            self.train_eval_set = self.train_set
        else:
            self.train_eval_set = ShuffledDataset(
                self.train_set,
                hparams.n_eval_samples
            )

        self.eval_set = ShuffledDataset(
            validation_dataset_registry[hparams.eval_set](hparams.data_root),
            hparams.n_eval_samples
        )

        self.test_set = ShuffledDataset(
            test_dataset_registry[hparams.test_set](hparams.data_root),
            hparams.n_eval_samples
        )

        self.preprocess = preprocess_registry[hparams.preprocess]()
        channels_in = self.preprocess.output_channels(hparams.channels_in)

        self.network = model_registry[hparams.model](hparams.n_convolutions, channels_in, hparams.channels_out)
        self._loss = loss_registry[hparams.loss]()
        self._lr = hparams.learning_rate

        self._n_top_patches = hparams.n_top_patches
        self._inlier_radius = hparams.inlier_radius

        # store data between training_step calls with different optimizer indices
        self.__training_step_cache = {}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root', default="./data")
        parser.add_argument('--train_set', choices=train_dataset_registry.keys(), default="tum-mono")
        parser.add_argument('--eval_set', choices=validation_dataset_registry.keys(), default="kitti-gray")
        parser.add_argument('--test_set', choices=test_dataset_registry.keys(), default="kitti-gray")
        parser.add_argument('--n_eval_samples', type=int, default=50)
        parser.add_argument('--n_convolutions', type=int, default=14)
        parser.add_argument('--model', choices=model_registry.keys(), default="simple-conv")
        parser.add_argument('--preprocess', choices=preprocess_registry.keys(), default="")
        parser.add_argument('--channels_in', type=int, default=1)
        parser.add_argument('--channels_out', type=int, default=128)
        parser.add_argument('--loss', choices=loss_registry.keys(), default="outlier-balanced-classic")
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--inlier_radius', type=float, default=3.0)
        parser.add_argument('--learning_rate', type=float, default=10e-6)
        parser.add_argument('--n_top_patches', type=int, default=1)
        parser.add_argument('--overfit_n', type=int, default=0)
        return parser

    def get_name(self):
        preprocess_tag = self.hparams.preprocess + "-" if len(self.hparams.preprocess) > 0 else ""
        return preprocess_tag + "sc-" + str(self.hparams.n_convolutions) + "_" + \
               "ohnm-" + str(self.hparams.n_top_patches) + "_" + \
               self.hparams.model + "-model_" + \
               self.hparams.loss + "-loss_" + \
               "train-" + self.hparams.train_set

    def get_new_run_name(self):
        return self.get_name() + "_" + \
               (("overfit-" + str(self.hparams.overfit_n) + "_") if self.hparams.overfit_n > 0 else "") + \
               "eval-" + self.hparams.eval_set + "_" + \
               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), self._lr)
        # return optimizer twice so we get two train steps per minibatch
        return [optimizer, optimizer]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=1, collate_fn=CorrespondencePair.collate_for_torch,
            num_workers=1 + multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        train_eval_loader = DataLoader(
            self.train_eval_set, batch_size=1, collate_fn=CorrespondencePair.collate_for_torch,
            num_workers=1 + multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=True
        )

        eval_loader = DataLoader(
            self.eval_set, batch_size=1, collate_fn=CorrespondencePair.collate_for_torch,
            num_workers=1 + multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=True
        )

        return [train_eval_loader, eval_loader]

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=1, collate_fn=CorrespondencePair.collate_for_torch,
            num_workers=1 + multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=True
        )

    def forward(self, patch_batch: torch.Tensor, keepDim: bool):
        return self.network(patch_batch, keepDim)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # set modules to training mode
        self.network.train(True)
        self._loss.train(True)

        # Run preprocess step and store the results for the next pass
        batch[0] = torch.stack([self.preprocess(img) for img in batch[0]], dim=0)
        batch[1] = torch.stack([self.preprocess(img) for img in batch[1]], dim=0)

        # unpack data since batch size is 1
        img_1 = batch[0][0]
        img_2 = batch[1][0]
        # name = batch[2][0]
        correspondence_func = batch[3][0]

        if optimizer_idx == 0:  # train on image 1 of pair
            # Find top k keypoints in each image
            img_1_kp_candidates, _ = self.network.extract_top_k_keypoints(img_1, self._n_top_patches)  # 2 x c x k
            img_2_kp_candidates, _ = self.network.extract_top_k_keypoints(img_2, self._n_top_patches)
            self.__training_step_cache.update({
                "img_1_kp_candidates": img_1_kp_candidates,
                "img_2_kp_candidates": img_2_kp_candidates,
            })

            img_1_correspondences, img_1_correspondences_mask = self.find_correspondences(
                correspondence_func, img_2_kp_candidates[:, :, 0], img_1.shape, inverse=True,
                exclude_border_px=(self.network.receptive_field_diameter() - 1) // 2
            )  # 2 x c

            (img_1_kp_candidates, img_1_inlier_channels_by_max, img_1_outlier_channels_by_max,
             img_1_inlier_channels_by_top_k,
             img_1_outlier_channels_by_top_k) = self.sort_candidates_and_generate_labels(
                img_1_kp_candidates, img_1_correspondences, img_1_correspondences_mask, self._inlier_radius
            )
            self.__training_step_cache.update({
                "img_1_inlier_channels_by_max": img_1_inlier_channels_by_max,
                "img_1_outlier_channels_by_max": img_1_outlier_channels_by_max,
                "img_1_inlier_channels_by_top_k": img_1_inlier_channels_by_top_k,
                "img_1_outlier_channels_by_top_k": img_1_outlier_channels_by_top_k
            })

            patch_diameter = self.network.receptive_field_diameter()

            # Generate a loss for image 1
            maxima_patches = self.image_to_patch_batch(
                img_1, img_1_kp_candidates.flatten(1),
                patch_diameter
            )
            corr_patches = torch.zeros(
                img_1_correspondences.shape[1], img_1.shape[0], patch_diameter, patch_diameter,
                dtype=maxima_patches.dtype,
                device=self._device
            )
            corr_patches[img_1_correspondences_mask, :, :, :] = self.image_to_patch_batch(
                img_1, img_1_correspondences[:, img_1_correspondences_mask], patch_diameter
            )

            maximizer_outputs: torch.Tensor = self(maxima_patches, False)
            correspondence_outputs: torch.Tensor = self(corr_patches, False)

            loss, img_1_loss_logs = self._loss.forward_with_log_data(
                maximizer_outputs, correspondence_outputs,
                img_1_inlier_channels_by_top_k, img_1_outlier_channels_by_top_k
            )

            img_1_loss_logs = {
                "training/image 1/" + key: img_1_loss_logs[key] for key in img_1_loss_logs
                if img_1_loss_logs[key] is not None
            }

            loss_logs = img_1_loss_logs

        else:  # train on image 2 of pair
            img_1_kp_candidates = self.__training_step_cache["img_1_kp_candidates"]
            img_2_kp_candidates = self.__training_step_cache["img_2_kp_candidates"]

            img_2_correspondences, img_2_correspondences_mask = self.find_correspondences(
                correspondence_func, img_1_kp_candidates[:, :, 0], img_2.shape, inverse=False,
                exclude_border_px=(self.network.receptive_field_diameter() - 1) // 2
            )  # 2 x c

            (img_2_kp_candidates, img_2_inlier_channels_by_max, img_2_outlier_channels_by_max,
             img_2_inlier_channels_by_top_k,
             img_2_outlier_channels_by_top_k) = self.sort_candidates_and_generate_labels(
                img_2_kp_candidates, img_2_correspondences, img_2_correspondences_mask, self._inlier_radius
            )

            img_1_inlier_channels_by_max = self.__training_step_cache["img_1_inlier_channels_by_max"]
            img_1_outlier_channels_by_max = self.__training_step_cache["img_1_outlier_channels_by_max"]
            img_1_inlier_channels_by_top_k = self.__training_step_cache["img_1_inlier_channels_by_top_k"]
            img_1_outlier_channels_by_top_k = self.__training_step_cache["img_1_outlier_channels_by_top_k"]

            apparent_inliers = (img_1_inlier_channels_by_max & img_2_inlier_channels_by_max).sum()
            apparent_outliers = (img_1_outlier_channels_by_max | img_2_outlier_channels_by_max).sum()

            apparent_inliers_top_k = (img_1_inlier_channels_by_top_k & img_2_inlier_channels_by_top_k).sum()
            apparent_outliers_top_k = (img_1_outlier_channels_by_top_k | img_2_outlier_channels_by_top_k).sum()

            inliers_outliers_logs = {
                "training/apparent inliers": apparent_inliers,
                "training/apparent outliers": apparent_outliers,
                "training/apparent inliers (top k)": apparent_inliers_top_k,
                "training/apparent outliers (top k)": apparent_outliers_top_k,
            }

            patch_diameter = self.network.receptive_field_diameter()

            # Generate a loss for image 2
            maxima_patches = self.image_to_patch_batch(
                img_2, img_2_kp_candidates.flatten(1),
                patch_diameter
            )
            corr_patches = torch.zeros(
                img_2_correspondences.shape[1], img_2.shape[0], patch_diameter, patch_diameter,
                dtype=maxima_patches.dtype,
                device=self._device
            )
            corr_patches[img_2_correspondences_mask, :, :, :] = self.image_to_patch_batch(
                img_2, img_2_correspondences[:, img_2_correspondences_mask], patch_diameter
            )

            maximizer_outputs: torch.Tensor = self(maxima_patches, False)
            correspondence_outputs: torch.Tensor = self(corr_patches, False)

            loss, img_2_loss_logs = self._loss.forward_with_log_data(
                maximizer_outputs, correspondence_outputs,
                img_2_inlier_channels_by_top_k, img_2_outlier_channels_by_top_k
            )

            img_2_loss_logs = {
                "training/image 2/" + key: img_2_loss_logs[key] for key in img_2_loss_logs
                if img_2_loss_logs[key] is not None
            }

            loss_logs = {**inliers_outliers_logs, **img_2_loss_logs}
            self.__training_step_cache = {}

        return {
            'loss': loss,
            'log': loss_logs
        }

    def validation_step(self, batch, batch_idx, dataloader_index):
        # set modules to test mode
        self.network.train(False)
        self._loss.train(False)

        # Run preprocess step in place
        batch[0] = torch.stack([self.preprocess(img) for img in batch[0]], dim=0)
        batch[1] = torch.stack([self.preprocess(img) for img in batch[1]], dim=0)

        # unpack data since batch size is 1
        img_1 = batch[0][0]
        img_2 = batch[1][0]
        # name = batch[2][0]
        correspondence_func = batch[3][0]

        img_1_kp_candidates, _ = self.network.extract_top_k_keypoints(img_1, self._n_top_patches)
        img_2_kp_candidates, _ = self.network.extract_top_k_keypoints(img_2, self._n_top_patches)

        num_apparent_inliers, num_true_inliers, num_inliers_by_top_k = self.count_inliers(
            correspondence_func, img_1_kp_candidates, img_2_kp_candidates,
            img_1.shape, img_2.shape, self._inlier_radius
        )

        return {
            "apparent inliers": num_apparent_inliers,
            "true inliers": num_true_inliers,
            "apparent inliers (top k)": num_inliers_by_top_k
        }

    def validation_epoch_end(self, outputs: List[List[Dict[str, torch.Tensor]]]):
        return {
            "train_eval_true_inliers": torch.stack([x["true inliers"] for x in outputs[0]]).mean(),
            "eval_true_inliers": torch.stack([x["true inliers"] for x in outputs[1]]).mean(),
            "log": {
                "training_evaluation/apparent inliers": torch.stack([x['apparent inliers'] for x in outputs[0]]).mean(),
                "training_evaluation/true inliers": torch.stack([x["true inliers"] for x in outputs[0]]).mean(),
                "training_evaluation/apparent inliers (top k)": torch.stack(
                    [x["apparent inliers (top k)"] for x in outputs[0]]).mean(),

                "evaluation/apparent inliers": torch.stack([x['apparent inliers'] for x in outputs[1]]).mean(),
                "evaluation/true inliers": torch.stack([x["true inliers"] for x in outputs[1]]).mean(),
                "evaluation/apparent inliers (top k)": torch.stack(
                    [x["apparent inliers (top k)"] for x in outputs[1]]).mean()
            }
        }

    def test_step(self, batch, batch_idx):
        # set modules to test mode
        self.network.train(False)
        self._loss.train(False)

        # Run preprocess step in place
        batch[0] = torch.stack([self.preprocess(img) for img in batch[0]], dim=0)
        batch[1] = torch.stack([self.preprocess(img) for img in batch[1]], dim=0)

        # unpack data since batch size is 1
        img_1 = batch[0][0]
        img_2 = batch[1][0]
        # name = batch[2][0]
        correspondence_func = batch[3][0]

        img_1_kp_candidates, _ = self.network.extract_top_k_keypoints(img_1, self._n_top_patches)
        img_2_kp_candidates, _ = self.network.extract_top_k_keypoints(img_2, self._n_top_patches)

        num_apparent_inliers, num_true_inliers, num_inliers_by_top_k = self.count_inliers(
            correspondence_func, img_1_kp_candidates, img_2_kp_candidates,
            img_1.shape, img_2.shape, self._inlier_radius
        )
        return {
            "apparent inliers": num_apparent_inliers,
            "true inliers": num_true_inliers,
            "apparent inliers (top k)": num_inliers_by_top_k
        }

    def test_epoch_end(self, outputs):
        return {
            "log": {
                "test/apparent inliers": torch.stack([x['apparent inliers'] for x in outputs]).mean(),
                "test/true inliers": torch.stack([x["true inliers"] for x in outputs]).mean(),
                "test/apparent inliers (top k)": torch.stack(
                    [x["apparent inliers (top k)"] for x in outputs]).mean()
            },
            "matching_scores": {
                "apparent": torch.sort(
                    torch.stack([x['apparent inliers'] for x in outputs])).values / self.hparams.channels_out,
                "true": torch.sort(
                    torch.stack([x['apparent inliers'] for x in outputs])).values / self.hparams.channels_out,
            }
        }

    @staticmethod
    def find_correspondences(correspondence_func,
                             keypoints_xy: torch.Tensor,
                             target_shape: torch.Size,
                             inverse: bool = False,
                             exclude_border_px: int = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        keypoints_np = keypoints_xy.cpu().numpy()
        packed_correspondences_xy, correspondences_indices = correspondence_func(
            keypoints_np,
            inverse=inverse
        )

        unpacked_correspondences_xy = np.zeros(keypoints_np.shape, dtype=packed_correspondences_xy.dtype)
        unpacked_correspondences_xy[:, correspondences_indices] = packed_correspondences_xy

        correspondences_mask = np.zeros(keypoints_np.shape[1], dtype=np.bool)
        correspondences_mask[correspondences_indices] = True

        correspondences_xy = torch.tensor(unpacked_correspondences_xy, device=keypoints_xy.device)
        correspondences_mask = torch.tensor(correspondences_mask, device=keypoints_xy.device)

        # remove correspondences in border area
        correspondence_in_border = (
                (correspondences_xy < exclude_border_px).sum(0).to(torch.bool) |
                ((target_shape[2] - exclude_border_px) <= correspondences_xy[0, :]) |
                ((target_shape[1] - exclude_border_px) <= correspondences_xy[1, :])
        )
        correspondences_xy[:, correspondence_in_border] = 0
        correspondences_mask[correspondence_in_border] = False
        return correspondences_xy, correspondences_mask

    @staticmethod
    def sort_candidates_and_generate_labels(kp_candidates: torch.Tensor, correspondences: torch.Tensor,
                                            correspondences_mask: torch.Tensor, inlier_radius: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # find the distance between the kp candidates and the correspondence for each heat map (cxkx1)
        kp_distances = torch.cdist(
            kp_candidates.permute(1, 2, 0),
            correspondences.unsqueeze(-1).permute(1, 2, 0)
        ).squeeze(-1)  # cxkx1 -> cxk

        # determine if highest scoring keypoints are inliers or not for reporting
        inlier_channels_by_max = (kp_distances[:, 0] < inlier_radius) & correspondences_mask
        outlier_channels_by_max = (kp_distances[:, 0] >= inlier_radius) & correspondences_mask

        # sort the keypoint candidates by their match distance to the correspondence
        candidate_distances, candidate_rankings = kp_distances.topk(
            k=kp_distances.shape[-1], largest=False, sorted=True, dim=-1
        )  # cxk
        kp_candidates = torch.gather(kp_candidates, -1, candidate_rankings.unsqueeze(0).expand(2, -1, -1))

        # determine if any of the top k scoring keypoints would qualify as an inlier if it were the maximum response
        inlier_channels_by_top_k = (candidate_distances[:, 0] < inlier_radius) & correspondences_mask
        outlier_channels_by_top_k = (candidate_distances[:, 0] >= inlier_radius) & correspondences_mask

        return (kp_candidates, inlier_channels_by_max, outlier_channels_by_max,
                inlier_channels_by_top_k, outlier_channels_by_top_k)

    @staticmethod
    def image_to_patch_batch(image: torch.Tensor, keypoints_xy: torch.Tensor, diameter: int) -> torch.Tensor:
        if diameter % 2 != 1:
            raise ValueError("diameter must be odd")
        assert len(keypoints_xy.shape) == 2 and keypoints_xy.shape[0] == 2
        radius = (diameter - 1) // 2
        keypoints_xy = keypoints_xy.to(torch.int)
        batch = torch.zeros((keypoints_xy.shape[1], image.shape[0], diameter, diameter), device=keypoints_xy.device)
        for point_idx in range(keypoints_xy.shape[1]):
            keypoint_x = keypoints_xy[0, point_idx]
            keypoint_y = keypoints_xy[1, point_idx]
            batch[point_idx, :, :, :] = image[
                                        :,
                                        keypoint_y - radius: keypoint_y + radius + 1,
                                        keypoint_x - radius: keypoint_x + radius + 1
                                        ]
        return batch

    @staticmethod
    def count_inliers(correspondence_func,
                      img_1_kp_candidates: torch.Tensor, img_2_kp_candidates: torch.Tensor,
                      img_1_shape: torch.Size, img_2_shape: torch.Size,
                      inlier_radius: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_1_correspondences, img_1_correspondences_mask = IMIPLightning.find_correspondences(
            correspondence_func, img_2_kp_candidates[:, :, 0], img_1_shape, inverse=True,
        )  # 2 x c

        img_2_correspondences, img_2_correspondences_mask = IMIPLightning.find_correspondences(
            correspondence_func, img_1_kp_candidates[:, :, 0], img_2_shape, inverse=False,
        )

        (img_1_kp_candidates, img_1_inlier_channels_by_max, img_1_outlier_channels_by_max,
         img_1_inlier_channels_by_top_k, img_1_outlier_channels_by_top_k) = \
            IMIPLightning.sort_candidates_and_generate_labels(
                img_1_kp_candidates, img_1_correspondences, img_1_correspondences_mask, inlier_radius
            )

        (img_2_kp_candidates, img_2_inlier_channels_by_max, img_2_outlier_channels_by_max,
         img_2_inlier_channels_by_top_k, img_2_outlier_channels_by_top_k) = \
            IMIPLightning.sort_candidates_and_generate_labels(
                img_2_kp_candidates, img_2_correspondences, img_2_correspondences_mask, inlier_radius
            )

        apparent_inliers_by_max = img_1_inlier_channels_by_max & img_2_inlier_channels_by_max
        num_apparent_inliers_by_max = apparent_inliers_by_max.sum().to(torch.float32)

        num_apparent_inliers_by_top_k = (
                img_1_inlier_channels_by_top_k & img_2_inlier_channels_by_top_k
        ).sum().to(torch.float32)

        apparent_inlier_img_1_keypoints_xy = img_1_kp_candidates[:, apparent_inliers_by_max, 0]
        num_true_inliers_by_max = torch.zeros([1], device=img_1_kp_candidates.device)
        if num_apparent_inliers_by_max > 0:
            max_inlier_distance = torch.tensor([inlier_radius], device=img_1_kp_candidates.device)
            unique_inlier_img_1_keypoints_xy = apparent_inlier_img_1_keypoints_xy[:, 0:1]
            num_true_inliers_by_max += 1  # on Apr 22nd 2020, an off by one error was discovered +1, to old results
            for i in range(1, int(num_apparent_inliers_by_max)):
                test_inlier = apparent_inlier_img_1_keypoints_xy[:, i:i + 1]
                if (torch.norm(unique_inlier_img_1_keypoints_xy - test_inlier, p=2, dim=0) > max_inlier_distance).all():
                    unique_inlier_img_1_keypoints_xy = torch.cat((unique_inlier_img_1_keypoints_xy, test_inlier), dim=1)
                    num_true_inliers_by_max += 1

        return num_apparent_inliers_by_max, num_true_inliers_by_max, num_apparent_inliers_by_top_k
