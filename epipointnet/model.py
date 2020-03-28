from typing import Tuple, Optional

import torch.autograd

from epipointnet.bilevel.conv import Sobel2DArgMax
from epipointnet.dfe.models import NormalizedEightPointNet
from epipointnet.imips_pytorch.models.imips import ImipNet


class PatchSampler(torch.nn.Module):
    def __init__(self):
        super(PatchSampler, self).__init__()

    def forward(self, bchw: torch.Tensor, patch_centers: torch.Tensor, patch_diameter: int) -> torch.Tensor:
        flow_field = torch.zeros(
            bchw.shape[0],
            bchw.shape[1],
            patch_diameter,
            patch_diameter,
            2,
            dtype=bchw.dtype, device=bchw.device
        )
        flow_field_step_x = 2.0 / bchw.shape[3]
        flow_field_step_y = 2.0 / bchw.shape[2]
        patch_radius = patch_diameter // 2

        for b in bchw.shape[0]:
            for c in bchw.shape[1]:
                left = -1 + (patch_centers[b, 0, c] - patch_radius) * flow_field_step_x
                right = -1 + (patch_centers[b, 0, c] + patch_radius) * flow_field_step_x
                top = -1 + (patch_centers[b, 1, c] - patch_radius) * flow_field_step_y
                bottom = -1 + (patch_centers[b, 1, c] + patch_radius) * flow_field_step_y

                x_grid, y_grid = torch.meshgrid([
                    torch.arange(left, right, flow_field_step_x,
                                 dtype=bchw.dtype, device=bchw.device),
                    torch.arange(top, bottom, flow_field_step_y,
                                 dtype=bchw.dtype, device=bchw.device),
                ])
                flow_field[b, c, :, :, :] = torch.stack((x_grid, y_grid), dim=-1)

        return torch.nn.functional.grid_sample(bchw, flow_field, mode="nearest")


class SideInfoNet(torch.nn.Module):
    def __init__(self, input_channels: int = 128, side_info_size: int = 4, num_convolutions: int = 3):
        super(SideInfoNet, self).__init__()

        # Inspired by depth-wise separable convolutions, we first process the spatial information
        # in the results and then process the filter population information.

        # Upsample the input data for the desired number of spatial convolutions and run the convolutions
        # against each channel separately
        spatial_layers = [
            torch.nn.Upsample(size=(num_convolutions * 2 + 1, num_convolutions * 2 + 1), mode='bilinear'),
        ]
        for i in range(num_convolutions):
            spatial_layers.extend([
                torch.nn.Conv2d(2 * input_channels, 2 * input_channels, kernel_size=3, groups=2 * input_channels),
                torch.nn.LeakyReLU(negative_slope=0.02)
            ])

        # Use DFE equivariant network to process channel information
        # TODO: experiment training with track_running_stats=True on instancenorm
        channel_layers = [
            torch.nn.Conv1d(2 * input_channels, 256, kernel_size=1),
            torch.nn.InstanceNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, 512, kernel_size=1),
            torch.nn.InstanceNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(512, 1024, kernel_size=1),
            torch.nn.InstanceNorm1d(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(1024, 512, kernel_size=1),
            torch.nn.InstanceNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(512, 256, kernel_size=1),
            torch.nn.InstanceNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(256, side_info_size, kernel_size=1)
        ]

        self.spatial_layers = torch.nn.Sequential(*spatial_layers)
        self.channel_layers = torch.nn.Sequential(*channel_layers)

    def forward(self, img_1_imip_outs: torch.Tensor, img_2_imip_outs: torch.Tensor):
        # Interleave the IMIP outputs along the correspondence dimension
        # img_1_imip_outs size: bxcxhxw
        # img_2_imip_outs size: bxcxhxw
        bchw = torch.stack((img_1_imip_outs, img_2_imip_outs), dim=2).view(
            img_1_imip_outs.shape[0], 2 * img_1_imip_outs.shape[1], img_1_imip_outs.shape[2], img_1_imip_outs.shape[3]
        )

        # bx2cxhxw -> bx2cx1x1
        spatially_reduced_data = self.spatial_layers(bchw)

        # bx2xcx1x1 -> bx2c -> 2cxb -> 1x2cxb
        spatially_reduced_data = spatially_reduced_data.squeeze().permute(1, 0).unsqueeze(0)

        # 1x2cxb -> 1xkxb -> 1xbxk
        return self.channel_layers(spatially_reduced_data).permute(0, 2, 1)


class PatchBatchEpiPointNet(torch.nn.Module):
    def __init__(self,
                 imipNet: ImipNet,
                 irls_depth: int = 3,
                 side_info_size: int = 4,
                 ):
        super(PatchBatchEpiPointNet, self).__init__()
        self.imip_net = imipNet
        self.argmax = Sobel2DArgMax.apply
        self.normalized_eight_point_net = NormalizedEightPointNet(irls_depth, side_info_size)
        self.patch_sampler = PatchSampler()
        self._f32_eps = float.fromhex('1p-23')
        if side_info_size != 0:
            self.side_info_net = SideInfoNet(self.imip_net.output_channels(),
                                             self.normalized_eight_point_net.side_info_size)
        else:
            self.side_info_net = None

    def patch_size(self):
        """
        :return: The diameter of the patches needed to train PatchBatchEpiPointNet
        """
        return self.imip_net.receptive_field_diameter() + Sobel2DArgMax.neighborhood_size - 1

    def _forward_train(self,
                       images_1: torch.Tensor,
                       images_2: torch.Tensor,
                       image_1_patch_coords: torch.Tensor,
                       image_2_patch_coords: torch.Tensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor]:
        # images_1: bxcxhxw
        # images_2: bxcxhxw
        # image_1_patch_coords: 2xb
        # image_2_patch_coords: 2xb
        # corr_images_1: bxcxhxw
        # corr_images_2: bxcxhxw

        # The input to PatchBatchEpiPointNet._forward_train must be
        # (images_1, images_2, image_1_patch_coords, image_2_patch_coords) or
        # (images_1, images_2, image_1_patch_coords, image_2_patch_coords, corr_images_1, corr_images_2)
        # where images_1, images_2, corr_images_1, and corr_images_2 are IMIP image patches shaped as bxcxhxw.
        # image_1_patch_coords and image_2_patch_coords should have the shape 2xb and contain the x, y positions
        # of the anchor patch centers from each image.
        assert (
                images_1.shape[0] == images_2.shape[0] == image_1_patch_coords.shape[1] ==
                image_2_patch_coords.shape[1] and
                images_1.shape[0] > 1
        )

        # Run the patches through the IMIP network to produce correspondence heat maps.
        # This returns each filter channel's response to each other filter's argmax.
        image_1_imip_outs = self.imip_net(images_1, keepDim=False)  # shape: bxcxhxw
        image_2_imip_outs = self.imip_net(images_2, keepDim=False)  # shape: bxcxhxw

        # Find each filter channel's argmax in each image in order to obtain keypoints.
        # We are only interested in each filter channel's response
        # to their own argmax.
        aligned_image_1_imip_outs = image_1_imip_outs[
                                    range(image_1_imip_outs.shape[0]),
                                    range(image_1_imip_outs.shape[1]), :, :].unsqueeze(0)  # shape: 1xbxhxw

        aligned_image_2_imip_outs = image_2_imip_outs[
                                    range(image_2_imip_outs.shape[0]),
                                    range(image_2_imip_outs.shape[1]), :, :].unsqueeze(0)  # shape: 1xbxhxw

        # Bias the center pixel ever so slightly so as to break ties
        imip_out_1_patch_center = ((torch.tensor(image_1_imip_outs.shape[2:], dtype=image_1_imip_outs.dtype,
                                                 device=image_1_imip_outs.device) - 1) / 2).long()
        imip_out_2_patch_center = ((torch.tensor(image_2_imip_outs.shape[2:], dtype=image_2_imip_outs.dtype,
                                                 device=image_2_imip_outs.device) - 1) / 2).long()

        aligned_image_1_imip_outs[:, :, imip_out_1_patch_center[1], imip_out_1_patch_center[0]] += self._f32_eps
        aligned_image_2_imip_outs[:, :, imip_out_2_patch_center[1], imip_out_2_patch_center[0]] += self._f32_eps

        # The argmax should always be the center of the patch
        image_1_keypoints = self.argmax(aligned_image_1_imip_outs).squeeze(0)  # shape: 1xbxhxw -> 1x2xb -> 2xb
        image_2_keypoints = self.argmax(aligned_image_2_imip_outs).squeeze(0)  # shape: 1xbxhxw -> 1x2xb -> 2xb
        # NOTE: There is an edge case where the keypoint might not be in the center of the patch.
        # This happens because we select the patches within a border about the image
        # such that we don't have any missing data when we run the argmax.
        # This means there must be a neighborhood of data bout the selected keypoint.
        # There may be a better argmax in this neighborhood that wasn't selected
        # since it was too close to the image border. In this case, the returned
        # keypoint is used for forward calculations, but gradients are truncated for the
        # point in the argmax layer since there is not a valid neighborhood to generate
        # derivatives.

        # We need to add the patch coordinates back into the argmaxes.
        # The patches should be centered on the argmax.
        image_1_keypoints -= imip_out_1_patch_center.reshape(2, 1).flip((0,))
        image_1_keypoints += image_1_patch_coords
        image_2_keypoints -= imip_out_2_patch_center.reshape(2, 1).flip((0,))
        image_2_keypoints += image_2_patch_coords

        point_info = torch.cat((image_1_keypoints, image_2_keypoints), dim=0).permute(1, 0) \
            .unsqueeze(0)  # shape: bx4 -> 1xbx4

        if self.side_info_net is not None:
            # The side info net needs to be supplied with the imip outputs. It processes every channel's response
            # to each keypoint
            side_info = self.side_info_net(image_1_imip_outs, image_1_imip_outs)  # shape: 1xbxk
        else:
            side_info = torch.zeros((1, images_1.shape[0], 0), device=point_info.device, dtype=point_info.dtype)

        norm_f_mats, image_1_rescalings, image_2_rescalings, weights = self.normalized_eight_point_net(
            point_info,
            side_info
        )  # norm_f_mats: List[torch.Tensor] where each tensor has shape 1x3x3.
        # image_1_rescalings, image_2_rescalings size: 1x3x3

        for i in range(len(norm_f_mats)):
            norm_f_mats[i] = image_1_rescalings.permute(0, 2, 1).bmm(norm_f_mats[i].bmm(image_2_rescalings))

        return (norm_f_mats, image_1_keypoints, image_2_keypoints, weights,
                image_1_imip_outs, image_2_imip_outs)

    def _forward_test(self,
                      images_1: torch.Tensor,
                      images_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor, torch.Tensor]:
        # images_1: bxcxhxw
        # images_2: bxcxhxw

        # Run the images through the IMIP network to produce correspondence heat maps.
        image_1_imip_outs = self.imip_net(images_1, keepDim=True)  # shape: bxcxhxw
        image_2_imip_outs = self.imip_net(images_2, keepDim=True)  # shape: bxcxhxw

        image_1_keypoints = self.argmax(image_1_imip_outs)  # shape: bx2xc
        image_2_keypoints = self.argmax(image_1_imip_outs)  # shape: bx2xc

        # The side info net needs to be supplied with IMIP net output within a neighborhood about each
        # keypoint. Extract the patches around the detected keypoints to generate side info
        # TODO: Test the patch sampler
        side_info_image_1_imip_outs = self.patch_sampler(
            image_1_imip_outs, image_1_keypoints,
            Sobel2DArgMax.neighborhood_size
        )  # shape: bxcxhxw

        side_info_image_2_imip_outs = self.patch_sampler(
            image_1_imip_outs, image_2_keypoints,
            Sobel2DArgMax.neighborhood_size
        )  # shape: bxcxhxw

        side_info = self.side_info_net(side_info_image_1_imip_outs, side_info_image_2_imip_outs)  # shape: bxcxk

        point_info = torch.cat((image_1_keypoints, image_2_keypoints), dim=1).permute(0, 2, 1)  # shape: bxcx4

        norm_f_mats, image_1_rescalings, image_2_rescalings, weights = self.normalized_eight_point_net(
            point_info,
            side_info
        )  # norm_f_mats: List[torch.Tensor] where each tensor has shape bx3x3.
        # image_1_rescalings, image_2_rescalings size: bx3x3

        for i in range(len(norm_f_mats)):
            norm_f_mats[i] = image_1_rescalings.permute(0, 2, 1).bmm(norm_f_mats[i].bmm(image_2_rescalings))

        return (norm_f_mats, image_1_keypoints, image_2_keypoints, weights,
                image_1_imip_outs, image_2_imip_outs)

    def forward(self,
                images_1: torch.Tensor,
                images_2: torch.Tensor,
                image_1_patch_coords: Optional[torch.Tensor],
                image_2_patch_coords: Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            return self._forward_train(images_1, images_2,
                                       image_1_patch_coords, image_2_patch_coords)
        else:
            return self._forward_test(images_1, images_2)


"""
Uses too much ram. imipet simipconv requires 4gb per KITTI image
class ImageBatchEpiPointNet(torch.nn.Module):
    def __init__(self,
                 imipNet: ImipNet,
                 normalized_eight_point_net: NormalizedEightPointNet,
                 use_patches_for_side_info: Optional[bool] = False,
                 ):
        super(ImageBatchEpiPointNet, self).__init__()
        self.imip_net = imipNet
        self.argmax = Sobel2DArgMax.apply
        self.normalized_eight_point_net = normalized_eight_point_net
        self.side_info_net = SideInfoNet(self.imip_net.output_channels(),
                                         self.normalized_eight_point_net.side_info_size)
        self._use_patches_for_side_info = use_patches_for_side_info

        if self._use_patches_for_side_info:
            self.patch_sampler = PatchSampler()

    def forward(self,
                images_1: torch.Tensor,
                images_2: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # images_1: bxcxhxw
        # images_2: bxcxhxw

        # Run the patches/ the images through the IMIP network to produce correspondence heat maps.
        image_1_imip_outs = self.imip_net(images_1, keepDim=True)  # shape: bxcxhxw
        image_2_imip_outs = self.imip_net(images_2, keepDim=True)  # shape: bxcxhxw

        # Find each filter channel's argmax in each image in order to obtain keypoints.
        image_1_keypoints = self.argmax(image_1_imip_outs)  # shape: bx2xc
        image_2_keypoints = self.argmax(image_2_imip_outs)  # shape: bx2xc

        if self._use_patches_for_side_info:
            # TODO: Test the patch sampler
            # Extract the patches around the detected keypoints to generate side info
            side_info_image_1_imip_outs = self.patch_sampler(
                image_1_imip_outs, image_1_keypoints,
                self.imip_net.receptive_field_diameter()
            )  # shape: bxcxhxw where hxw is the imip net receptive field

            side_info_image_2_imip_outs = self.patch_sampler(
                image_1_imip_outs, image_2_keypoints,
                self.imip_net.receptive_field_diameter()
            )  # shape: bxcxhxw where hxw is the imip net receptive field
        else:
            side_info_image_1_imip_outs = image_1_imip_outs
            side_info_image_2_imip_outs = image_2_keypoints

        side_info = self.side_info_net(side_info_image_1_imip_outs, side_info_image_2_imip_outs)  # shape: bxcxk

        point_info = torch.cat((image_1_keypoints, image_2_keypoints), dim=1).permute(0, 2, 1)  # shape: bxcx4

        norm_f_mats, image_1_rescalings, image_2_rescalings, weights = self.normalized_eight_point_net(
            point_info,
            side_info
        )  # norm_f_mats: List[torch.Tensor] where each tensor has shape bx3x3.
        # image_1_rescalings, image_2_rescalings size: bx3x3

        for i in range(len(norm_f_mats)):
            norm_f_mats[i] = image_1_rescalings.permute(0, 2, 1).bmm(norm_f_mats[i].bmm(image_2_rescalings))

        return norm_f_mats, image_1_keypoints, image_2_keypoints, weights
"""
