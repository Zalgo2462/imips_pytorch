from typing import Optional

import kornia
import torch

from .preprocess import PreprocessModule
from .pyramid import ScalePyramid


class PreprocessHessian(PreprocessModule):
    def __init__(self):
        super(PreprocessModule, self).__init__()
        self.hessian = MaxScaleHessian(
            max_octaves=3,
            scales_per_octave=3,
        )

    def preprocess(self, image: torch.Tensor):
        # normalize image and create hessian blobs
        image = (image / 127.5) - 1.0  # scale to [-1, 1]

        # convert to grayscale
        image_gray = kornia.color.rgb_to_grayscale(image) if image.shape[0] == 3 else image

        # contrast stretch image for hessian http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
        image_gray_1p_idx = 1 + round(.02 * (image_gray.numel() - 1))
        image_gray_99p_idx = 1 + round(.98 * (image_gray.numel() - 1))
        image_gray_1p_val = image_gray.view(-1).kthvalue(image_gray_1p_idx).values
        image_gray_99p_val = image_gray.view(-1).kthvalue(image_gray_99p_idx).values
        image_gray = (image_gray - image_gray_1p_val) * (2.0 / (image_gray_99p_val - image_gray_1p_val)) - 1
        image_gray.clamp_(-1, 1)
        image_hessian = self.hessian(image_gray.unsqueeze(0))[0]
        # the old hessian module used to histogram stretch the resulting hessian values to [0, 255]
        # we'll stretch it to [-1, 1]
        image_hessian = (image_hessian / image_hessian.max()) * 2.0 - 1.0

        return torch.cat((image, image_hessian), dim=0)

    def output_channels(self, input_channels: int) -> int:
        return input_channels + 1


class MaxScaleHessian(torch.nn.Module):

    def __init__(self, max_octaves: Optional[int] = None,
                 scales_per_octave: Optional[int] = 3,
                 ):
        super(MaxScaleHessian, self).__init__()

        self.scale_pyramid = ScalePyramid(
            max_octaves=max_octaves,
            n_levels=scales_per_octave,
            init_sigma=1.6,
            double_image=False
        )
        self.hessian_module = kornia.feature.BlobHessian()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pyramid, sigmas, _ = self.scale_pyramid(images)
        max_scale_hessian = torch.zeros_like(images)
        for oct_idx, octave in enumerate(pyramid):
            B, L, CH, H, W = octave.size()  # L dim is the levels per octave
            hessian_response = self.hessian_module(octave.view(B * L, CH, H, W), sigmas[oct_idx].view(-1))
            hessian_response = torch.max(hessian_response, -hessian_response)

            hessian_response, _ = torch.nn.functional.interpolate(
                hessian_response,
                size=(images.shape[2], images.shape[3]),
                mode='bilinear', align_corners=False
            ).view(B, L, CH, images.shape[2], images.shape[3]).max(dim=1)
            max_scale_hessian = torch.max(max_scale_hessian, hessian_response)
            pyramid[oct_idx] = None  # drop the reference to the scaled image for GC
            sigmas[oct_idx] = None

        return max_scale_hessian
