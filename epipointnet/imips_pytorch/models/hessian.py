from typing import Optional

import kornia
import torch

from . import pyramid


class MaxScaleHessian(torch.nn.Module):

    def __init__(self, max_octaves: Optional[int] = None,
                 scales_per_octave: Optional[int] = 3,
                 ):
        super(MaxScaleHessian, self).__init__()

        self.scale_pyramid = pyramid.ScalePyramid(
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
