import torch.autograd

import epipointnet.imips_pytorch.models.imips
import epipointnet.irls.models


class EpiPointNet(torch.nn.Module):
    def __init__(self,
                 imipNet: epipointnet.imips_pytorch.models.imips.ImipNet,
                 side_info_net: torch.nn.Module,
                 normalized_eight_point_net: epipointnet.irls.models.NormalizedEightPointNet
                 ):
        super(EpiPointNet, self).__init__()
        self.imipNet = imipNet
        self.argmax = None
        self.side_info_net = side_info_net
        self.normalized_eight_point_net = normalized_eight_point_net

    def forward(self,
                image_1_keypoints: torch.Tensor, images_1: torch.Tensor,
                image_2_keypoints: torch.Tensor, images_2: torch.Tensor) -> torch.Tensor:
        # images_1_keypoints: bx2xc
        # images_1: bxcxhxw
        # images_2_keypoints: bx2xc
        # images_2: bxcxhxw

        imip_out_img_1 = self.imipNet(images_1, keepDim=True)
        imip_out_img_2 = self.imipNet(images_2, keepDim=True)
