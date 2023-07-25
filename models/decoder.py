
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from util.misc import inverse_sigmoid
from einops import rearrange


class MSO(nn.Module):
    def __init__(self, mask_dim=16, img_dim=[96, 192], out_dim=16):
        super().__init__()

        self.mask_dim = mask_dim
        self.img_dim = img_dim
        self.out_dim = out_dim

        self.conv1_1div8 = nn.Conv2d(mask_dim+img_dim[1], mask_dim, kernel_size=3, padding=1)
        self.conv2_1div8 = nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1)

        self.conv1_1div4 = nn.Conv2d(mask_dim + img_dim[0], mask_dim, kernel_size=3, padding=1)
        self.conv2_1div4 = nn.Conv2d(mask_dim, mask_dim, kernel_size=3, padding=1)

    # TODO: add image on channel.  deconv to upsample
    def forward(self, pred_masks, image_features):
        image_features = [x.tensors for x in image_features]  # 1/4 & 1/8

        # merge with 1/8 image
        assert pred_masks.shape[-1] == image_features[-1].shape[-1], "First size wrong."
        x = torch.cat([pred_masks, image_features[-1]], dim=1)
        pred_masks += self.conv2_1div8(F.relu(self.conv1_1div8(F.relu(x))))

        # merge with 1/4 image
        pred_masks = F.interpolate(pred_masks, size=(image_features[-2].shape[-2], image_features[-2].shape[-1]), mode='bilinear', align_corners=False)
        assert pred_masks.shape[-1] == image_features[-2].shape[-1], "Second size wrong."
        x = torch.cat([pred_masks, image_features[-2]], dim=1)
        pred_masks += self.conv2_1div4(F.relu(self.conv1_1div4(F.relu(x))))

        return pred_masks

