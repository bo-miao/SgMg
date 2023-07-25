"""
Segmentaion Part 
Modified from DETR (https://github.com/facebookresearch/detr)
"""
from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass

import fvcore.nn.weight_init as weight_init

from .position_encoding import PositionEmbeddingSine1D

BN_MOMENTUM = 0.1

def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, visual, text,
                text_key_padding_mask: Optional[Tensor] = None,
                text_pos: Optional[Tensor] = None,
                visual_pos: Optional[Tensor] = None):
        visual = rearrange(visual, 't h w b c -> (t h w) b c')
        visual2 = self.multihead_attn(query=self.with_pos_embed(visual, visual_pos),
                                   key=self.with_pos_embed(text, text_pos),
                                   value=text, attn_mask=None,
                                   key_padding_mask=text_key_padding_mask)[0]
        visual = visual * visual2
        return visual


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


