"""
SgMg model class.
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import time

import torch
import torch.nn.functional as F
from torch import nn

import os
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .position_encoding import PositionEmbeddingSine1D
from .backbone import build_backbone
from .deformable_transformer import build_deforamble_transformer
from .segmentation import VisionLanguageFusionModule
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocessors import build_postprocessors
from .decoder import MSO
from .modules import LFMResizeAdaptive
from .text_encoder.text_encoder import TextEncoder, FeatureResizer
import copy
from einops import rearrange, repeat
import warnings
warnings.filterwarnings("ignore")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)

class SgMg(nn.Module):
    """ This is the SgMg module that performs referring video object detection """
    def __init__(self, args, backbone, transformer, num_classes, num_queries, num_feature_levels,
                    num_frames, mask_dim, dim_feedforward,
                    controller_layers, dynamic_mask_channels,
                    aux_loss=False, with_box_refine=False, two_stage=False,
                    freeze_text_encoder=False, rel_coord=True, matcher=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         SgMg can detect in a video. For ytvos, we recommend 5 queries for each frame.
            num_frames:  number of clip frames
            mask_dim: dynamic conv inter layer channel number.
            dim_feedforward: vision-language fusion module ffn channel number.
            dynamic_mask_channels: the mask feature output channel number.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        self.matcher = matcher
        self.num_frames = num_frames
        self.num_feature_levels = num_feature_levels
        self.training = not args.eval
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        assert two_stage == False, "args.two_stage must be false!"
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, self.num_classes)  # 256->1
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.mask_dim = mask_dim
        self.controller_layers = controller_layers
        self.dynamic_mask_channels = dynamic_mask_channels
        self.backbone = backbone

        # Build Text Encoder
        self.text_encoder = TextEncoder(args)
        self.text_pos = PositionEmbeddingSine1D(hidden_dim, normalize=True)
        self.text_proj = FeatureResizer(
            input_feat_size=self.text_encoder.feat_dim,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        self.sentence_proj = FeatureResizer(
            input_feat_size=self.text_encoder.feat_dim,
            output_feat_size=hidden_dim,
            dropout=0.1,
        )
        self.fusion_module = VisionLanguageFusionModule(d_model=hidden_dim, nhead=8)

        # Build SCF
        # convert 8x 16x 32x 64x to 256 dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides[-3:])
            input_proj_list = []
            input_fft_list = []
            input_fft_post_list = []

            for idx_ in range(num_backbone_outs):
                in_channels = backbone.num_channels[-3:][idx_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                input_fft_list.append(LFMResizeAdaptive(hidden_dim, 7))
                input_fft_post_list.append(LFMResizeAdaptive(hidden_dim, 7))

            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
                input_fft_list.append(LFMResizeAdaptive(hidden_dim, 7))
                input_fft_post_list.append(LFMResizeAdaptive(hidden_dim, 7))

            self.input_proj = nn.ModuleList(input_proj_list)
            self.input_fft = nn.ModuleList(input_fft_list)
            self.input_fft_post = nn.ModuleList(input_fft_post_list)
        else:
            raise NotImplementedError

        # Build MSO
        self.mask_refine = MSO(mask_dim=self.dynamic_mask_channels, img_dim=backbone.num_channels[:2], out_dim=self.dynamic_mask_channels)

        self.rel_coord = rel_coord
        # Parameter initialization
        self.init_aux_head()
        # Build CPK
        self.build_controller()

    def init_aux_head(self):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if isinstance(self.input_proj, nn.ModuleList):
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
        else:
            nn.init.xavier_uniform_(self.input_proj[0].weight, gain=1)
            nn.init.constant_(self.input_proj[0].bias, 0)

        # multi-scale middle layers prediction
        num_pred = self.transformer.decoder.num_layers
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def build_controller(self):
        self.controller_layers = self.controller_layers
        self.in_channels = self.mask_dim
        self.dynamic_mask_channels = self.dynamic_mask_channels
        self.mask_out_stride = 4
        self.mask_feat_stride = 8
        # compute parameter number
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = MLP(self.hidden_dim, self.hidden_dim, self.num_gen_params, 3)
        for layer in self.controller.layers:
            nn.init.zeros_(layer.bias)
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, samples: NestedTensor, captions, targets):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels
               - captions: list[str]
               - targets:  list[dict]

            It returns a dict with the following elements:
               - "pred_masks": Shape = [batch_size x num_queries x out_h x out_w]

               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # ***** Visual and Texual Encoding *****
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples, 1 if self.training else 16)

        features, visual_pos = self.backbone(samples)
        b = len(captions)
        t = visual_pos[0].shape[0] // b

        # For A2D-Sentences and JHMDB-Sentencs dataset, only one frame is annotated for a clip
        if 'valid_indices' in targets[0]:
            valid_indices = torch.tensor([i * t + target['valid_indices'] for i, target in enumerate(targets)]).to(visual_pos[0].device)
            for feature in features:
                feature.tensors = feature.tensors.index_select(0, valid_indices)
                feature.mask = feature.mask.index_select(0, valid_indices)
            for i, p in enumerate(visual_pos):
                visual_pos[i] = p.index_select(0, valid_indices)
            samples.mask = samples.mask.index_select(0, valid_indices)
            t = 1

        text_features, text_sentence_features = self.forward_text(captions, device=visual_pos[0].device)
        text_pos = self.text_pos(text_features).permute(2, 0, 1)  # [length, batch_size, c]
        text_word_features, text_word_masks = text_features.decompose()
        text_word_features = text_word_features.permute(1, 0, 2)  # [length, batch_size, c]

        # ***** Spectrum-guided Cross-modal Fusion *****
        srcs = []
        masks = []
        poses = []
        multi_scale_level = 3
        high_filter = None
        for l, (feat, pos_l) in enumerate(zip(features[-multi_scale_level:], visual_pos[-multi_scale_level:])):
            src, mask = feat.decompose()
            src_proj_l = self.input_proj[l](src)
            n, c, h, w = src_proj_l.shape

            src_proj_l, high_filter = self.input_fft[l](src_proj_l, high_filter)
            src_proj_l = rearrange(src_proj_l, '(b t) c h w -> t h w b c', b=b, t=t)
            src_proj_l = self.fusion_module(visual=src_proj_l,
                                            text=text_word_features,
                                            text_key_padding_mask=text_word_masks,
                                            text_pos=text_pos,
                                            visual_pos=None
                                            )
            src_proj_l = rearrange(src_proj_l, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
            src_proj_l, high_filter = self.input_fft_post[l](src_proj_l, high_filter)

            srcs.append(src_proj_l)
            masks.append(mask)
            poses.append(pos_l)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = self.num_feature_levels - 1
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                n, c, h, w = src.shape

                src, high_filter = self.input_fft[l](src, high_filter)
                src = rearrange(src, '(b t) c h w -> t h w b c', b=b, t=t)
                src = self.fusion_module(visual=src,
                                         text=text_word_features,
                                         text_key_padding_mask=text_word_masks,
                                         text_pos=text_pos,
                                         visual_pos=None
                                         )
                src = rearrange(src, '(t h w) b c -> (b t) c h w', t=t, h=h, w=w)
                src, high_filter = self.input_fft_post[l](src, high_filter)

                srcs.append(src)
                masks.append(mask)
                poses.append(pos_l)

        # ***** Deformable Transformer *****
        out = {}
        query_embeds = self.query_embed.weight
        text_embed = repeat(text_sentence_features, 'b c -> b t q c', t=t, q=self.num_queries)
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, inter_samples = \
                                            self.transformer(srcs, text_embed, masks, poses, query_embeds)
        # srcs: [8x,16x,32x,64x]. memory: [8x,16x,32x].  features: [4x,8x,16x,32x]
        # hs: [l, batch_size*time, num_queries_per_frame, c]
        # memory: list[Tensor], shape of tensor is [batch_size*time, c, hi, wi]
        # init_reference: [batch_size*time, num_queries_per_frame, 2]
        # inter_references: [l, batch_size*time, num_queries_per_frame, 4]

        # ***** Prediction *****
        # Bbox and Score
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # cxcywh, range in [0,1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = rearrange(outputs_class, 'l (b t) q k -> l b t q k', b=b, t=t)
        outputs_coord = rearrange(outputs_coord, 'l (b t) q n -> l b t q n', b=b, t=t)
        out['pred_logits'] = outputs_class[-1]
        out['pred_boxes'] = outputs_coord[-1]

        # Segmentation
        tar_h, tar_w = memory[0].shape[-2:]
        memory_fusion = sum([F.interpolate(x, size=(tar_h, tar_w), mode="bicubic", align_corners=False) for x in memory])  # bt, c, h, w
        mask_features = rearrange(memory_fusion, '(b t) c h w -> b t c h w', b=b, t=t)
        outputs_seg_masks = []
        outputs_seg_masks_formatcher = []
        for lvl in range(hs.shape[0]):
            dynamic_mask_head_params = self.controller(hs[lvl])   # [batch_size*time, num_queries_per_frame, num_params]
            dynamic_mask_head_params = rearrange(dynamic_mask_head_params, '(b t) q n -> b (t q) n', b=b, t=t)
            lvl_references = inter_references[lvl, ..., :2]
            lvl_references = rearrange(lvl_references, '(b t) q n -> b (t q) n', b=b, t=t)

            outputs_seg_mask = self.dynamic_mask_with_coords(mask_features, dynamic_mask_head_params, lvl_references, targets)
            outputs_seg_masks.append(outputs_seg_mask)
            # use pixel_shuffle to convert [h1,w1,p^2] to [h1p,w1p]
            outputs_seg_masks_formatcher.append(rearrange(F.pixel_shuffle(outputs_seg_mask.flatten(0,1), 4).squeeze(1), '(b t q) h w -> b t q h w', b=b,t=t,q=self.num_queries))

        if self.training:
            # *** Perform HUNGARIAN matching before decoding ***
            with torch.no_grad():
                out['pred_masks'] = outputs_seg_masks_formatcher[-1]
                indices = self.matcher(out, targets)
                out["main_matcher_index"] = indices
                select_idxs = _get_src_permutation_idx(indices)

                if self.aux_loss:
                    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_seg_masks_formatcher)
                    aux_indices = []
                    for i, aux_outputs in enumerate(out['aux_outputs']):
                        aux_indices.append(self.matcher(aux_outputs, targets))
                    out["aux_matcher_index"] = aux_indices
                    aux_select_idxs = [_get_src_permutation_idx(x) for x in aux_indices]

            if self.aux_loss:
                select_idxs = aux_select_idxs + [select_idxs]
                pred_masks = [rearrange(x, 'b (t q) c h w -> b q t c h w', t=t)[y] for x, y in zip(outputs_seg_masks, select_idxs)]  # b t c h w
                pred_masks = [x.flatten(0, 1) for x in pred_masks]
                pred_masks = [self.mask_refine(x, features[:2]) for x in pred_masks]
                pred_masks = [F.pixel_shuffle(x, 4) for x in pred_masks]
                outputs_seg_masks = [rearrange(x.squeeze(1), '(b t) h w -> b t h w', b=b, t=t) for x in pred_masks]
                # pred_masks is after MSO (final mask) and pred_masks_low is before MSO (patch mask).
                out['pred_masks'] = outputs_seg_masks[-1]
                outputs_seg_masks_formatcher = [x.transpose(1, 2)[y] for x, y in zip(outputs_seg_masks_formatcher, select_idxs)]  # b t h w
                out['pred_masks_low'] = outputs_seg_masks_formatcher[-1]
                out['aux_outputs'] = self._set_aux_loss_comprehensive(outputs_class, outputs_coord, outputs_seg_masks, outputs_seg_masks_formatcher)

        elif self.args.dataset_file != "a2d" and self.args.dataset_file != "jhmdb" and 'refcoco' not in self.args.dataset_file:
            # Support DVS and YTVOS
            pred_masks = rearrange(outputs_seg_masks[-1], 'b (t q) c h w -> b q t c h w', t=t)
            pred_logits_ = []
            pred_boxes_ = []
            pred_inter_references_ = []
            pred_masks_ = []
            for idx_ in range(out["pred_logits"].shape[0]):
                # select the best result (query)
                pred_scores = out["pred_logits"][idx_].sigmoid()
                pred_scores = pred_scores.mean(0)
                max_scores, _ = pred_scores.max(-1)
                _, max_ind = max_scores.max(-1)

                pred_logit_ = out["pred_logits"][idx_,:,max_ind:max_ind+1]
                pred_box_ = out["pred_boxes"][idx_,:,max_ind:max_ind+1]
                pred_inter_reference_ = inter_references[-2, (idx_*t):((idx_+1)*t), max_ind:max_ind + 1, :2]
                pred_mask_ = pred_masks[idx_, max_ind]

                pred_logits_.append(pred_logit_)
                pred_boxes_.append(pred_box_)
                pred_inter_references_.append(pred_inter_reference_)
                pred_masks_.append(pred_mask_)

            out["pred_logits"] = torch.stack(pred_logits_, dim=0)
            out["pred_boxes"] = torch.stack(pred_boxes_, dim=0)
            out['reference_points'] = torch.stack(pred_inter_references_, dim=0)
            pred_masks = torch.stack(pred_masks_, dim=0).flatten(0,1)
            pred_masks = self.mask_refine(pred_masks, features[:2])
            pred_masks = F.pixel_shuffle(pred_masks, 4)
            outputs_seg_masks = rearrange(pred_masks.squeeze(1), '(b t) h w -> b t h w', b=b, t=t)
            out["pred_masks"] = outputs_seg_masks.unsqueeze(2)
        else:
            # only for A2D, JHMDB, Ref-COCO
            pred_masks = rearrange(outputs_seg_masks[-1], 'b (t q) c h w -> b q t c h w', t=t)
            out["pred_logits"] = out["pred_logits"]
            pred_masks_total = []
            for idx_ in range(self.num_queries):
                pred_masks_ = self.mask_refine(pred_masks[:,idx_].flatten(0,1), features[:2])
                pred_masks_ = F.pixel_shuffle(pred_masks_, 4)
                pred_masks_ = rearrange(pred_masks_.squeeze(1), '(b t) h w -> b t h w', b=b, t=t)
                pred_masks_total.append(pred_masks_)
            out["pred_masks"] = torch.stack(pred_masks_total, dim=2)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1])]

    @torch.jit.unused
    def _set_aux_loss_comprehensive(self, outputs_class, outputs_coord, outputs_seg_masks, outputs_seg_masks_low):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b, "pred_masks": c, "pred_masks_low": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_seg_masks[:-1], outputs_seg_masks_low[:-1])]

    def forward_text(self, captions, device):
        if isinstance(captions[0], str):
            text_features, text_sentence_features, text_pad_mask = self.text_encoder(captions, device)
            text_features = self.text_proj(text_features)
            text_sentence_features = self.sentence_proj(text_sentence_features)
            text_features = NestedTensor(text_features, text_pad_mask)  # NestedTensor
        else:
            raise ValueError("Please mask sure the caption is a list of string")
        return text_features, text_sentence_features

    def dynamic_mask_with_coords(self, mask_features, mask_head_params, reference_points, targets):
        """
        Add the relative coordinates to the mask_features channel dimension,
        and perform dynamic mask conv.

        Args:
            mask_features: [batch_size, time, c, h, w]
            mask_head_params: [batch_size, time * num_queries_per_frame, num_params]
            reference_points: [batch_size, time * num_queries_per_frame, 2], cxcy
            targets (list[dict]): length is batch size
                we need the key 'size' for computing location.
        Return:
            outputs_seg_mask: [batch_size, time * num_queries_per_frame, h, w]
        """
        device = mask_features.device
        b, t, c, h, w = mask_features.shape
        _, num_queries = reference_points.shape[:2]
        q = num_queries // t  # num_queries_per_frame

        # prepare reference points in image size (the size is input size to the model)
        # use xyxy rather than xy in xywh
        new_reference_points = []
        for i in range(b):
            img_h, img_w = targets[i]['size']
            scale_f = torch.stack([img_w, img_h], dim=0)
            tmp_reference_points = reference_points[i] * scale_f[None, :]
            new_reference_points.append(tmp_reference_points)
        new_reference_points = torch.stack(new_reference_points, dim=0)
        reference_points = new_reference_points

        # prepare the mask features
        if self.rel_coord:
            reference_points = rearrange(reference_points, 'b (t q) n -> b t q n', t=t, q=q)
            locations = compute_locations(h, w, device=device, stride=self.mask_feat_stride)
            relative_coords = reference_points.reshape(b, t, q, 1, 1, 2) - \
                                    locations.reshape(1, 1, 1, h, w, 2)
            relative_coords = relative_coords.permute(0, 1, 2, 5, 3, 4)

            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q)
            mask_features = torch.cat([mask_features, relative_coords], dim=3)
        else:
            mask_features = repeat(mask_features, 'b t c h w -> b t q c h w', q=q)
        mask_features = mask_features.reshape(1, -1, h, w)

        # parse dynamic params
        mask_head_params = mask_head_params.flatten(0, 1)  # btq, n
        weights, biases = parse_dynamic_params(
            mask_head_params, self.dynamic_mask_channels,
            self.weight_nums, self.bias_nums
        )

        # dynamic conditional segmentation
        mask_logits = self.mask_heads_forward(mask_features, weights, biases, mask_head_params.shape[0])
        mask_logits = rearrange(mask_logits, 'n (b t q c) h w -> (n b) (t q) c h w', t=t, q=self.num_queries, c=16)
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
        bias_splits[l] = bias_splits[l].reshape(num_insts * channels)

    return weight_splits, bias_splits

def aligned_bilinear(tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def build(args):
    if args.binary:
        num_classes = 1  # use this one
    else:
        if args.dataset_file == 'ytvos':
            num_classes = 65
        elif args.dataset_file == 'davis':
            num_classes = 78
        elif args.dataset_file == 'a2d' or args.dataset_file == 'jhmdb':
            num_classes = 1
        else:
            num_classes = 91
    device = torch.device(args.device)

    # backbone
    if 'video_swin' in args.backbone:
        from models.video_swin_transformer import build_video_swin_backbone
        backbone = build_video_swin_backbone(args)
    elif 'swin' in args.backbone:
        from models.swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args)
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    matcher = build_matcher(args)

    model = SgMg(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_frames=args.num_frames,
        mask_dim=args.mask_dim,
        dim_feedforward=args.dim_feedforward,
        controller_layers=args.controller_layers,
        dynamic_mask_channels=args.dynamic_mask_channels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        freeze_text_encoder=args.freeze_text_encoder,
        rel_coord=args.rel_coord,
        matcher=matcher
    )
    weight_dict = {}
    weight_dict['loss_ce'] = args.cls_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict['loss_mask'] = args.mask_loss_coef
        weight_dict['loss_dice'] = args.dice_loss_coef
        weight_dict['loss_mask_low'] = args.mask_loss_coef
        weight_dict['loss_dice_low'] = args.dice_loss_coef

    # this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    if args.masks:
        losses += ['masks']

    criterion = SetCriterion(
            args,
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=args.eos_coef,
            losses=losses,
            focal_alpha=args.focal_alpha)
    criterion.to(device)

    postprocessors = build_postprocessors(args, args.dataset_file)
    return model, criterion, postprocessors



