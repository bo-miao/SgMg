"""
This file contains a wrapper for Video-Swin-Transformer so it can be properly used as a temporal encoder for MTTR.
"""
import torch
import os
from torch import nn, Tensor
from einops import rearrange, repeat

from transformers import RobertaModel, RobertaTokenizerFast
from models.text_encoder.tokenizer import RobertaTokenizer

import warnings
warnings.filterwarnings("ignore")


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.text_backbone_name = args.text_backbone
        self.token_size = 32
        if self.text_backbone_name == "Roberta":
            self.text_backbone = RobertaModel.from_pretrained("roberta-base")
            # self.text_backbone.pooler = None  # this pooler is never used, this is a hack to avoid DDP problems...
            self.tokenizer = RobertaTokenizer()
            self.feat_dim = 768
        else:
            assert False, f'error: Text Encoder "{self.text_backbone_name}" is not supported'

        self.freeze_text_encoder = args.freeze_text_encoder
        if self.freeze_text_encoder:
            # self.text_backbone.eval()
            for p in self.text_backbone.parameters():
                p.requires_grad_(False)
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)
        print("Use {} as text encoder. Freeze: {}".format(self.text_backbone_name, self.freeze_text_encoder))

        self.target_len = None

    def forward(self, texts, device):
        if self.freeze_text_encoder:
            with torch.no_grad():
                tokenized_queries = self.tokenizer(texts).to(device)
                if self.text_backbone_name == "Roberta":
                    encoded_text = self.text_backbone(**tokenized_queries)
                    text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()
                    text_features = encoded_text.last_hidden_state
                    text_sentence_features = encoded_text.pooler_output
                else:
                    raise NotImplementedError
        else:
            tokenized_queries = self.tokenizer(texts).to(device)
            if self.text_backbone_name == "Roberta":
                encoded_text = self.text_backbone(**tokenized_queries)
                text_pad_mask = tokenized_queries.attention_mask.ne(1).bool()
                text_features = encoded_text.last_hidden_state
                text_sentence_features = encoded_text.pooler_output
            else:
                raise NotImplementedError

        return text_features, text_sentence_features, text_pad_mask

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

