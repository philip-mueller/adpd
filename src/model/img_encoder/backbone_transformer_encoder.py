
from dataclasses import dataclass
from typing import Optional
import einops
from omegaconf import MISSING
from transformers import DetrConfig
from torch import FloatTensor, Tensor
from torch import nn
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.detr import DetrConfig
import transformers.models.detr.modeling_detr as detr
from model.components.transformer import TransformerEncoderModule

from model.detector import TokenDetectorOutput
from model.img_encoder import ImageEncoderOutput
from model.img_encoder.timm_conv_model import TimmConvModelConfig
from utils.model_utils import BaseModel, BaseModelConfig, MainModelConfig, instantiate_model


@dataclass
class BackboneTransformerEncoderConfig(BaseModelConfig):
    encoder_layers: int = 3
    backbone: BaseModelConfig = TimmConvModelConfig()
    shortcut_pos_embeddings: bool = False

class BackboneTransformerEncoder(BaseModel):
    CONFIG_CLS = BackboneTransformerEncoderConfig

    def __init__(self, config: BackboneTransformerEncoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: BackboneTransformerEncoderConfig

        self.d = main_config.d_model

        from model import img_encoder
        self.backbone = instantiate_model(self.config.backbone, model_module=img_encoder, main_config=main_config)

        self.encoder = TransformerEncoderModule(
            d_model=main_config.d_model, n_layers=self.config.encoder_layers,
            nhead=main_config.n_head, act=main_config.act, 
            dropout=main_config.dropout, attention_dropout=main_config.attention_dropout,
            droppath_prob=main_config.droppath_prob,
            layer_scale=main_config.layer_scale, layer_scale_init=main_config.layer_scale_init,
            shortcut_pos_embeddings=self.config.shortcut_pos_embeddings)

        #self.apply(self._init_weights)

    def forward(self, 
        x: Tensor, 
        **kwargs) -> TokenDetectorOutput:
        """
        :param x: Image (N x 3 x H_pixel x W_pixel)
        :param pixel_mask: Image pixel mask (N x H_pixel x W_pixel)

        Note: implementation adapted from huggingface's DetrModel (https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/detr/modeling_detr.py)
        """

        backbone_output: ImageEncoderOutput = self.backbone(x, **kwargs)
        N, H, W, d = backbone_output.patch_features.shape

        # (N x (H*W) x d)
        flattened_features = backbone_output.patch_features.flatten(1, 2)
        # (N x (H*W) x d)
        flattened_pos_emb = backbone_output.pos_embeddings.flatten(1, 2)

        # (N x (H*W) x d)
        encoded_patch_features = self.encoder(
            features=flattened_features,
            mask=backbone_output.patch_mask,
            pos_embeddings=flattened_pos_emb,
        )
        encoded_patch_features = einops.rearrange(encoded_patch_features, 'n (h w) d -> n h w d', h=H, w=W)

        return ImageEncoderOutput(
            patch_features=encoded_patch_features,
            patch_mask=backbone_output.patch_mask,
            pos_embeddings=backbone_output.pos_embeddings
        )

    #def _init_weights(self, module):
    # std = 0.02

    # if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
    #     # Slightly different from the TF version which uses truncated_normal for initialization
    #     # cf https://github.com/pytorch/pytorch/pull/5617
    #     module.weight.data.normal_(mean=0.0, std=std)
    #     if module.bias is not None:
    #         module.bias.data.zero_()
