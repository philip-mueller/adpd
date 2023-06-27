
from dataclasses import dataclass
from typing import Optional
import einops
from omegaconf import MISSING
from transformers import DetrConfig
from torch import Tensor
from torch import nn
import torch
from timm import create_model
import torch.nn.functional as F
from transformers.models.detr import modeling_detr as detr
from metrics.detection_metrics import FixedSetDetectionMetrics
from model.components.mlp import MLP
from model.img_encoder import ImageEncoderOutput

from utils.model_utils import BaseModel, BaseModelConfig, MainModelConfig


@dataclass
class TimmConvModelConfig(BaseModelConfig):
    backbone: str = "resnet50"
    use_pretrained_backbone: bool = True
    dilation: bool = False
    frozen_backbone: bool = False

    # 0 = no projection, 1 = linear, 2 = one hidden layer
    n_projection_layers: int = 1
    backbone_dropout: float = 0.0
    backbone_drop_path: float = 0.0
    projection_bn: bool = False

class TimmConvModel(BaseModel):
    CONFIG_CLS = TimmConvModelConfig
    MODIFYABLE_CONFIGS = ('frozen_backbone', )

    def __init__(self, config: TimmConvModelConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: TimmConvModelConfig

        self.d = main_config.d_model

        kwargs = {}
        if self.config.dilation:
            kwargs["output_stride"] = 16
        if self.config.backbone_dropout > 0.0:
             kwargs["drop_rate"] = self.config.backbone_dropout
        if self.config.backbone_drop_path > 0.0:
             kwargs["drop_path_rate"] = self.config.backbone_drop_path

        self.backbone = create_model(self.config.backbone, pretrained=self.config.use_pretrained_backbone, features_only=True, out_indices=(1, 2, 3, 4), **kwargs)
        if config.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        d_backbone = self.backbone.feature_info.channels()[-1]
        self.position_embeddings = detr.build_position_encoding(DetrConfig(position_embedding_type="sine", d_model=main_config.d_model))
        
        self.patch_projection = MLP(
            self.config.n_projection_layers, 
            d_in=d_backbone, 
            d_out=self.d, 
            use_bn=self.config.projection_bn,
            act=main_config.act,
            dropout=main_config.dropout)

        self.apply(self._init_weights)

    def forward(self, 
        x: Tensor, 
        **kwargs) -> ImageEncoderOutput:
        """
        :param x: Image (N x 3 x H_pixel x W_pixel)
        :param pixel_mask: Image pixel mask (N x H_pixel x W_pixel)

        Note: implementation adapted from huggingface's DetrModel (https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/detr/modeling_detr.py)
        """
        if x.ndim == 3:
            x = einops.repeat(x, 'n h w -> n c h w', c=3)
        N, _, H_pixel, W_pixel = x.shape
        device = x.device
        dtype = x.dtype
        # Encode image using backbone
        # (N x d_backbone x H x W)
        patch_features = self.backbone(x)[-1]
        N, _, H, W = patch_features.shape

        # Reshape and project
        # (N x H x W x d)
        patch_features = einops.rearrange(patch_features, 'n d h w -> n h w d')
        # (N x H x W x d)
        projected_patch_features = self.patch_projection(patch_features)
        
        pos_embeddings = self.position_embeddings(projected_patch_features, torch.ones(N, H, W, dtype=dtype, device=device)).to(dtype)
        pos_embeddings = einops.rearrange(pos_embeddings, 'n d h w -> n h w d')

        return ImageEncoderOutput(
            patch_features=projected_patch_features,
            pos_embeddings=pos_embeddings,
            patch_mask=None)

    def _init_weights(self, module):
        std = 0.02

        if isinstance(module, detr.DetrLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        