

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import einops
from omegaconf import MISSING
from torch import FloatTensor, Tensor
from torch import nn
import torch
import torch.nn.functional as F
from transformers.models.detr.modeling_detr import generalized_box_iou
from metrics.detection_metrics import FixedSetDetectionMetrics
from model.components.losses import weighted_bce_logits
from model.components.mlp import MLP
from model.components.transformer import TransformerTokenDecoder

from model.detector import TokenDetectorOutput
from model.img_encoder import ImageEncoderOutput
from utils.model_utils import BaseModel, BaseModelConfig, MainModelConfig
from utils.plot_utils import prepare_wandb_bbox_images
from torchvision.ops import roi_pool, box_convert


def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

@dataclass
class TokenDecoderDetectorConfig(BaseModelConfig):
    auxiliary_loss: bool = False
    bbox_loss_coefficient: float = 5.
    giou_loss_coefficient: float = 2.
    box_mask_loss_coefficient: float = 1.

    n_joint_encoder_layers: int = 0
    n_decoder_layers: int = 6
    n_output_encoder_layers: int = 0
    # 0 = no projection, 1 = linear, 2 = one hidden layer
    n_feature_mlp_layers: int = 0

    enc_dec_droppath: bool = False
    decoder_sa: bool = True
    decoder_ff: bool = True
    shortcut_tokens: bool = False
    use_pos_embeddings: bool = True
    shortcut_pos_embeddings: bool = False
    # roi_pool, soft_roi_pool
    roi_pool: Optional[str] = None
    skip_con_roi_pool: bool = False

class TokenDecoderDetector(BaseModel):

    CONFIG_CLS = TokenDecoderDetectorConfig
    def __init__(self, config: TokenDecoderDetectorConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: TokenDecoderDetectorConfig

        self.d = main_config.d_model
        
        self.decoder = TransformerTokenDecoder(
            d_model=self.d, nhead=main_config.n_head, 
            n_joint_encoder_layers=self.config.n_joint_encoder_layers,
            n_decoder_layers=self.config.n_decoder_layers,
            n_output_encoder_layers=self.config.n_output_encoder_layers,
            act=main_config.act, dropout=main_config.dropout, attention_dropout=main_config.attention_dropout,
            droppath_prob=main_config.droppath_prob,
            layer_scale=main_config.layer_scale, layer_scale_init=main_config.layer_scale_init,
            enc_dec_droppath=self.config.enc_dec_droppath, 
            decoder_sa=self.config.decoder_sa, decoder_ff=self.config.decoder_ff,
            shortcut_tokens=self.config.shortcut_tokens, shortcut_pos_embeddings=self.config.shortcut_pos_embeddings
        )

        self.bbox_predictor = MLP(
            n_layers=3,
            d_in=self.d, d_hidden=self.d, d_out=4,
            dropout_last_layer=False,
            act=main_config.act, dropout=main_config.dropout
        )
        self.query_present_classifier = nn.Linear(main_config.d_model, 1)

        if self.config.roi_pool == 'roi_pool':
            self.roi_pool = FeatureRoiPool(d=self.d, skip_con_roi_pool=self.config.skip_con_roi_pool, feature_projector=True)
        else:
            assert self.config.roi_pool is None, f"Unknown ROI pooling method: {self.config.roi_pool}"
            self.roi_pool = None

        self.output_feature_projection = MLP(
            self.config.n_feature_mlp_layers, self.d,
            act=main_config.act, dropout=main_config.dropout)

        self.loss_weights = {
            'loss_bbox': self.config.bbox_loss_coefficient,
            'loss_giou': self.config.giou_loss_coefficient,
            'loss_box_mask': self.config.box_mask_loss_coefficient
        }
        for i in range(self.config.n_decoder_layers):
            self.loss_weights.update({k + f"_{i}": v for k, v in self.loss_weights.items()})


        # self.apply(self._init_weights)

    def build_metrics(self, query_names) -> nn.Module:
        return FixedSetDetectionMetrics(class_names=query_names)

    def plot(self, model_output: TokenDetectorOutput, input: dict, target_dir: str, prefix: str = 'bounding_boxes', **kwargs) -> dict:
        x = input['x']
        class_names = input['class_names']
        return {
            prefix: prepare_wandb_bbox_images(
                images=x, 
                preds=model_output.boxes,
                predicted_box_masks=model_output.box_mask,
                predicted_box_mask_logits=model_output.box_mask_logits,
                targets=model_output.target_boxes,
                target_box_masks=model_output.target_box_mask,
                class_names=class_names,
                one_box_per_class=True)
        }

    def forward(self, 
        encoded_image: ImageEncoderOutput,
        query_tokens: Tensor, 
        target_boxes:  Optional[Tensor]=None, 
        target_mask:  Optional[Tensor]=None,
        return_predictions=True, 
        return_attentions=False,
        compute_loss=True,
        **kwargs) -> TokenDetectorOutput:
        """
        :param x: Image (N x 3 x H_pixel x W_pixel)
        :param pixel_mask: Image pixel mask (N x H_pixel x W_pixel)
        :param tokens: (Q x d_q) Note that there is no batch dimension!

        Note: implementation adapted from huggingface's DetrModel (https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/models/detr/modeling_detr.py)
        """
        assert query_tokens is not None

        output = self.encode_queries(encoded_image, query_tokens, return_attentions=return_attentions)
        if return_predictions or compute_loss or self.roi_pool is not None:
            output = self.predict_boxes(output)
        output.target_box_mask = target_mask
        output.target_boxes = target_boxes

        if compute_loss:
            assert target_boxes is not None and target_mask is not None
            output = self.compute_loss(output, target_boxes, target_mask)

        if self.roi_pool is not None:
            output.box_features = self.roi_pool(
                patch_features=encoded_image.patch_features, 
                query_features=output.box_features,
                query_boxes=output.boxes,
                query_box_masks=output.box_mask)

        # project the features for output -> allows to make this more independent from position and present prediction
        output.box_features = self.output_feature_projection(output.box_features)
        if output.intermediate_features is not None:
             output.intermediate_features = self.output_feature_projection(output.intermediate_features)

        return output

    def encode_queries(self, 
        encoded_image: ImageEncoderOutput,
        query_tokens: Tensor, 
        return_attentions=False) -> TokenDetectorOutput:

        assert query_tokens.shape[-1] == encoded_image.patch_features.shape[-1], f"Query tokens ({query_tokens.shape[-1]}) and patch features ({encoded_image.patch_features.shape[-1]}) must have the same dimensionality"
        N, H, W, d = encoded_image.patch_features.shape
        assert d == self.d, f"Patch features ({d}) and model dimensionality ({self.d}) must match"
        # (N x (H*W) x d)
        flattened_features = encoded_image.patch_features.flatten(1, 2)
        if query_tokens.ndim == 2:
            query_tokens = einops.repeat(query_tokens, 'q d -> n q d', n=N)
        N, Q, d = query_tokens.shape
        assert query_tokens.shape[0] == flattened_features.shape[0], f"Query tokens ({query_tokens.shape[0]}) and patch features ({flattened_features.shape[0]}) must have the same batch size"

        if self.config.use_pos_embeddings:
            # (N x (H*W) x d_model)
            flattened_pos_emb = encoded_image.pos_embeddings.flatten(1, 2)
        else:
            flattened_pos_emb = None

        if encoded_image.patch_mask is not None:
            # (N x (H*W))
            flattened_mask = encoded_image.patch_mask.flatten(1)
        else:
            flattened_mask = None

        # Apply decoder
        query_features, assigment_probs, intermediate_features, intermediate_atts = self.decoder(
            token_features=query_tokens, region_features=flattened_features,
            region_pos_embeddings=flattened_pos_emb, region_mask=flattened_mask,
            return_intermediate=self.config.auxiliary_loss, return_intermediate_attentions=False
        )
        # (N x Q x d_model)
        output = TokenDetectorOutput(box_features=query_features)

        if self.config.auxiliary_loss:
            output.intermediate_features = torch.stack(intermediate_features, dim=0)

        if return_attentions:
            output.query_patch_assignment_probs = assigment_probs.view(N, Q, H, W)

        return output

    def predict_boxes(self, output: TokenDetectorOutput) -> TokenDetectorOutput:
        # (N x Q)
        output.box_mask_logits = self.query_present_classifier(output.box_features).squeeze(-1)
        # (N x Q)
        output.box_mask = output.box_mask_logits > 0.5
        # (N x Q x 4)
        output.boxes = self.bbox_predictor(output.box_features).sigmoid()

        # Auxiliary predictors
        if self.config.auxiliary_loss:
            # (N_layers x N x Q x d_model) -> used for auxiliary losses
            intermediate_query_feature = output.intermediate_features
            # (N_layers x N x Q)
            output.intermediate_mask_logits = self.query_present_classifier(intermediate_query_feature).squeeze(-1)
            output.intermediate_boxes = self.bbox_predictor(intermediate_query_feature).sigmoid()

        return output

    def compute_loss(self, output: TokenDetectorOutput, target_boxes: Tensor, target_mask: Tensor) -> TokenDetectorOutput:
        num_boxes = target_mask.sum()

        losses = {}
        losses["loss_bbox"], losses["loss_giou"], losses["loss_box_mask"] =\
        self._compute_loss(
            output.boxes, output.box_mask_logits,
            target_boxes, target_mask, num_boxes)
        
        if self.config.auxiliary_loss:
            for i, (aux_boxes, aux_boxmask_logits) in enumerate(zip(output.intermediate_boxes, 
                                                                    output.intermediate_mask_logits)):
                losses[f"aux_loss/bbox_{i}"], losses[f"aux_loss/giou_{i}"], losses[f"aux_loss/box_mask_{i}"] =\
                    self._compute_loss(
                        aux_boxes, aux_boxmask_logits,
                        target_boxes, target_mask, num_boxes)

        loss = sum(losses[k] * self.loss_weights[k] for k in losses.keys())

        output.loss = loss
        output.step_metrics.update(losses)
        return output

    def _compute_loss(
        self,
        preditecd_boxes: Tensor, predicted_mask_logits: Tensor,
        target_boxes: Tensor, target_mask: Tensor, 
        num_boxes: Tensor):
        # (N*A x 4)
        preditecd_boxes = preditecd_boxes.flatten(0, 1)
        target_boxes = target_boxes.flatten(0, 1)
        # (N*A)
        predicted_mask_logits = predicted_mask_logits.flatten()
        target_mask = target_mask.flatten().to(dtype=predicted_mask_logits.dtype)

        # (N*A)
        loss_bbox = nn.functional.l1_loss(preditecd_boxes, target_boxes, reduction="none").sum(-1)
        loss_bbox = (target_mask * loss_bbox).sum() / num_boxes

        # Bbox GIoU loss
        # (N*A)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(preditecd_boxes), 
                center_to_corners_format(target_boxes))
        )
        loss_giou = (target_mask * loss_giou).sum() / num_boxes
        
        # Box mask loss (BCE)
        loss_box_mask = weighted_bce_logits(predicted_mask_logits, target_mask)

        return loss_bbox, loss_giou, loss_box_mask


class FeatureRoiPool(nn.Module):
    def __init__(self, d: int, skip_con_roi_pool: bool, feature_projector=True):
        super().__init__()
        self.skip_con_roi_pool = skip_con_roi_pool

        if feature_projector:
            self.feature_projector = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d)
            )
        else:
            self.feature_projector = nn.Identity()

    def forward(self, patch_features, query_features, query_boxes, query_box_masks):
        """
        patch_features: (N x H x W x d)
        query_features: (N x Q x d)
        query_boxes: (N x Q x 4)
        query_box_masks: (N x Q)
        """
        N, H, W, d = patch_features.shape
        assert H == W
        patch_features = self.feature_projector(einops.rearrange(patch_features, "N H W d -> N (H W) d"))
        # (N x d x H x W)
        patch_features = einops.rearrange(patch_features, "N (H W) d -> N d H W", H=H, W=W)
        # (N)
        box_sample_indices = torch.arange(query_features.shape[0], device=query_features.device, dtype=query_features.dtype)
        # (N x Q)
        box_sample_indices = einops.repeat(box_sample_indices, "N -> N Q", Q=query_features.shape[1])
        # (N x Q x 5)
        boxes = torch.cat([box_sample_indices.unsqueeze(-1), query_boxes], dim=-1)
        # (K x 5)
        boxes = boxes[query_box_masks]
        boxes[:, 1:] = box_convert(boxes[:, 1:], "cxcywh", "xyxy")
        # (K x d x 1 x 1)
        roi_pooled_features = roi_pool(patch_features, boxes, output_size=(1, 1), spatial_scale=H)
        roi_pooled_features = roi_pooled_features.squeeze(-1).squeeze(-1)
        # (N x Q x d)
        roi_pooled_features_reshaped = torch.zeros_like(query_features, dtype=query_features.dtype)
        roi_pooled_features_reshaped[query_box_masks] = roi_pooled_features.to(dtype=query_features.dtype)

        if self.skip_con_roi_pool:
            roi_pooled_features_reshaped = roi_pooled_features_reshaped + query_features

        return roi_pooled_features_reshaped
