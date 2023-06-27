from dataclasses import dataclass
import dataclasses
from typing import Dict, List, Optional
from omegaconf import MISSING
from torch import BoolTensor, FloatTensor

from utils.model_utils import BaseModelOutput



@dataclass
class TokenDetectorOutput(BaseModelOutput):
    # (N x Q x d) 
    box_features: FloatTensor = MISSING
    # (N x Q x 4)
    # in the (x_c, y_c, w, h) format (center-format)
    # with values in [0, 1] relative to the original (masked) image size
    boxes: Optional[FloatTensor] = None
    # (N x Q)
    box_mask_logits: Optional[FloatTensor] = None
    # (N x Q)
    box_mask: Optional[BoolTensor] = None

    # (N x Q x H x W)
    patch_map: Optional[FloatTensor] = None

    # (N x Q)
    target_box_mask: Optional[BoolTensor] = None
    # (N x Q x 4)
    # in the (x_c, y_c, w, h) format (center-format)
    # with values in [0, 1] relative to the original (masked) image size
    target_boxes: Optional[FloatTensor] = None

    # (N_layers x N x Q x d) 
    intermediate_features: Optional[FloatTensor] = None
    # (N_layers x N x Q x 4) 
    intermediate_boxes: Optional[FloatTensor] = None
    # (N_layers x N x Q) 
    intermediate_mask_logits: Optional[FloatTensor] = None

    # (N x Q x H x W)
    query_patch_assignment_probs: Optional[FloatTensor] = None
