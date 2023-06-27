
from dataclasses import dataclass
from typing import Optional

from torch import BoolTensor, Tensor


@dataclass
class ImageEncoderOutput:
    # (N x H x W x d) -> already projected to model space
    patch_features: Tensor
    # (N x H x W x d)
    pos_embeddings: Tensor
    # (N x d)
    global_features: Optional[Tensor] = None
    # (N x H x W)
    patch_mask: Optional[BoolTensor] = None
