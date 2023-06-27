
from functools import partial
from itertools import repeat
import random
from typing import List, Optional
import einops
from scipy import ndimage
from skimage import measure

import torch
from torchvision.ops import batched_nms, box_convert
from ensemble_boxes import weighted_boxes_fusion

from model.detector import TokenDetectorOutput


def predict_observation_boxes_from_detected_regions(
    region_class_probs: torch.Tensor,
    region_class_preds: torch.Tensor,
    region_box_coords: torch.Tensor,
    region_mask: Optional[torch.Tensor],
    threshold: Optional[float] = None,
    box_postprocessing: Optional[str] = None,
    nms_iou_threshold: float = 0.5):
    """
    region_class_probs: (N x A x C)
    region_class_preds: (N x A x C)
    region_box_coords: (N x A x 4)
    region_box_mask: (N x A)
    threshold: If None -> use region_class_preds
    """
    device = region_class_probs.device
    N, A, C = region_class_probs.shape
    # --- Filter boxes ---
    if threshold is not None:
        region_class_preds = (region_class_probs > threshold)
    # (N x C x A)
    pred_boxes_mask = region_class_preds.transpose(1, 2)
    if region_mask is not None:
        pred_boxes_mask = pred_boxes_mask * region_mask.unsqueeze(1)

    # --- Construct boxes ---
    # (N x C x A)
    box_scores = region_class_probs.transpose(1, 2) * pred_boxes_mask
    # (N x C x A)
    box_cls_preds = einops.repeat(torch.arange(C, device=box_scores.device), 'c -> n c a', n=N, a=A)
    # (N x C x A x 4)
    box_coords = einops.repeat(region_box_coords, 'n a d -> n c a d', c=C)
    
    pred_boxes_mask = pred_boxes_mask.cpu().numpy()
    box_scores = box_scores.cpu().numpy()
    box_cls_preds = box_cls_preds.cpu().numpy()
    box_coords = box_convert(box_coords, in_fmt='cxcywh', out_fmt='xyxy').clamp(0., 1.).cpu().numpy()
    boxes = []
    for i in range(N):
        coords = box_coords[i, pred_boxes_mask[i]]
        scores = box_scores[i, pred_boxes_mask[i]]
        preds = box_cls_preds[i, pred_boxes_mask[i]]
    
        coords, scores, preds = weighted_boxes_fusion(
            [coords], [scores], [preds],
            weights=None,
            iou_thr=0.03,
            skip_box_thr=0.0,
            conf_type='max')

        coords = box_convert(torch.from_numpy(coords), in_fmt='xyxy', out_fmt='cxcywh')
        scores = torch.from_numpy(scores)
        preds = torch.from_numpy(preds)
        boxes.append(torch.cat([coords, preds.unsqueeze(-1), scores.unsqueeze(-1)], dim=-1).to(device=device))

    if box_postprocessing == 'nms':
        boxes = apply_nms(boxes, nms_iou_threshold)
    elif box_postprocessing == 'top1perclass':
        boxes = filter_top1_box_per_class(boxes)
    else:
        assert box_postprocessing is None, f'Unknown box postprocessing: {box_postprocessing}, must be one of [None, "nms", "top1perclass"]'

    return boxes


def apply_nms(predicted_boxes: List[torch.Tensor], iou_threshold: float):
    predicted_boxes_after_nms = []
    for sample_boxes in predicted_boxes:
        boxes_coords = box_convert(sample_boxes[:, 0:4], in_fmt='cxcywh', out_fmt='xyxy')
        cls_idxs = sample_boxes[:, 4]
        scores = sample_boxes[:, 5]
        nms_indices = batched_nms(boxes_coords, scores, cls_idxs, iou_threshold=iou_threshold)
        predicted_boxes_after_nms.append(sample_boxes[nms_indices, :])
    return predicted_boxes_after_nms


def filter_top1_box_per_class(boxes: List[torch.Tensor]):
    filtered_boxes = []
    for sample_boxes in boxes:
        unique_classes = torch.unique(sample_boxes[:, 4])
        filtered_sample_boxes = []
        for c in unique_classes:
            # Select boxes of class c
            class_inds = sample_boxes[:, 4] == c
            sample_boxes_c = sample_boxes[class_inds]
            # Select box with highest confidence for class c
            best_idx = sample_boxes_c[:, -1].argmax()
            best_sample_box = sample_boxes_c[None, best_idx]
            # Gather over boxes
            filtered_sample_boxes.append(best_sample_box)
        # Gather over samples
        filtered_boxes.append(torch.cat(filtered_sample_boxes) if len(filtered_sample_boxes) > 0 else torch.zeros(0, 6))
    return filtered_boxes
