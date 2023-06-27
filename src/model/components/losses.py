
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.loss import AsymmetricLossMultiLabel


def get_loss_fn(loss_name: str, label_smoothing=None, loss_coeff=None, **kwargs):
    if loss_name == 'bce':
        loss = nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == 'wbce':
        loss = weighted_bce_logits
    elif loss_name == 'asl':
        loss = AsymmetricLossMultiLabel(**kwargs)
    elif loss_name == 'rank':
        loss = multilabel_ranking_loss
    elif loss_name == 'mlsl':
        loss = partial(multilabel_softmax_loss, **kwargs)
    else:
        raise ValueError(loss_name)

    if label_smoothing is not None and loss_name not in ('rank', 'mlsl'):
        loss = LabelSmoothingWrapper(loss, label_smoothing)

    if loss_coeff is not None:
        loss = LossCoeffWrapper(loss, loss_coeff)
    return loss

class LossCoeffWrapper(nn.Module):
    def __init__(self, loss_fn, loss_coeff):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_coeff = loss_coeff

    def forward(self, pred, target):
        return self.loss_coeff * self.loss_fn(pred, target)
    

class LabelSmoothingWrapper(nn.Module):
    def __init__(self, loss_fn, label_smoothing=0.1):
        super().__init__()
        self.loss_fn = loss_fn
        self.label_smoothing = label_smoothing

    def forward(self, pred, target):
        target = target.to(dtype=pred.dtype)
        target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.loss_fn(pred, target)


def multilabel_ranking_loss(predictions, labels, margin=1.0):
    """Compute the multilabel ranking loss.
    See https://arxiv.org/pdf/2207.01887.pdf
    
    Args:
        predictions: A tensor of shape (N x C) representing the
            predicted scores for each class.
        labels: A tensor of shape (N x C) representing the
            binary labels for each class.
        margin: The margin to use in the loss.
    
    Returns:
        A scalar tensor representing the multilabel ranking loss.
    """
    N, C = labels.shape

    # Compute the pairwise differences between the predictions
    # (N x Cp x 1) - (N x 1 x Cn) = (N x Cp x Cn)
    differences = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    # (N x Cp x Cn)
    margin_differences = torch.clamp(margin - differences, min=0)

    # Only consider differences where Cp is positive and Cn is negative
    # (N x Cp x Cn)
    mask = labels.unsqueeze(2) * (1 - labels.unsqueeze(1))
    # (N)
    loss = (margin_differences * mask).sum(dim=(1, 2)) / C
    return loss.mean()


def multilabel_softmax_loss(logits, labels, gamma: float = 1.0):
    # see https://ieeexplore-ieee-org/stamp/stamp.jsp?tp=&arnumber=9906105

    N, C = labels.shape
    # (N x C)
    pos_mask = labels.bool()
    # (N x C)
    neg_mask = ~pos_mask
    # (N x 1)
    neutral_scores = torch.zeros((N, 1), device=logits.device, dtype=logits.dtype)

    # (N x C)
    pos_scores = - gamma * logits
    pos_scores = pos_scores.masked_fill(neg_mask, -torch.inf)
    # (N x C+1)
    pos_scores = torch.cat([neutral_scores, pos_scores], dim=1)
    # (N)
    pos_loss = pos_scores.logsumexp(dim=1) / gamma

    # (N x C)
    neg_scores = gamma * logits
    neg_scores = neg_scores.masked_fill(pos_mask, -torch.inf)
    # (N x C+1)
    neg_scores = torch.cat([neutral_scores, neg_scores], dim=1)
    # (N)
    neg_loss = neg_scores.logsumexp(dim=1) / gamma

    return (pos_loss + neg_loss).mean()


def compute_weights_per_batch(target_classes: Tensor) -> Tensor:
    C = target_classes.shape[-1]
    target_classes = target_classes.view(-1, C)
    N, C = target_classes.shape
    N_pos = target_classes.sum(0)  # (C)
    N_neg = N - N_pos  # (C)

    weight_pos = (N + 1) / (N_pos + 1)  # (C)
    weight_neg = (N + 1) / (N_neg + 1)  # (C)

    return weight_pos, weight_neg


def weighted_bce(pred_class_probs, target_classes, clamp_min=None):
    if pred_class_probs.ndim > 2:
        *dims, C = pred_class_probs.shape
        pred_class_probs = pred_class_probs.reshape(-1, C)
        target_classes = target_classes.reshape(-1, C)

    weight_pos, weight_neg = compute_weights_per_batch(target_classes)  # (C)

    pred_class_probs = pred_class_probs.float()
    if clamp_min is not None:
        pred_class_probs = pred_class_probs.clamp(min=clamp_min, max=1. - clamp_min)
    target_classes = target_classes.float()

    loss = - weight_pos * target_classes * pred_class_probs.log() - weight_neg * (1. - target_classes) * (1. - pred_class_probs).log()
    return loss.mean()


def weighted_bce_logits(pred_class_logits, target_classes):
    if pred_class_logits.ndim > 2:
        *dims, C = pred_class_logits.shape
        pred_class_logits = pred_class_logits.reshape(-1, C)
        target_classes = target_classes.reshape(-1, C)

    weight_pos, weight_neg = compute_weights_per_batch(target_classes)  # (C)

    pred_class_logits = pred_class_logits.float()
    target_classes = target_classes.float()

    # (C)
    total_weight = weight_neg
    pos_weight = weight_pos / total_weight
    # (N x C)
    loss = total_weight * F.binary_cross_entropy_with_logits(
                                pred_class_logits, 
                                target_classes, 
                                pos_weight=pos_weight,
                                reduction='none')
    return loss.mean()
