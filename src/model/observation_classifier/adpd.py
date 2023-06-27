import csv
from dataclasses import dataclass, field
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import einops
import numpy as np
from model.detector.token_decoder_detector import TokenDecoderDetectorConfig
from omegaconf import MISSING
from settings import RESOURCES_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.datasets import DatasetConfig, ObservationSet
from model import detector
from model import img_encoder
from model.components.bbox_prediction import predict_observation_boxes_from_detected_regions
from model.components.mil import get_mil_aggregation
from model.components.losses import get_loss_fn

from model.detector import TokenDetectorOutput
from model.img_encoder import ImageEncoderOutput
from model.observation_classifier import ObservtationClassifierMetrics, ObservationClassifierOutput
from utils.model_utils import BaseModel, MainModelConfig, instantiate_model
from model.components.mlp import MLP
from utils.plot_utils import plot_and_save_img_bboxes, prepare_wandb_bbox_images


@dataclass
class AnatObservationClassifierConfig(MainModelConfig):
    image_encoder: Any = MISSING
    detector: TokenDecoderDetectorConfig = MISSING

    # bce, wbce, asl, rank, mlsl
    per_region_loss_fn: Optional[str] = None
    per_region_loss_args: Dict[str, Any] = field(default_factory=dict)

    mil_loss_fn: Optional[str] = None
    mil_loss_args: Dict[str, Any] = field(default_factory=dict)
    
    n_cls_hidden_layers: int = 0  # -> linear
    d_classifier_hidden: Optional[int] = None

    learned_observation_groups: Optional[List[ObservationSet]] = None
    learned_anatomy_names: Optional[List[str]] = None
    
    # None, LSE, noisy_or, avg, max
    mil_aggregation: Optional[str] = None
    # None, top1_per_class, nms
    box_postprocessing: Optional[str] = None
    nms_iou_threshold: float = 0.5
    
   
class AnatObservationClassifier(BaseModel):
    CONFIG_CLS = AnatObservationClassifierConfig

    def __init__(self, config: AnatObservationClassifierConfig) -> None:
        super().__init__(config)
        self.config: AnatObservationClassifierConfig
        self.d = self.config.d_model

        self.image_encoder = instantiate_model(
            self.config.image_encoder, model_module=img_encoder, main_config=self.config)
        n_anat = len(self.config.learned_anatomy_names)
        self.anatomy_tokens = nn.Parameter(torch.randn(n_anat, self.d) / math.sqrt(self.d))
        self.detector = instantiate_model(self.config.detector, model_module=detector, main_config=self.config)
        
        self.classifier = ObservationClassifier(
            n_cls_hidden_layers=self.config.n_cls_hidden_layers,
            d_classifier_hidden=self.config.d_classifier_hidden,
            learned_observation_groups=self.config.learned_observation_groups,
            d=self.d, act=self.config.act, dropout=self.config.dropout)

        self.per_region_loss_fn = get_loss_fn(self.config.per_region_loss_fn, **self.config.per_region_loss_args) if self.config.per_region_loss_fn is not None else None
        self.mil_aggregation = get_mil_aggregation(self.config.mil_aggregation)
        self.mil_loss = get_loss_fn(self.config.mil_loss_fn, **self.config.mil_loss_args) if self.config.mil_loss_fn is not None else None

    def forward(self, 
        x: torch.Tensor, 
        observation_class_names: List[str],
        anatomy_names: Optional[List[str]] = None,
        target_anatomy_boxes: Optional[torch.FloatTensor] = None, 
        target_anatomy_box_masks: Optional[torch.BoolTensor] = None, 
        target_observation_classes: Optional[torch.BoolTensor] = None, 
        target_anatomy_observations: Optional[torch.BoolTensor] = None, # (N x A x C')
        targets_with_anatomy_observations: Optional[torch.BoolTensor] = None, # (C) with C' positive labels
        target_observation_bboxes: Optional[List[List[torch.Tensor]]] = None, # List (N) of tensors (M_i x 5)
        compute_loss=False, return_predictions=False, **kwargs) -> ObservationClassifierOutput:
         
        # ---------- Encode the image features (i.e. patches) ----------
        encoded_image: ImageEncoderOutput = self.image_encoder(x=x)

        # ---------- Detect anatomical regions and extract anatomy features  ----------
        assert anatomy_names is None or anatomy_names == self.config.learned_anatomy_names
        anatomy_detector_output: TokenDetectorOutput = \
            self.detector(
                encoded_image=encoded_image,
                query_tokens=self.anatomy_tokens, # (A x d)
                target_boxes=target_anatomy_boxes, target_mask=target_anatomy_box_masks,
                compute_loss=compute_loss and target_anatomy_boxes is not None,
                return_predictions=return_predictions)
        
        # (N x A)
        region_mask = anatomy_detector_output.box_mask \
            if anatomy_detector_output.target_box_mask is None else anatomy_detector_output.target_box_mask
        region_box_coords = anatomy_detector_output.boxes
        # (N x A x d)
        region_features = anatomy_detector_output.box_features
        
        # ---------- Classify region features / MIL aggregation ----------
        # (N x A x C)
        regions_class_logits, regions_class_probs, regions_class_preds = self.classifier(
            region_features, observation_class_names=observation_class_names)

        # Optional: MIL aggregation of region classification probs
        mil_class_probs, mil_class_preds, mil_class_logits = None, None, None
        if self.config.mil_aggregation is not None and regions_class_probs is not None:
            # (N x C)
            mil_class_probs = self.mil_aggregation(regions_class_probs, mask=region_mask)
            # (N x C)
            mil_class_preds = (region_mask[:, :, None] * regions_class_preds).any(dim=1)
            # (N x C)
            mil_class_logits = mil_class_probs.clamp(min=1e-7, max=1-1e-7).log() - (1 - mil_class_probs).clamp(min=1e-7, max=1-1e-7).log()

        
        # ---------- (Optional) Predict observation boxes ----------
        if return_predictions or target_observation_bboxes is not None:
            assert regions_class_probs is not None, "Must use region classification for observation detection"
            assert region_box_coords is not None, "Must use region boxes for observation detection"
            
            pred_observation_boxes: List[torch.Tensor] = predict_observation_boxes_from_detected_regions(
                region_class_probs=regions_class_probs,
                region_class_preds=regions_class_preds,
                region_box_coords=region_box_coords,
                region_mask=region_mask,
                box_postprocessing=self.config.box_postprocessing,
                nms_iou_threshold=self.config.nms_iou_threshold,
            )
        else:
            pred_observation_boxes = None

        # ---------- Output and loss ----------
        output = ObservationClassifierOutput(
            observation_class_probs=mil_class_probs if mil_class_probs is not None else None,
            observation_class_preds=mil_class_preds if mil_class_preds is not None else None,
            target_observation_classes=target_observation_classes if target_observation_classes is not None else None,
            anatomy_observation_probs=regions_class_probs if regions_class_probs is not None else None,
            anatomy_observation_preds=regions_class_preds if regions_class_preds is not None else None,
            target_anatomy_observations=target_anatomy_observations,
            targets_with_anatomy_observations=targets_with_anatomy_observations,
            anatomy_detection=anatomy_detector_output,
            predicted_observation_boxes=pred_observation_boxes,
            target_observation_boxes=target_observation_bboxes
        )

        step_metrics = {}
        losses = {}
        if compute_loss:
            if self.per_region_loss_fn is not None and target_anatomy_observations is not None:
                assert regions_class_logits is not None
                # (N x A x C')
                pred_regions_class_logits = regions_class_logits[..., targets_with_anatomy_observations]
                pred_regions_class_logits = pred_regions_class_logits[region_mask, :]
                target_anatomy_observations = target_anatomy_observations[region_mask, :]
                losses['obser_anat_loss'] = self.per_region_loss_fn(pred_regions_class_logits, target_anatomy_observations.to(dtype=pred_regions_class_logits.dtype))

            if self.mil_loss is not None:
                assert mil_class_logits is not None
                losses['obser_mil_loss'] = self.mil_loss(mil_class_logits, target_observation_classes.to(dtype=mil_class_logits.dtype))

            if anatomy_detector_output is not None:
                losses['anat_detect_loss'] = anatomy_detector_output.loss
                step_metrics.update({f'anat_detect/{name}': value for name, value in anatomy_detector_output.step_metrics.items()})
            step_metrics.update(losses)
            output.step_metrics = step_metrics
            output.loss = sum(loss for loss in losses.values() if loss is not None)

        return output

    def plot(self, model_output: ObservationClassifierOutput, input: dict, target_dir: str, plot_local: bool = False, **kwargs) -> dict:
        x = input['x']
        sample_ids = input['sample_id']
        plots = {}
        if self.detector is not None and model_output.anatomy_detection is not None:
            anatomy_names = input['anatomy_names'] if 'anatomy_names' in input else self.config.learned_anatomy_names
            plots.update(self.detector.plot(model_output.anatomy_detection, 
                            input=dict(x=x, class_names=anatomy_names), 
                            target_dir=target_dir, prefix='anatomy', **kwargs))
        if model_output.predicted_observation_boxes is not None:
            observation_names = input['observation_class_names']
            plots['pathology_boxes'] = prepare_wandb_bbox_images(
                images=x, 
                preds=model_output.predicted_observation_boxes,
                targets=model_output.target_observation_boxes,
                class_names=observation_names,
                one_box_per_class=False)

            if plot_local:
                plot_and_save_img_bboxes(
                    target_dir=target_dir,
                    class_names=observation_names,
                    images=x,
                    target_boxes=model_output.target_observation_boxes,
                    predicted_boxes=model_output.predicted_observation_boxes,
                    prefix='pathology_boxes',
                    sample_ids=sample_ids,
                    one_box_per_class=False,
                )
        return plots


    def build_metrics(self, dataset_info: DatasetConfig) -> nn.Module:
        return ObservtationClassifierMetrics(
            observation_names=dataset_info.observation_names, 
            class_groups=dataset_info.reported_observation_groups,
            anatomy_names=dataset_info.anatomy_names if dataset_info.load_anatomy_boxes else None,
            has_observation_boxes=dataset_info.load_observation_boxes)


class ObservationClassifier(nn.Module):
    def __init__(self, 
        n_cls_hidden_layers: int, d_classifier_hidden: Optional[int], d: int, act: str, dropout: float, 
        learned_observation_groups: List[ObservationSet]):
        super().__init__()
        
        if d_classifier_hidden is None:
            d_classifier_hidden = d * 4

        assert learned_observation_groups is not None
        observation_names = [f'{ob_set.observation_set_name}/{name}' 
            for ob_set in learned_observation_groups 
            for name in ob_set.observation_names]
        self.observation_classifier = MLP(
            n_cls_hidden_layers + 1,
            d_in=d, d_out=len(observation_names), d_hidden=d_classifier_hidden,
            act=act,
            dropout=dropout, dropout_last_layer=False)
        observation_class_mapping = {name: [i] for i, name in enumerate(observation_names)}
        cross_dataset_observation_mappings: Dict[str, List[str]] = load_observation_mappings(trained_observations=observation_names)
        for mapped_observation, trained_observations in cross_dataset_observation_mappings.items():
            observation_class_mapping[mapped_observation] = [i for train_name in trained_observations for i in observation_class_mapping[train_name]]
        self.observation_class_mapping = observation_class_mapping
        self.trained_observation_names = observation_names
        
    def forward(self, 
        features: torch.Tensor, 
        observation_class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        observation_features: (N x C x d) or (N x A x C x d)
        observation_tokens: (C x d)
        :return: logits, probs, preds each of shape (N x C) or (N x A x C)
        """
        assert features.ndim >= 2
        *dims, D = features.shape
        C = len(observation_class_names)
        features = features.reshape(-1, D)

        # ((dims) x C)
        logits = self.observation_classifier(features)
        probs = logits.sigmoid()
        preds = logits > 0.0
        if observation_class_names != self.trained_observation_names:
            # --- map between training and evaluation classes ---
            # ((dims) x C')
            logits_mapped = torch.zeros(logits.shape[0], len(observation_class_names), device=features.device, dtype=logits.dtype)
            probs_mapped = torch.zeros(logits.shape[0], len(observation_class_names), device=features.device, dtype=logits.dtype) + 0.5
            preds_mapped = torch.zeros(logits.shape[0], len(observation_class_names), device=features.device, dtype=torch.bool)
            for i, cls_name in enumerate(observation_class_names):
                if cls_name in self.observation_class_mapping:
                    logits_mapped[:, i] = logits[:, self.observation_class_mapping[cls_name]].mean(dim=1)
                    probs_mapped[:, i] = probs[:, self.observation_class_mapping[cls_name]].mean(dim=1)
                    preds_mapped[:, i] = preds[:, self.observation_class_mapping[cls_name]].any(dim=1)
            logits = logits_mapped
            probs = probs_mapped
            preds = preds_mapped

        logits = logits.reshape(*dims, C)
        probs = probs.reshape(*dims, C)
        preds = preds.reshape(*dims, C)
        
        return logits, probs, preds


def load_observation_mappings(trained_observations: List[str]):
    observation_mappings = {}
    with open(os.path.join(RESOURCES_DIR, 'observation_mappings.csv'), 'r') as f:
        dict_reader = csv.DictReader(f)
        for row in dict_reader:
            source_name = row['source_name']
            target_names = row['target_name'].split(';')
            target_names = [name for name in target_names if name in trained_observations]
            if source_name not in observation_mappings and len(target_names) > 0:
                observation_mappings[source_name] = target_names
    return observation_mappings
