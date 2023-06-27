
from dataclasses import dataclass
from typing import Dict, List, Optional
from omegaconf import MISSING

from torch import FloatTensor, Tensor
from torch import nn
from metrics.detection_metrics import DynamicSetDetectionMetrics, FixedSetDetectionMetrics
from metrics.multilabel_classification_metrics import MultiLabelClassificationMetrics
from model.detector import TokenDetectorOutput

from utils.model_utils import BaseModelOutput


@dataclass
class ObservationClassifierOutput(BaseModelOutput):
    # Observation Classificaiton (image-level)
    # (N x C)
    observation_class_probs: FloatTensor = MISSING
    # (N x C)
    observation_class_preds: Tensor = MISSING
    # (N x C)
    target_observation_classes: Tensor = MISSING

    # Anatomy Observation Detection
    # (N x A x C)
    anatomy_observation_probs: Optional[FloatTensor] = None
    # (N x A x C)
    anatomy_observation_preds: Optional[Tensor] = None
    # (N x A x C)
    target_anatomy_observations: Optional[Tensor] = None
    # (C), True for each class with region-level annotations
    targets_with_anatomy_observations: Optional[Tensor] = None

    # Observation Detection
    predicted_observation_boxes: Optional[List[Tensor]] = None
    target_observation_boxes: Optional[List[Tensor]] = None

    # Anatomy Detection
    anatomy_detection: Optional[TokenDetectorOutput] = None

class ObservtationClassifierMetrics(nn.Module):
    def __init__(self, 
    observation_names: List[str], 
    class_groups: Dict[str, List[str]] = None,
    anatomy_names=None, 
    has_observation_boxes=False) -> None:
        super().__init__()

        self.observation_classification_metrics = MultiLabelClassificationMetrics(observation_names, class_groups=class_groups)
        self.anatomy_detection_metrics = FixedSetDetectionMetrics(anatomy_names) if anatomy_names is not None else None
        
        if has_observation_boxes:
            self.observation_detection_metrics = DynamicSetDetectionMetrics(observation_names)
        else:
            self.observation_detection_metrics = None
        self.has_cls_samples = False

    def add(self, model_output: ObservationClassifierOutput):
        if model_output.observation_class_probs is not None:
            self.has_cls_samples = True
            self.observation_classification_metrics.add(
                preds=model_output.observation_class_preds,
                pred_probs=model_output.observation_class_probs,
                target=model_output.target_observation_classes)
        if model_output.anatomy_detection is not None and self.anatomy_detection_metrics is not None:
            self.anatomy_detection_metrics.add(model_output.anatomy_detection)

        if self.observation_detection_metrics is not None:
            assert model_output.predicted_observation_boxes is not None
            self.observation_detection_metrics.add(
                predicted_observation_boxes=model_output.predicted_observation_boxes,
                target_observation_boxes=model_output.target_observation_boxes)

    def reset(self):
        self.observation_classification_metrics.reset()
        if self.anatomy_detection_metrics is not None:
            self.anatomy_detection_metrics.reset()
        if self.observation_detection_metrics is not None:
            self.observation_detection_metrics.reset()
        self.has_cls_samples = False

    def compute(self, prefix = ''):
        if self.has_cls_samples:
            metrics = self.observation_classification_metrics.compute(prefix=prefix+'patho_cls/')
        else:
            metrics = {}
        if self.anatomy_detection_metrics is not None:
            metrics.update(self.anatomy_detection_metrics.compute(prefix=prefix+'anat_detect/'))
        if self.observation_detection_metrics is not None:
            metrics.update(self.observation_detection_metrics.compute(prefix=prefix+'patho_detect/'))
        return metrics
    