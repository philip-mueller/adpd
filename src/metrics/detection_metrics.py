
from typing import List, Optional, Sequence, Tuple
from torch import Tensor, nn
import torch
from torchmetrics import MetricCollection, Precision, Recall
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou, box_convert
import numpy as np
from mean_average_precision import MeanAveragePrecision2d

from model.detector import TokenDetectorOutput

class FixedSetDetectionMetrics(nn.Module):
    def __init__(self, class_names: List[str]) -> None:
        super().__init__()

        self.class_names = class_names # note: no-finding should already be excluded
        self.register_buffer('count', torch.tensor(0.))  # (1)
        self.register_buffer('num_boxes_sum', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('samplemircoIoU_sum', torch.tensor(0.))  # (1)
        self.register_buffer('label_intersection_sum', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('label_union_sum', torch.zeros(len(class_names)))  # (C)

        self.prec_recall = nn.ModuleDict({
            cls_name: MetricCollection({'precision': Precision(task='binary'), 'recall': Recall(task='binary')})
            for cls_name in class_names
        })
        #self.map = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox', iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))

    def add(self, model_output: TokenDetectorOutput):
        predicted_boxes = model_output.boxes  # (N x A x 4) (x_c, y_c, w, h) format
        predicted_mask = model_output.box_mask  # (N x A) (x_c, y_c, w, h) format
        target_boxes = model_output.target_boxes
        target_mask = model_output.target_box_mask
        assert target_boxes is not None and target_mask is not None

        N, A, _ = target_boxes.shape
        self.count += N
        self.num_boxes_sum += predicted_mask.sum(0)

        # --- IoU metrics ---
        # Compute intersections and unions
        # (N x A)
        intersection_area, union_area = batched_box_intersection_and_union(
            predicted_boxes, target_boxes,
            box_mask_1=predicted_mask, box_mask_2=target_mask)
        # For sample micro IoU
        samples_micro_ious = intersection_area.sum(1) / union_area.sum(1).clamp_min(1e-7) # (N)
        self.samplemircoIoU_sum += samples_micro_ious.sum(0)  # (1)
        # For class and macro IoU
        self.label_intersection_sum += intersection_area.sum(0)  # (C)
        self.label_union_sum += union_area.sum(0)  # (C)
    
        # --- Precision, Recall metrics ---
        for cls_name, pred, target in zip(self.class_names, predicted_mask.T, target_mask.T):
            self.prec_recall[cls_name].update(pred, target)

        # --- mAP ---
        return 
        # (A)
        
    def compute(self, prefix='') -> dict:
        avg_num_boxes = (self.num_boxes_sum / self.count).sum()

        sample_micro_iou = self.samplemircoIoU_sum / self.count  # (1)
        class_ious = self.label_intersection_sum / self.label_union_sum  # (C)
        macro_iou = class_ious.mean()

        precs_recalls = [self.prec_recall[cls_name].compute() for cls_name in self.class_names]
        precisions = torch.stack([prec_recall['precision'] for prec_recall in precs_recalls])  # (C)
        recalls = torch.stack([prec_recall['recall'] for prec_recall in precs_recalls])  # (C)
        f1s = 2 * precisions * recalls / (precisions + recalls).clamp_min(1e-7) # (C)
        #map_results = self.map.compute()
        return {
            prefix+'num_boxes': avg_num_boxes,
            prefix+'IoU_macro': macro_iou,
            prefix+'IoU_sample_micro': sample_micro_iou,
            prefix+'prec_macro': precisions.mean(),
            prefix+'recall_macro': recalls.mean(),
            prefix+'f1_macro': f1s.mean(),
            **{f'{prefix}IoU_classes/{cls_name}': value for cls_name, value in zip(self.class_names, class_ious)},
            #**{f'{prefix}precision_classes/{cls_name}': prec_recall['precision'] 
            #    for cls_name, prec_recall in zip(self.class_names, precs_recalls)},
            #**{f'{prefix}recall_classes/{cls_name}': prec_recall['recall'] 
            #    for cls_name, prec_recall in zip(self.class_names, precs_recalls)},
            #prefix+'mAP': map_results['map'],
            #prefix+'mAP@50': map_results['map_50'],
        }

    def reset(self):
        self.count.zero_()
        self.num_boxes_sum.zero_()
        self.samplemircoIoU_sum.zero_()
        self.label_intersection_sum.zero_()
        self.label_union_sum.zero_()
        #self.map.reset()

        for metric in self.prec_recall.values():
            metric.reset()


class DynamicSetDetectionMetrics(nn.Module):
    def __init__(self, class_names: List[str]) -> None:
        super().__init__()
        self.class_names = class_names
        self.threshold = 0.7 
        self.map = BBoxMeanAPMetric(class_names=class_names, iou_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), extra_reported_thresholds=(0.1, 0.3, 0.5, 0.7))
        self.acc_iou = PRAtIoUMetric(class_names=class_names, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))

    def add(self, predicted_observation_boxes: Optional[List[Tensor]], target_observation_boxes: Optional[List[Tensor]]):
        """
        :param predicted_observation_boxes: List (N) of tensors (M_i x 6) in the (x_c, y_c, w, h, cls, score) format
        :param target_observation_boxes: List (N) of tensors (M_i x 5) in the (x_c, y_c, w, h, cls) format
        """
        self.map.add(predicted_observation_boxes, target_observation_boxes)

        filtered_predicted_observation_boxes = [
            sample_boxes[sample_boxes[:, 5] > self.threshold]
            for sample_boxes in predicted_observation_boxes
        ]

        self.acc_iou.add(filtered_predicted_observation_boxes, target_observation_boxes)

    def compute(self, prefix: str=''):
        map_results = self.map.compute()
        acc_iou_results = self.acc_iou.compute()
        return {
            **{f'{prefix}{key}': value for key, value in map_results.items()},
            **{f'{prefix}{key}': value for key, value in acc_iou_results.items()},
        }

    def reset(self):
        self.map.reset()
        self.acc_iou.reset()

@torch.jit.script
def batched_box_intersection_and_union(boxes_1, boxes_2, box_mask_1: Optional[Tensor]=None, box_mask_2: Optional[Tensor]=None):
    """
    :param boxes_1: (... x 4) in the (x_c, y_c, w, h) format
    :param boxes_2: (... x 4) in the (x_c, y_c, w, h) format
    :param box_mask_1: (...)
    :param box_mask_2: (...)
    :return (...)
    """
    wh_1 = boxes_1[..., 2:4]
    areas_1 = wh_1[..., 0] * wh_1[..., 1]  # (...)
    x1y1_1 = boxes_1[..., :2] - 0.5 * wh_1  # (... x 2)
    x2y2_1 = boxes_1[..., :2] + 0.5 * wh_1  # (... x 2)
    if box_mask_1 is not None:
        areas_1 = box_mask_1 * areas_1

    wh_2 = boxes_2[..., 2:4]
    areas_2 = wh_2[..., 0] * wh_2[..., 1]  # (...)
    x1y1_2 = boxes_2[..., :2] - 0.5 * wh_2 # (... x 2)
    x2y2_2 = boxes_2[..., :2] + 0.5 * wh_2  # (... x 2)
    if box_mask_2 is not None:
        areas_2 = box_mask_2 * areas_2

    xx1yy1 = torch.maximum(x1y1_1, x1y1_2)  # (... x 2)
    xx2yy2 = torch.minimum(x2y2_1, x2y2_2)  # (... x 2)
    intersection_wh = (xx2yy2 - xx1yy1).clamp_min(0.)  # (... x 2)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (...)
    if box_mask_1 is not None:
        intersection_area = box_mask_1 * intersection_area
    if box_mask_2 is not None:
        intersection_area = box_mask_2 * intersection_area

    union_area = areas_1 + areas_2 - intersection_area  # (...)
    return intersection_area, union_area  # (...)
    

class PRAtIoUMetric(nn.Module):
    def __init__(self, class_names: List[str], thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), obj_thres=None):
        super(PRAtIoUMetric, self).__init__()
        self.class_names = class_names  # note: no-finding should already be excluded
        self.obj_thres = obj_thres
        self.register_buffer('n_tp', torch.zeros(len(class_names), len(thresholds)))  # (C, n_thres)
        self.register_buffer('n_tn', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('n_gt', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('n_pred', torch.zeros(len(class_names)))  # (C)
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.register_buffer('n', torch.tensor(0))
        self.threshold_values = thresholds

    def reset(self):
        self.n.zero_()
        self.n_tp.zero_()
        self.n_tn.zero_()
        self.n_gt.zero_()
        self.n_pred.zero_()

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        predictions: List (N) of (n_pred, 6) tensor in (x_c, y_c, w, h, class, score) format
        targets: List (N) of (n_gt, 5) tensor in (x_c, y_c, w, h, class) format
        """
        N = len(predictions)
        C = len(self.class_names)
        self.n += N
        for c in range(C):
            for pred, target in zip(predictions, targets):
                pred = pred.to(device=self.n_tp.device)
                target = target.to(device=self.n_tp.device)
                # filter class
                pred = pred[pred[:, 4].long() == c]
                if self.obj_thres is not None:
                    pred = pred[pred[:, 5] >= self.obj_thres]
                target = target[target[:, 4].long() == c]
                n_pred = pred.shape[0]
                n_gt = target.shape[0]
                self.n_pred[c] += n_pred
                self.n_gt[c] += n_gt

                if n_gt == 0 or n_pred == 0:
                    if n_gt == n_pred:  # both zero
                        self.n_tn[c] += 1
                    continue

                # convert xcycwh -> x1y1x2y2
                pred_boxes = box_convert(pred[:, :4], 'cxcywh', 'xyxy')
                target_boxes = box_convert(target[:, :4], 'cxcywh', 'xyxy')
                ious = box_iou(pred_boxes, target_boxes)  # (n_pred x n_target)
                # select best matching pred box for each target
                ious = ious.amax(0)  # (n_target)
                tp = (ious[:, None] > self.thresholds[None, :]).sum(0)  # (n_thres)
                self.n_tp[c] += tp

    def compute(self):
        tp = self.n_tp.cpu().numpy()
        tn = self.n_tn[:, None].cpu().numpy()
        fp = self.n_pred[:, None].cpu().numpy() - tp
        fn = self.n_gt[:, None].cpu().numpy() - tp

        acc = (tp + tn) / (tp + tn + fp + fn)
        afp = fp / self.n.cpu().float().numpy()

        metrics = {}

        for t, thres in enumerate(self.threshold_values):
            #for c, class_name in enumerate(self.class_names):
            #    metrics[f'acc_classes/acc@{thres}_{class_name}'] = acc[c, t]
            #    metrics[f'afp_classes/afp@{thres}_{class_name}'] = afp[c, t]

            metrics[f'acc_thres/acc@{thres}'] = np.mean([acc[c, t] for c in range(len(self.class_names))])
            metrics[f'afp_thres/afp@{thres}'] = np.mean([afp[c, t] for c in range(len(self.class_names))])

        return metrics

class BBoxMeanAPMetric():
    """
    Compute mean Average Precision for bounding boxes with
    https://github.com/bes-dev/mean_average_precision

    For COCO, select iou_thresholds = np.arange(0.5, 1.0, 0.05)
    For Pascal VOC, select iou_thresholds = 0.5
    """
    def __init__(self, class_names: int, iou_thresholds=(0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
                 extra_reported_thresholds: Tuple = (0.5, 0.75)):
        super(BBoxMeanAPMetric, self).__init__()
        self.iou_thresholds = np.array(iou_thresholds)
        self.extra_reported_thresholds = []
        for extra_thres in extra_reported_thresholds:
            found_close = False
            for thres in self.iou_thresholds:
                if np.isclose(thres, extra_thres):
                    self.extra_reported_thresholds.append(thres)
                    found_close = True
            if not found_close:
                raise ValueError(f'{extra_thres} not found in {self.iou_thresholds}')
        self.metric = MeanAveragePrecision2d(len(class_names))
        self.class_names = class_names

    def reset(self):
        self.metric.reset()

    def add(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Add a batch of predictions and targets to the metric.
        N samples with each M bounding boxes of C classes.

        src.model.model_interface.ObjectDetectorPrediction.box_prediction_hard
        has the correct format for predictions.

        :param predictions: List of N predictions, each a tensor of shape (M x 6)
                            (x, y, w, h, class_id, confidence)
        :param targets: List of N targets, each a tensor of shape (M x 5)
                        (x, y, w, h, class_id)
        """
        for predicted, target in zip(predictions, targets):
            predicted_np = predicted.detach().cpu().numpy().copy()
            target_np = target.detach().cpu().numpy().copy()
            assert predicted_np.shape[1] == 6
            assert target_np.shape[1] == 5

            # Convert from [xc, yc, w, h, class_id, confidence]
            # to [xmin, ymin, xmax, ymax, class_id, confidence]
            preds = np.zeros((len(predicted_np), 6))
            preds[:, 0:2] = predicted_np[:, :2] - predicted_np[:, 2:4] / 2
            preds[:, 2:4] = predicted_np[:, :2] + predicted_np[:, 2:4] / 2
            preds[:, 4:6] = predicted_np[:, 4:6]

            # Convert from [xc, yc, w, h, class_id]
            # to [xmin, ymin, xmax, ymax, class_id, difficult]
            gt = np.zeros((len(target_np), 7))
            gt[:, 0:2] = target_np[:, :2] - target_np[:, 2:4] / 2
            gt[:, 2:4] = target_np[:, :2] + target_np[:, 2:4] / 2
            gt[:, 4] = target_np[:, 4]

            # --- correction as the metric implementation assumes pixels and therefore computes width/height offset by 1. ---
            preds[:, 0:4] *= 1000 
            preds[:, 2:4] -= 1.0
            gt[:, 0:4] *= 1000 
            gt[:, 2:4] -= 1.0
            # --- end correction ---

            self.metric.add(preds, gt)

    def compute(self):
        computed_metrics = self.metric.value(iou_thresholds=self.iou_thresholds,
                                             mpolicy="soft",
                                             recall_thresholds=np.arange(0., 1.01, 0.01))
        metrics = {'mAP': computed_metrics['mAP']}

        for c, class_name in enumerate(self.class_names):
            metrics[f'mAP_classes/{class_name}'] = np.mean([computed_metrics[t][c]['ap'] for t in self.iou_thresholds])
            for t in self.extra_reported_thresholds:
                metrics[f'mAP@{t}_classes/{class_name}'] = computed_metrics[t][c]['ap']
            
        if self.extra_reported_thresholds is not None:
            for t in self.extra_reported_thresholds:
                metrics[f'mAP_thres/mAP@{t}'] = np.mean([computed_metrics[t][c]['ap'] for c in range(len(self.class_names))])
                
        return metrics
