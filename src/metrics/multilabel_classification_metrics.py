from typing import Dict, List
from torch import nn
import torch
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelMatthewsCorrCoef, MultilabelRankingAveragePrecision, MulticlassAUROC

class MultiLabelClassificationMetrics(nn.Module):
    def __init__(self, class_names: List[str], class_groups: Dict[str, List[str]] = None) -> None:
        super().__init__()
        self.class_names = [cls_name.replace(' ', '_').replace('/', '-') for cls_name in class_names]
        if class_groups is not None:
            self.class_groups = {group_name: [cls_name.replace(' ', '_').replace('/', '-') for cls_name in group_cls_names]
                                    for group_name, group_cls_names in class_groups.items()}
        else:
            self.class_groups = {}

        self.auroc = MultilabelAUROC(num_labels=len(class_names), average='none')
        self.f1 = MultilabelF1Score(num_labels=len(class_names), average='none')
        self.mcc_macro = MultilabelMatthewsCorrCoef(num_labels=len(class_names), average='macro')
        self.label_rank_avg_prec = MultilabelRankingAveragePrecision(num_labels=len(class_names))

    def add(self, preds, pred_probs, target):
        self.auroc.update(pred_probs, target)
        self.f1.update(preds, target)
        self.mcc_macro.update(preds, target)
        self.label_rank_avg_prec.update(pred_probs, target)

    def reset(self):
        self.auroc.reset()
        self.f1.reset()
        self.mcc_macro.reset()
        self.label_rank_avg_prec.reset()

    def compute(self, prefix = '') -> dict:
        auroc_values = self.auroc.compute()
        f1_values = self.f1.compute()

        auroc_group_values = {
            group_name: 
                torch.stack([auroc_value 
                 for cls_name, auroc_value 
                 in zip(self.class_names, auroc_values) 
                 if cls_name in group_cls_names]).mean()
            for group_name, group_cls_names in self.class_groups.items()
        }
        return {
            **{f'{prefix}auroc_cls/{cls_name}': value 
                for cls_name, value in zip(self.class_names, auroc_values)},
            **{f'{prefix}auroc_group/{group_name}': value
                for group_name, value in auroc_group_values.items()},
            prefix+'auroc_macro': auroc_values.mean(),
            **{f'{prefix}f1_cls/{cls_name}': value 
                for cls_name, value in zip(self.class_names, f1_values)},
            prefix+'f1_macro': f1_values.mean(),
            prefix+'mcc_macro': self.mcc_macro.compute(),
            prefix+'label_rank_ap': self.label_rank_avg_prec.compute()
        }
