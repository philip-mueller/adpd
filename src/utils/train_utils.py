
from collections import defaultdict
from dataclasses import dataclass, field
import dataclasses
import glob
import logging
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Collection
import numpy as np
from timm.scheduler import CosineLRScheduler
from torch import optim
from omegaconf import MISSING, OmegaConf
import torch
from dataset.image_transform import TransformConfig

from dataset.datasets import DatasetConfig, build_dataloader


log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = MISSING

    model: Any = MISSING

    continue_from_checkpoint: Optional[str] = None

    train_dataset: Dict[str, DatasetConfig] = field(default_factory=dict)
    val_dataset: Dict[str, DatasetConfig] = field(default_factory=dict)
    transform: TransformConfig = MISSING
    train: bool = True
    evaluate: bool = True
    eval_mode: str = 'val'
    eval_datasets: Dict[str, DatasetConfig] = field(default_factory=dict)

    batch_size: int = MISSING
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: float = MISSING
    min_lr: float = MISSING
    warmup_lr: Optional[float] = MISSING
    warmup_steps: int = MISSING
    weight_decay: float = MISSING
    accumulation_steps: int = MISSING
    grad_clip_norm: Optional[float] = MISSING
    early_sopping_patience: Optional[int] = MISSING

    metric: str = MISSING
    metric_mode: str = MISSING

    val_freq: int = MISSING
    print_freq: int = MISSING
    num_workers: int = MISSING
    prefetch: bool = MISSING
    device: str = MISSING
    debug: bool = False
    save_components: List[str] = field(default_factory=list)

    plot_wandb: bool = True
    plot_local: bool = False
    plot_val_batches: int = 0
    plot_val_arguments: Dict[str, Any] = field(default_factory=dict)


def build_optimizer(model: 'BaseModel', config: ExperimentConfig):
    return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                       weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: ExperimentConfig):
    num_steps = int(config.max_steps)
    warmup_steps = int(config.warmup_steps)

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=config.min_lr,
        warmup_lr_init=config.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )


def build_dataloders_for_training(config: ExperimentConfig):
    assert len(config.train_dataset) == 1, config.train_dataset
    train_dataset = list(config.train_dataset.values())[0]
    train_dl = build_dataloader(
        mode='train',
        config=train_dataset,
        pixel_mean=train_dataset.pixel_mean,
        pixel_std=train_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch)
    assert len(config.val_dataset) == 1
    val_dataset = list(config.val_dataset.values())[0]
    val_dl = build_dataloader(
        mode='val',
        config=val_dataset,
        pixel_mean=train_dataset.pixel_mean,
        pixel_std=train_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch)

    return train_dl, val_dl


def build_dataloaders_for_eval(config: ExperimentConfig, eval_datasets: Optional[Dict[str, DatasetConfig]] = None):
    assert len(config.train_dataset) == 1
    train_dataset = list(config.train_dataset.values())[0]
    if eval_datasets is None:
        eval_datasets = config.eval_datasets
    for name, dataset in eval_datasets.items():
        dataloader = build_dataloader(
            mode=config.eval_mode,
            config=dataset,
            pixel_mean=train_dataset.pixel_mean,
            pixel_std=train_dataset.pixel_std,
            transform=config.transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch=config.prefetch)

        yield name, dataset, dataloader


def get_best_results(results, best_results, config: ExperimentConfig):
    if best_results is None:
        return results, True
    assert config.metric_mode in ('min', 'max')
    best_value = best_results['val_metric']
    value = results['val_metric']
    if (value > best_value and config.metric_mode == 'max') or \
            (value < best_value and config.metric_mode == 'min'):
        return results, True
    else:
        return best_results, False


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AvgDictMeter:
    def __init__(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            if value is None:
                continue
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}



class TensorDataclassMixin:
    def __init__(self):
        super(TensorDataclassMixin, self).__init__()
        assert dataclasses.is_dataclass(self), f'{type(self)} has to be a dataclass to use TensorDataclassMixin'

    def apply(self, tensor_fn: Callable[[torch.Tensor], torch.Tensor], ignore=None):
        def apply_to_value(value):
            if value is None:
                return None
            elif isinstance(value, torch.Tensor):
                return tensor_fn(value)
            elif isinstance(value, list):
                return [apply_to_value(el) for el in value]
            elif isinstance(value, tuple):
                return tuple(apply_to_value(el) for el in value)
            elif isinstance(value, dict):
                return {key: apply_to_value(el) for key, el in value.items()}
            elif isinstance(value, TensorDataclassMixin):
                return value.apply(tensor_fn)
            else:
                return value

        def apply_to_field(field: dataclasses.Field):
            value = getattr(self, field.name)
            if ignore is not None and field.name in ignore:
                return value
            else:
                return apply_to_value(value)

        return self.__class__(**{field.name: apply_to_field(field) for field in dataclasses.fields(self)})

    def to(self, device, *args, non_blocking=True, **kwargs):
        return self.apply(lambda x: x.to(device, *args, non_blocking=non_blocking, **kwargs))

    def view(self, *args):
        return self.apply(lambda x: x.view(*args))

    def detach(self):
        return self.apply(lambda x: x.detach())
    
    def unsqueeze(self, dim):
        return self.apply(lambda x: x.unsqueeze(dim))
    
    def squeeze(self, dim):
        return self.apply(lambda x: x.squeeze(dim))

    def __getitem__(self, *args):
        return self.apply(lambda x: x.__getitem__(*args))

    def to_dict(self):
        return dataclasses.asdict(self)
        

def to_device(data, device: str, non_blocking=True):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        if device == 'cpu':
            non_blocking = False
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, Mapping):
        return {key: to_device(data[key], device, non_blocking=non_blocking) for key in data}
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [to_device(d, device, non_blocking=non_blocking) for d in data]
    elif isinstance(data, str):
        return data
    elif isinstance(data, TensorDataclassMixin):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError(type(data))


def save_training_checkpoint(model: 'BaseModel', optimizer, lr_scheduler, scaler, results,
                             best_results, config, step, saved_components=(), is_best=False):
    saved_components = () if saved_components is None else saved_components
    saved_states = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'amp': scaler.state_dict(),
        'step': step,
        'results': results,
        'best_results': best_results,
        'experiment_config': OmegaConf.to_container(config)
    }

    # Save the current model
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{step:09d}.pth')
    model.save_model(checkpoint_path, **saved_states)
    for component_name in saved_components:
        os.makedirs(os.path.join('checkpoints', component_name), exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', component_name, f'checkpoint_{step:09d}.pth')
        model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Save as best model
    if is_best:
        checkpoint_path = os.path.join('checkpoints', 'checkpoint_best.pth')
        model.save_model(checkpoint_path, **saved_states)
        for component_name in saved_components:
            checkpoint_path = os.path.join('checkpoints', component_name, 'checkpoint_best.pth')
            model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Remove the previous model
    if step > 0:
        for chkpt_path in glob.glob(os.path.join('checkpoints', f'checkpoint_*.pth')):
            if not chkpt_path.endswith(f'checkpoint_{step:09d}.pth') and not chkpt_path.endswith('checkpoint_best.pth'):
                os.remove(chkpt_path)

