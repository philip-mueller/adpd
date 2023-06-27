import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Dict, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch import autocast
from torchmetrics import AUROC
from tqdm import tqdm
import wandb

                            

from dataset.datasets import DatasetConfig
from settings import MODELS_DIR, WANDB_ENTITY, WANDB_PROJECT
from utils.model_utils import get_model_dir, get_run_dir, get_wandb_run_from_model_name, load_model_by_name
from utils.train_utils import ExperimentConfig, build_dataloaders_for_eval, seed_everything
from train import validate

log = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    model_name: str = MISSING
    run_name: Optional[str] = None

    eval_mode: str = 'val'
    eval_datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    config_override: Dict[str, Any] = field(default_factory=dict)

    update_wandb: bool = MISSING

    plot_wandb: bool = False
    plot_local: bool = False
    plot_val_batches: int = 0
    plot_val_arguments: Dict[str, Any] = field(default_factory=dict)

    device: str = MISSING
    num_workers: int = MISSING
    prefetch: bool = MISSING
    seed: int = MISSING

    debug: bool = False


def evaluate(config: EvaluationConfig):
    """"""""""""""""""""""""""""""" Setup """""""""""""""""""""""""""""""
    log.info(f'Starting Evaluation of {config.model_name}')
    model_dir = get_run_dir(get_model_dir(config.model_name), config.run_name)
    model, checkpoint_dict = load_model_by_name(
        config.model_name,
        run_name=config.run_name,
        load_best=True,
        return_dict=True,
        config_override=config.config_override,
    )
    train_experiment_config: ExperimentConfig = OmegaConf.create(checkpoint_dict['experiment_config'])
    step = checkpoint_dict['step']
    log.info(f'Evaluating step {step}')

    seed_everything(config.seed)
    torch.backends.cudnn.benchmark = True
    model = model.to(device=config.device)
    log.info(f'Using {config.device}')
    model.eval()
    
    wandb_run = None
    if not config.debug:
        try:
            wandb_run = get_wandb_run_from_model_name(
                config.model_name,
                run_name=config.run_name
            )
            assert wandb_run.state != 'running', 'Run is still running'
        except Exception as e:
            log.error(f'Could not get wandb run: {e}\n'
                      'Evaluating without saving to wandb.')
            wandb_run = None

        if config.plot_wandb:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f'plot_{config.model_name}',
                tags=[type(model).__name__],
                dir='.',
                resume=False, #'must' if config.resume else False,
                settings=wandb.Settings(start_method='thread'), 
            )
    
    all_results = {}
    train_experiment_config.prefetch = config.prefetch
    train_experiment_config.num_workers = config.num_workers
    train_experiment_config.device = config.device
    train_experiment_config.debug = config.debug
    train_experiment_config.plot_wandb = config.plot_wandb
    train_experiment_config.plot_val_batches = config.plot_val_batches
    train_experiment_config.plot_val_arguments = config.plot_val_arguments
    for dataset_name, dataset, dataloader in build_dataloaders_for_eval(train_experiment_config, eval_datasets=config.eval_datasets):
        eval_prefix = f'{config.eval_mode}_{dataset_name}'
        log.info(f'Evaluating on {dataset_name} ({config.eval_mode})')
        metrics = model.build_metrics(dataset_info=dataloader.dataset.dataset_info)
        metrics = metrics.to(config.device)
        eval_results = validate(model=model, val_metrics=metrics, val_dataloader=dataloader, model_dir=model_dir, config=config, step=0, prefix=eval_prefix)
        all_results.update(eval_results)
        if wandb_run is not None and config.update_wandb:
            wandb_run.summary.update(eval_results)
            wandb_run.config.update({eval_prefix: {
                'dataset': dataloader.dataset.dataset_info,
                'config_override': config.config_override
            }})

    log.info('Finished evaluating')
    log.info(f'Results: {pformat(all_results)}')
    


@hydra.main(config_path="../conf", config_name="evaluate")
def run_evaluate(config):
    evaluate(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="EvaluationConfig", node=EvaluationConfig)
    OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver(
        "ifel",
        lambda flag, val_true, val_false: val_true if flag else val_false
    )
    run_evaluate()
