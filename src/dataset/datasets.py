from abc import abstractmethod
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import repeat
import logging
from multiprocessing import Pool
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from omegaconf import MISSING
import pandas as pd

import albumentations as A
import numpy as np
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset, default_collate


import tqdm

from dataset.image_transform import TransformConfig, build_transform, convert_bboxes_pixelx1y1x2y2_to_relxcyxwh
from dataset.mimic_cxr_datasets import load_mimic_cxr_datafile
from dataset.nih_cxr import load_cxr14_dataset
from utils.data_utils import load_pil_gray

log = logging.getLogger(__name__)


def preload(files: Sequence, load_fn: Callable = load_pil_gray,
            num_processes: int = min(12, os.cpu_count())) -> List:
    """
    Multiprocessing to load all files to RAM fast.

    :param files: List of file paths.
    :param load_fn: Function to load the image.
    :param num_processes: Number of processes to use.
    """
    with Pool(num_processes) as pool:
        results = pool.map(load_fn, files)
    return results


@dataclass
class ObservationSet:
    observation_set_name: str = MISSING
    observation_names: List[str] = MISSING

    has_bboxes: bool = False
    has_anatomy_observations: bool = False
    anatomy_observation_set_name: Optional[str] = None

    uncertain_pos: bool = True

@dataclass
class DatasetConfig:
    name: str = MISSING

    # Normalization parameters
    pixel_mean: List[float] = MISSING
    pixel_std: List[float] = MISSING

    observation_sets: Optional[List[ObservationSet]] = None
    reported_observation_groups: Any = None
    anatomy_names: Optional[List[str]] = None

    load_anatomy_boxes: bool = True
    load_observation_classes: bool = True
    load_anatomy_observations: bool = False
    load_observation_boxes: bool = False

    load_memmap: bool = True

    @property
    def n_anatomy_boxes(self) -> int:
        anatomy_names = self.anatomy_names
        return len(anatomy_names) if anatomy_names is not None else 0

    @property
    def has_anatomy_boxes(self) -> bool:
        return self.n_anatomy_boxes > 0

    @property
    def has_observation_boxes(self) -> bool:
        return any(ob_set.has_bboxes for ob_set in self.observation_sets)

    @property
    def has_anatomy_observations(self) -> bool:
        return any(ob_set.has_anatomy_observations for ob_set in self.observation_sets)

    @property
    def all_observation_name_pairs(self) ->  Optional[List[Tuple[ObservationSet, str]]]:
        if self.observation_sets is None or len(self.observation_sets) == 0:
            return None
        observation_names = [(ob_set, name) for ob_set in self.observation_sets for name in ob_set.observation_names]
        if len(observation_names) == 0:
            return None
        else:
            return observation_names

    @property
    def observation_names(self) -> Optional[List[str]]:
        all_observation_name_pairs = self.all_observation_name_pairs
        if all_observation_name_pairs is None:
            return None
        return [f'{ob_set.observation_set_name}/{name}' for ob_set, name in all_observation_name_pairs]
        
    @property
    def observation_names_with_anatomy_obs(self) -> Optional[List[str]]:
        if not self.has_anatomy_observations:
            return None
        all_observation_name_pairs = self.all_observation_name_pairs
        if all_observation_name_pairs is None:
            return None
        return [f'{ob_set.observation_set_name}/{name}' for ob_set, name in all_observation_name_pairs if ob_set.has_anatomy_observations]

    @property
    def has_observation_classes(self) -> bool:
        return self.n_observation_classes > 0

    @property
    def n_observation_classes(self) -> int:
        observation_names = self.observation_names
        return len(observation_names) if observation_names is not None else 0


class PrefetchDataset(Dataset):
    def __init__(self,
                 config: DatasetConfig,
                 mode: str,
                 image_size,
                 prefetch: bool = True,
                 transform: Optional[Callable] = None) -> None:
        super().__init__()

        self.dataset_info = config
        self.dataset_name = config.name
        self.sample_ids, self.load_indices, self.load_fn = self._load_dataset(self.dataset_info, mode=mode, image_size=image_size)

        if prefetch:
            from time import perf_counter
            log.info(f"Prefetching {len(self.image_paths)} images")
            start = perf_counter()
            self.images = preload(
                self.load_indices,
                load_fn=self.load_fn,
            )
            log.info(f'Prefetching images took {perf_counter() - start:.2f}s')

        self.prefetch = prefetch
        self.transform = transform

    @abstractmethod
    def _load_dataset(self, config: DatasetConfig, mode: str):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_img(self, index: int):
        if self.prefetch:
            img = self.images[index]
        else:
            img = self.load_fn(self.load_indices[index])
        #img = np.array(img, dtype=np.float32) / 255.
        return img


def parse_bbox(bbox: Union[str, Tuple, List]) -> np.ndarray:
    """
    tuple_str: "(x1, y1, x2, y2)"
    """
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)
    assert isinstance(bbox, (tuple, list))
    assert len(bbox) == 4
    return np.array(bbox, dtype=np.float32)
    #assert tuple_str[0] == '(' and tuple_str[-1] == ')'
    #return np.array([float(val.strip()) if len(val.strip()) > 0 else float('nan') for val in tuple_str[1:-2].split(',')])


def parse_and_filter_bbox_list(bbox_list: str, class_map: Dict[str, int]) -> List[np.ndarray]:
    if isinstance(bbox_list, str):
        bbox_list = ast.literal_eval(bbox_list)
    assert isinstance(bbox_list, (tuple, list))
    assert all(isinstance(bbox, (tuple, list)) for bbox in bbox_list)
    assert all(len(bbox) == 5 for bbox in bbox_list)
    assert all(isinstance(cls_name, str) for *_, cls_name in bbox_list)
    bbox_list = [[*bbox_coords, class_map[cls_name]] for *bbox_coords, cls_name in bbox_list if cls_name in class_map]
    return np.array(bbox_list, dtype=np.float32) if len(bbox_list) > 0 else np.zeros((0, 5), dtype=np.float32)


class ImageAnatomyClassificationDataset(PrefetchDataset):
    def _load_dataset(self, config: DatasetConfig, mode: str, image_size):
        # --- Load the dataset metadata ---
        if config.name.startswith('mimic_cxr-'):
            data_df, load_indices, load_fn, loaded_size = load_mimic_cxr_datafile(config.name, mode, image_size=image_size, load_memmap=config.load_memmap)
        elif config.name =='nih_cxr':
            data_df, load_indices, load_fn = load_cxr14_dataset(mode)
            loaded_size = None
        else:
            raise ValueError(f'Unknwon dataset {config.name}. Currently only MIMIC CXR datasets (with prefix "mimic_cxr-") are supported')
        sample_ids = data_df.sample_id.to_list()
        N = len(sample_ids)

         # --- Load the observation (target) classes, possible from several observation sets ---
        def _convert_labels(labels: pd.Series, uncertain_label=1, blank_label=0):
            # convert uncertain (-1) and blank (-2) labels to the given values
            labels = labels.fillna(value=-2.0)  # -2.0 = blank
            labels = labels.to_numpy(dtype=np.long)
            labels[labels == -1] = uncertain_label
            labels[labels == -2] = blank_label
            assert (labels >= 0).all() and (labels <= 1).all()
            return labels
        if config.has_observation_classes and config.load_observation_classes:
            all_observations = []
            for ob_set, ob_name in config.all_observation_name_pairs:
                binary_observation = _convert_labels(
                    data_df[f'{ob_set.observation_set_name}_{ob_name}'], 
                    uncertain_label=1 if ob_set.uncertain_pos else 0)
                binary_observation = np.array(binary_observation, dtype=np.int64)  # (N)
                all_observations.append(binary_observation)
            self.observations_labels = np.stack(all_observations, axis=1)  # (N x C)
        else:
            self.observations_labels = None

        # --- Load the anatomy-level observation (target) classes, possible from several observation sets ---
        if config.has_anatomy_observations and config.load_anatomy_observations:
            assert len(config.anatomy_names) > 0
            all_anatomy_observations = {}
            for anatomy_name in config.anatomy_names:
                anatomy_observations = []
                for ob_set in config.observation_sets:
                    if not ob_set.has_anatomy_observations:
                        continue
                    for ob_name in ob_set.observation_names:
                        if f'{ob_set.anatomy_observation_set_name}_{anatomy_name}_{ob_name}' not in data_df.columns:
                            log.warning(f'Anatomy label not found (this is expected if this pathology is never observed on that region): {ob_set.anatomy_observation_set_name}_{anatomy_name}_{ob_name}')
                            anatomy_observations.append(np.zeros(N, dtype=np.int64))
                            continue
                        binary_observation = _convert_labels(
                            data_df[f'{ob_set.anatomy_observation_set_name}_{anatomy_name}_{ob_name}'], 
                            uncertain_label=1 if ob_set.uncertain_pos else 0)
                        binary_observation = np.array(binary_observation, dtype=np.int64)
                        anatomy_observations.append(binary_observation)
                all_anatomy_observations[anatomy_name] = np.stack(anatomy_observations, axis=1)  # (N x C')
            self.anatomy_observations_labels = np.stack([all_anatomy_observations[anatomy_name] for anatomy_name in config.anatomy_names], axis=1)  # (N x A x C')
            # boolean mask (C) where C' elements (with anatomy_observation_labels) are set to True
            self.observations_with_anatomy_labels_mask = np.array([
                ob_set.has_anatomy_observations 
                for ob_set, _ in config.all_observation_name_pairs], dtype=np.bool)
        else:
            self.anatomy_observations_labels = None
            self.observations_with_anatomy_labels_mask = None

        # --- Load the anatomy boxes ---
        if config.has_anatomy_boxes and config.load_anatomy_boxes:
            # Box format: "(x1, y1, x2, y2)" in pixel coordinates
            log.info('Loading anatomical regions...')
            if loaded_size is not None:
                # images have already been resized during preprocessing which was not yet considered in the box coordinates
                # => rescale the box coordinates + shift them based on image cropping
                # load image sizes (one per sample)
                H, W = data_df.H.astype(float).to_numpy(),  data_df.W.astype(float).to_numpy()
                rescales = np.zeros((len(H),), dtype=np.float32)
                # rescale based on the smaller dimension
                rescales[H > W] = loaded_size / W[H > W]
                rescales[H <= W] = loaded_size / H[H <= W]
                crops = np.zeros((len(H), 2), dtype=np.float32)
                crops[H > W, 1] = (H - W)[H > W]  # y is cropped
                crops[H < W, 0] = (W - H)[H < W]  # x is cropped
            
            # load the boxes for each anatomy
            ob_set_bboxes = []  # box coordindates in x1y1x2y2 format in image coordinates
            all_box_masks = []  # box masks, i.e. true if box exists
            for box_name in tqdm.tqdm(config.anatomy_names):
                # load boxes for anatomy "box_name"
                boxes = data_df[f'anat_bbox_{box_name}'].fillna('(0, 0, 0, 0)').to_list()
                boxes = [parse_bbox(box) if box is not None else None for box in boxes]
                boxes = np.array(boxes, dtype=np.float)  # (N x 4)
                ob_set_bboxes.append(boxes)
                # remember which anatomy boxes exist
                box_masks = (~data_df[f'anat_bbox_{box_name}'].isna()).to_list()
                box_masks = np.array(box_masks, dtype=np.bool)  # (N)
                all_box_masks.append(box_masks)
            ob_set_bboxes = np.stack(ob_set_bboxes, axis=1)  # (N x A x 4)
            all_box_masks = np.stack(all_box_masks, axis=1)
            if loaded_size is not None:
                # rescale and shift the boxes
                ob_set_bboxes[:, :, 0::2] -= crops[:, None, 0, None] / 2.
                ob_set_bboxes[:, :, 1::2] -= crops[:, None, 1, None] / 2.
                ob_set_bboxes *= rescales[:, None, None]
                ob_set_bboxes = np.clip(ob_set_bboxes, 0.0, loaded_size)
            # remove too small boxes
            box_sizes = ob_set_bboxes[:, :, 2:4] - ob_set_bboxes[:, :, 0:2]
            # (N x A)
            empty_boxes = np.any((box_sizes <= 1e-3), axis=2)
            all_box_masks[empty_boxes] = False
            self.anatomy_boxes = ob_set_bboxes  # (N x A x 4)
            self.anatomy_box_masks = all_box_masks  # (N x A)
        else:
            self.anatomy_boxes = None
            self.anatomy_box_masks = None


        # --- Load the observation bounding boxes, possible from several observation sets ---
        # List of observation bounding boxes, one list per sample
        # Each list contains the bounding boxes of all observation sets as np.ndarray of dim (N x 5) in format (x1, y1, x2, y2, class_index)
        # where class_index is the index of the class in the list of all observation classes and therefore corresponds to the index in the observation labels
        self.observation_bboxes = None
        if config.has_observation_boxes and config.load_observation_boxes:
            # class index mapping with first key = observation set name, second key = class name
            class_maps: Dict[str, Dict[str, int]] = defaultdict(dict)
            for class_index, (ob_set, ob_name) in enumerate(config.all_observation_name_pairs):
                class_maps[ob_set.observation_set_name][ob_name] = class_index

            for ob_set in config.observation_sets:
                if not ob_set.has_bboxes:
                    continue
                ob_set_bboxes: List[str] = data_df[f'{ob_set.observation_set_name}_bboxes'].to_list()
                ob_set_bboxes: List[np.ndarray] = [parse_and_filter_bbox_list(sampe_bboxes, class_maps[ob_set.observation_set_name]) if sampe_bboxes is not None else None for sampe_bboxes in ob_set_bboxes]
                
                if self.observation_bboxes is None:
                    self.observation_bboxes = ob_set_bboxes
                else:
                    # concatenate the lists of boxes of different observation sets
                    self.observation_bboxes = [np.concatenate([a, b], axis=0) for a, b in zip(self.observation_bboxes, ob_set_bboxes)]

        return sample_ids, load_indices, load_fn

    def __getitem__(self, index) -> dict:
        img = self._load_img(index)
        if isinstance(img, Image.Image):
            img = np.array(img, dtype=np.float32) / 255.
        H, W = img.shape
        # (C) of type np.int64
        observations_labels: Optional[np.ndarray] = self.observations_labels[index] if self.observations_labels is not None else None
        # (A x C') of type np.int64
        anatomy_observation_labels: Optional[np.ndarray] = self.anatomy_observations_labels[index] if self.anatomy_observations_labels is not None else None
        
        assert self.transform is not None
        if self.anatomy_boxes is not None or self.observation_bboxes is not None:
            if self.anatomy_boxes is not None:
                # (n_anat_boxes x 4) of type np.float32 in format (xc, yc, w, h
                anatomy_boxes = self.anatomy_boxes[index]
                # (n_anat_boxes) of type np.bool
                anatomy_box_masks = self.anatomy_box_masks[index]
                n_anat_boxes = anatomy_boxes.shape[0]
                before = anatomy_box_masks.sum()
                anatomy_boxes = convert_bboxes_pixelx1y1x2y2_to_relxcyxwh(anatomy_boxes, H=H, W=W)
                box_label_mapping = np.arange(anatomy_box_masks.shape[0], dtype=np.int64)
                # (n_present_anat_boxes) of type np.int64
                present_labels = box_label_mapping[anatomy_box_masks]
                # (n_present_anat_boxes x 4) of type np.float32 in format (xc, yc, w, h)
                present_bboxes = anatomy_boxes[anatomy_box_masks, :]
            else:
                n_anat_boxes = 0
                present_labels = np.zeros((0,), dtype=np.int64)
                present_bboxes = np.zeros((0, 4), dtype=np.float32)
            if self.observation_bboxes is not None:
                # (M x 5) of type np.float32 in format (x1, y1, x2, y2, class_index)
                observation_bboxes: Optional[np.ndarray] = self.observation_bboxes[index]
                # (M) of type np.int64
                observation_bbox_labels = observation_bboxes[:, 4].astype(np.int64)
                observation_bboxes = observation_bboxes[:, :4]
                # (M x 4) of type np.float32 in format (xc, yc, w, h)
                observation_bboxes = convert_bboxes_pixelx1y1x2y2_to_relxcyxwh(observation_bboxes, H=H, W=W)
                # make sure the obvservation labels do not collide with the anatomy labels
                observation_bbox_labels = observation_bbox_labels + n_anat_boxes
                present_labels = np.concatenate([present_labels, observation_bbox_labels], axis=0)
                present_bboxes = np.concatenate([present_bboxes, observation_bboxes], axis=0)

            transformed = self.transform(image=img, bboxes=present_bboxes, labels=present_labels)

            present_labels = np.array(transformed['labels']).astype(np.int64)
            present_bboxes = np.array(transformed['bboxes'])

            if self.anatomy_boxes is not None:
                present_anat_labels = present_labels[present_labels < n_anat_boxes]
                present_anat_bboxes = present_bboxes[present_labels < n_anat_boxes]
                anatomy_box_masks = np.zeros_like(anatomy_box_masks)
                anatomy_box_masks[present_anat_labels] = True
                assert anatomy_box_masks.sum() > 0, f'No anatomy boxes remaining, before transform: {before}. Sample: {self.sample_ids[index]}'
                anatomy_boxes = np.zeros_like(anatomy_boxes)
                anatomy_boxes[present_anat_labels, :] = present_anat_bboxes
            else:
                anatomy_boxes = None
                anatomy_box_masks = None
            if self.observation_bboxes is not None:
                box_observations_labels = present_labels[present_labels >= n_anat_boxes] - n_anat_boxes
                observation_bboxes = present_bboxes[present_labels >= n_anat_boxes]
                observation_bboxes = np.concatenate([observation_bboxes, box_observations_labels[:, None]], axis=1)
            else:
                observation_bboxes = None
        else:
            transformed = self.transform(image=img, bboxes=[], labels=[])
            anatomy_boxes = None
            anatomy_box_masks = None
            observation_bboxes = None
        img = transformed['image']

        sample = {
            'x': img,
            'target_observation_classes': observations_labels,
            'target_observation_bboxes': observation_bboxes,
            'target_anatomy_observations': anatomy_observation_labels,
            'target_anatomy_boxes': anatomy_boxes,
            'target_anatomy_box_masks': anatomy_box_masks,
            'sample_id': str(self.sample_ids[index]),
            }
        return {key: value for key, value in sample.items() if value is not None}
    

def build_dataloader(config: DatasetConfig,
                     mode: str, 
                     pixel_mean: Tuple[float, float, float],
                     pixel_std: Tuple[float, float, float],
                     transform: TransformConfig,
                     batch_size: int,
                     num_workers: int = 0,
                     prefetch: bool = True) -> DataLoader:
    config = config if isinstance(config, DatasetConfig) else DatasetConfig(**config)
    if mode == 'train':
        is_train = True
    else:
        assert mode in ('val', 'test')
        is_train = False

    image_size = transform.train_size
    if not is_train and transform.val_size is not None:
        image_size = transform.val_size
    dataset = ImageAnatomyClassificationDataset(
                config, 
                mode=mode,
                image_size=image_size,
                prefetch=prefetch, 
                transform=build_transform(
                            transform,
                            is_train=is_train,
                            pixel_mean=pixel_mean, pixel_std=pixel_std))
    
    # Get one iter of train_ds and val_ds for debugging
    next(iter(dataset))

    # handle multiple views per sample (for contrastive loss)
    def collate_fn(batch: List[dict]):
        if 'target_observation_bboxes' in batch[0]:
            target_observation_bboxes_batch = [sample.pop('target_observation_bboxes') for sample in batch]
            target_observation_bboxes_batch = [torch.tensor(bboxes) for bboxes in target_observation_bboxes_batch]
            collated_batch: dict = default_collate(batch)
            collated_batch['target_observation_bboxes'] = target_observation_bboxes_batch
        else:
            collated_batch: dict = default_collate(batch)
        assert isinstance(collated_batch, dict)
        if config.has_anatomy_boxes and config.load_anatomy_boxes:
            collated_batch['anatomy_names'] = config.anatomy_names
        if config.has_observation_classes and config.load_observation_classes:
            collated_batch['observation_class_names'] = config.observation_names
        if config.has_anatomy_observations and config.load_anatomy_observations:
            collated_batch['targets_with_anatomy_observations'] = torch.tensor(dataset.observations_with_anatomy_labels_mask)
            collated_batch['observation_names_with_anatomy_obs'] = config.observation_names_with_anatomy_obs
        return collated_batch

    # Create dataloader
    return DataLoader(dataset,
                      batch_size,
                      shuffle=is_train,
                      drop_last=is_train,
                      pin_memory=True,
                      num_workers=num_workers,
                      collate_fn=collate_fn,
                      persistent_workers=False)
