from collections import defaultdict
from functools import partial
import getpass
from glob import glob
from io import BytesIO
import json
import logging
import os
from pathlib import Path
from tkinter import FALSE
from typing import Dict, List, Optional, Tuple
from urllib.request import HTTPBasicAuthHandler, HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, build_opener, install_opener, urlopen
from zipfile import ZipFile
import zipfile
import pandas as pd
from pyparsing import Opt, col
from PIL import ImageFile
import numpy as np
import albumentations as A
import cv2

from tqdm import tqdm

from settings import CHEST_IMAGEGENOME_DIR, MIMIC_CXR_JPG_DIR, MIMIC_CXR_PROCESSED_DIR, MS_CXR_DIR, PHYSIONET_PW, PHYSIONET_USER
from utils.data_utils import load_pil_gray


log = logging.getLogger(__name__)

MIMIC_CXR_TAGS = ['frontal', 'report', 'chexpert']
CHEST_IMAGENOME_TAGS = [ 'cig_anatboxes', 'cig_anatlabels', 'cig_labels', 'cig_split', 'cig_nogoldleak', 'cig_noleak', 'cigmimic_split']
MSCXR_TAGS= ['mscxr_exclude', 'mscxr_val', 'mscxr_boxes']
ALL_TAGS = MIMIC_CXR_TAGS + CHEST_IMAGENOME_TAGS + MSCXR_TAGS

IMAGE_IDS_TO_IGNORE = {
    "0518c887-b80608ca-830de2d5-89acf0e2-bd3ec900",
    "03b2e67c-70631ff8-685825fb-6c989456-621ca64d",
    "786d69d0-08d16a2c-dd260165-682e66e9-acf7e942",
    "1d0bafd0-72c92e4c-addb1c57-40008638-b9ec8584",
    "f55a5fe2-395fc452-4e6b63d9-3341534a-ebb882d5",
    "14a5423b-9989fc33-123ce6f1-4cc7ca9a-9a3d2179",
    "9c42d877-dfa63a03-a1f2eb8c-127c60c3-b20b7e01",
    "996fb121-fab58dd2-7521fd7e-f9f3133c-bc202556",
    "56b8afd3-5f6d4419-8699d79e-6913a2bd-35a08557",
    "93020995-6b84ca33-2e41e00d-5d6e3bee-87cfe5c6",
    "f57b4a53-5fecd631-2fe14e8a-f4780ee0-b8471007",
    "d496943d-153ec9a5-c6dfe4c0-4fb9e57f-675596eb",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50m",
    "422689b1-40e06ae8-d6151ff3-2780c186-6bd67271",
    "8385a8ad-ad5e02a8-8e1fa7f3-d822c648-2a41a205",
    "e180a7b6-684946d6-fe1782de-45ed1033-1a6f8a51",
    "f5f82c2f-e99a7a06-6ecc9991-072adb2f-497dae52",
    "6d54a492-7aade003-a238dc5c-019ccdd2-05661649",
    "2b5edbbf-116df0e3-d0fea755-fabd7b85-cbb19d84",
    "db9511e3-ee0359ab-489c3556-4a9b2277-c0bf0369",
    "87495016-a6efd89e-a3697ec7-89a81d53-627a2e13",
    "810a8e3b-2cf85e71-7ed0b3d3-531b6b68-24a5ca89",
    "a9f0620b-6e256cbd-a7f66357-2fe78c8a-49caac26",
    "46b02f13-69fb7e49-321880e4-80584065-c1f57b50",
}

def load_from_memmap(index, mmap_file):
    return mmap_file[index]

def load_mimic_cxr_datafile(
    dataset_name: str, split: str, image_size, load_memmap: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    max_size = max(image_size[0], image_size[1])
    if max_size <= 256:
        img_size_mode = 256
    else:
        img_size_mode = 512

    log.info(f'Loading MIMIC CXR dataset {dataset_name} ({split}) - size {img_size_mode}...')
    dataset_name = prepare_mimic_cxr_datasets(dataset_name)
    if load_memmap:
        mmap_file, file_mapping = downsample_and_load_mimic_cxr_images(img_size_mode)
    data_df = pd.read_csv(os.path.join(MIMIC_CXR_PROCESSED_DIR, f'{dataset_name}.{split}.csv'))
    data_df = data_df.astype({
        'subject_id': int,
        'study_id': int,
    })
    data_df = data_df.astype({
        'subject_id': str,
        'study_id': str,
        'dicom_id': str
    })
    data_df = data_df.copy()
    data_df['sample_id'] = pd.concat([data_df['subject_id'], data_df['study_id'], data_df['dicom_id']], axis=1).apply(lambda x: '/'.join(x), axis=1)
    
    sample_ids = data_df.sample_id.to_list()
    if load_memmap:
        log.info('Loading images from memmap...')
        indices = [file_mapping[sample_id] for sample_id in sample_ids]
        load_fn = partial(load_from_memmap, mmap_file=mmap_file)
    else:
        log.info('Loading images from jpg files...')
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img_dir = os.path.join(MIMIC_CXR_JPG_DIR, 'files')
        data_df['image_path'] = img_dir \
            + '/p' + data_df.subject_id.str.slice(stop=2) \
            + '/p' + data_df.subject_id \
            + '/s' + data_df.study_id \
            + '/' + data_df.dicom_id + '.jpg'
        indices = data_df.image_path
        load_fn = load_pil_gray
        img_size_mode = None
    return data_df, indices, load_fn, img_size_mode

def prepare_mimic_cxr_datasets(
    dataset_name: str,
    physionet_user: Optional[str]=None, 
    physionet_pw: Optional[str]=None,
    ):

    dataset_tags = dataset_name.split('-')
    assert dataset_tags[0] == 'mimic_cxr'
    dataset_tags = dataset_tags[1:]
    assert all(tag in ALL_TAGS for tag in dataset_tags), \
        f'Invalid dataset name {dataset_name}: Found invalid tags {[tag for tag in dataset_tags if tag not in ALL_TAGS]}. Available tags: {ALL_TAGS}'
    dataset_tags = [tag for tag in ALL_TAGS if tag in dataset_tags]
    dataset_name = 'mimic-cxr-' + '-'.join(dataset_tags)

    log.info(f'Note: Using MIMIC-CXR-JPG folder: {MIMIC_CXR_JPG_DIR}')
    log.info(f'Note: Using MIMIC-CXR-PROCESSED folder: {MIMIC_CXR_PROCESSED_DIR}')
    if all(os.path.exists(os.path.join(MIMIC_CXR_PROCESSED_DIR, f'{dataset_name}.{split}.csv')) for split in ['all', 'train', 'val', 'test']):
        log.info(f'Dataset {dataset_name} found. Skipping preparation')
        return dataset_name
    log.info(f'Preparing dataset {dataset_name}...')

    if physionet_user is None:
        assert PHYSIONET_USER is not None, 'No PhysioNet user provided, please set PHYSIONET_USER environment variable.'
        physionet_user = PHYSIONET_USER
    assert physionet_user is not None
    log.info(f'Using PhysioNet user {physionet_user}. Make sure this user is credentialed for MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.0.0/), Chest Imagenome (https://physionet.org/content/chest-imagenome/1.0.0/), and MS-CXR (https://physionet.org/content/ms-cxr).')
    if physionet_pw is None:
        physionet_pw = PHYSIONET_PW
    if physionet_pw is None:
        physionet_pw = getpass.getpass(f'PhysioNet Password for {physionet_user}:')
    assert physionet_pw is not None

    # MIMIC CXR
    mimic_cxr_tags = [tag for tag in MIMIC_CXR_TAGS if tag in dataset_tags]
    mimic_cxr_name = 'mimic-cxr-' + '-'.join(mimic_cxr_tags)
    mimic_cxr_meta = prepare_mimic_cxr(
        mimic_cxr_name, mimic_cxr_tags,
        MIMIC_CXR_JPG_DIR, MIMIC_CXR_PROCESSED_DIR,
        physionet_user=physionet_user, physionet_pw=physionet_pw)
    # CHEST IMAGENOME
    if any(tag in CHEST_IMAGENOME_TAGS for tag in dataset_tags):
        log.info(f'Note: Using CHEST IMAGENOME folder: {CHEST_IMAGEGENOME_DIR}')
        chest_imagenome_tags = [tag for tag in (MIMIC_CXR_TAGS + CHEST_IMAGENOME_TAGS) if tag in dataset_tags]
        chest_imagenome_name = 'mimic-cxr-' + '-'.join(chest_imagenome_tags)
        mimic_cxr_meta = prepare_chest_imagenome(
            chest_imagenome_name, chest_imagenome_tags,
            CHEST_IMAGEGENOME_DIR, MIMIC_CXR_PROCESSED_DIR,
            physionet_user=physionet_user, physionet_pw=physionet_pw,
            mimic_cxr_meta_df=mimic_cxr_meta
        )
    # MS CXR
    if any(tag in MSCXR_TAGS for tag in dataset_tags):
        log.info(f'Note: Using MS CXR folder: {MS_CXR_DIR}')
        prepare_ms_cxr(
            dataset_name, dataset_tags,
            MS_CXR_DIR, MIMIC_CXR_PROCESSED_DIR,
            physionet_user=physionet_user, physionet_pw=physionet_pw,
                mimic_cxr_meta_df=mimic_cxr_meta
            )

    return dataset_name

def prepare_ms_cxr(
    dataset_name, dataset_tags,
    mscxr_path, processed_dir,
    physionet_user, physionet_pw,
    mimic_cxr_meta_df: pd.DataFrame
):
    # download and extract
    if not os.path.exists(mscxr_path) or len(os.listdir(mscxr_path)) == 0:
        log.info(f'No MS CXR dataset found at {mscxr_path}')
        log.info('Downloading MS CXR dataset...')
        zip_file = os.path.join(mscxr_path, 'ms-cxr_0-1.zip')
        os.makedirs(mscxr_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/ms-cxr/get-zip/0.1/')

        log.info('Extracting dataset...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(mscxr_path)
        os.remove(zip_file)

    ms_cxr_annotations = pd.read_csv(
        os.path.join(
            mscxr_path, 
            'ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1', 
            'MS_CXR_Local_Alignment_v1.0.0.csv'), 
        index_col='dicom_id')

    if 'mscxr_exclude' in dataset_tags:
        log.info('Removing MS-CXR samples from train/val splits')
        mimic_cxr_meta_df = mimic_cxr_meta_df[(mimic_cxr_meta_df.split == 'test') | (~mimic_cxr_meta_df.index.isin(ms_cxr_annotations.index))]
        log.info(f'MIMIC-CXR images after removing MS-CXR from train/val: {mimic_cxr_meta_df.shape[0]} records')
    if 'mscxr_val' in dataset_tags:
        assert 'mscxr_exclude' not in dataset_tags
        log.info('Moving train/val MS-CXR samples to val split')
        mimic_cxr_meta_df[((mimic_cxr_meta_df.split != 'test') & mimic_cxr_meta_df.index.isin(ms_cxr_annotations.index)), 'split'] = 'val'
        log.info(f'MIMIC-CXR images after moving MS-CXR train/val MS-CXR samples to val split: {mimic_cxr_meta_df.shape[0]} records')

    split_and_save(mimic_cxr_meta_df, processed_dir, dataset_name)

    return mimic_cxr_meta_df


def prepare_chest_imagenome(
    dataset_name, dataset_tags,
    chest_imagenome_path, processed_dir,
    physionet_user, physionet_pw,
    mimic_cxr_meta_df: pd.DataFrame
):
    # download and extract
    if not os.path.exists(chest_imagenome_path) or len(os.listdir(chest_imagenome_path)) == 0:
        log.info(f'No Chest ImaGenome dataset found at {chest_imagenome_path}')
        log.info('Downloading Chest ImaGenome dataset...')
        zip_file = os.path.join(chest_imagenome_path, 'chest-imagenome-dataset-1.0.0.zip')
        os.makedirs(chest_imagenome_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/chest-imagenome/get-zip/1.0.0/')

        log.info('Extracting dataset...')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(chest_imagenome_path)
        os.remove(zip_file)
    
    # extract scene graphs
    silver_dir = os.path.join(chest_imagenome_path, 'chest-imagenome-dataset-1.0.0', 'silver_dataset')
    scene_graph_dir = os.path.join(silver_dir, 'scene_graph')
    if not os.path.exists(scene_graph_dir):
        log.info('Extracting scene graphs...')
        with zipfile.ZipFile(os.path.join(silver_dir, 'scene_graph.zip'), 'r') as zip_ref:
            zip_ref.extractall(silver_dir)

    # --------------------- convert scene graphs ------------------------------
    anat_bboxes_file = os.path.join(processed_dir, 'chest_imagenome-anat_bboxes.csv')
    anat_observations_file = os.path.join(processed_dir, 'chest_imagenome-anat_observations.csv')
    observation_file = os.path.join(processed_dir, 'chest_imagenome-observations.csv')
    if not os.path.exists(anat_bboxes_file) or not os.path.exists(anat_observations_file) or not os.path.exists(observation_file):
        scene_graphs = glob(f'{scene_graph_dir}/*.json')
        log.info('Processing scene graphs...')
        box_counts = defaultdict(int)
        invalid_boxes = []
        duplicate_boxes = []
        bbox_samples = []
        bbox_attribute_samples = []
        classification_samples = []
        samples_with_less_than_five_boxes = 0
        samples_with_non_frontal_view = 0
        for sg_path in tqdm(scene_graphs):
            with open(sg_path, 'r') as sg_file:
                sg = json.load(sg_file)
            image_id = sg['image_id']
            patient_id = sg['patient_id']
            study_id = sg['study_id']
            viewpoint = sg['viewpoint']
            if viewpoint not in ('PA', 'AP'):
                log.warning(f'Found sample with viewpoint {viewpoint}, Skipping.')
                samples_with_non_frontal_view += 1
                continue
            if image_id not in mimic_cxr_meta_df.index:
                log.warning(f'Sample not found in processed MIMIC CXR dataset: {image_id}. Skipping')
                continue
            img_h, img_w = mimic_cxr_meta_df.at[image_id, 'H'], mimic_cxr_meta_df.at[image_id, 'W']

            # ---> find bboxes
            boxes = {}
            has_duplicate_box = False
            for box_data in sg['objects']:
                bbox_name = box_data['bbox_name']
                box_coords = (
                    max(0, min(img_w, float(box_data['original_x1']))), 
                    max(0, min(img_h, float(box_data['original_y1']))), 
                    max(0, min(img_w, float(box_data['original_x2']))), 
                    max(0, min(img_h, float(box_data['original_y2'])))
                )
                if box_data['width'] <= 0 or box_data['height'] <= 0 \
                    or (box_coords[2] - box_coords[0]) <= 0 \
                    or (box_coords[3] - box_coords[1]) <= 0:
                    invalid_boxes.append((patient_id, study_id, image_id, bbox_name))
                    continue

                if bbox_name in boxes:
                    duplicate_boxes.append((patient_id, study_id, image_id, bbox_name))
                    old_cords = boxes[bbox_name]
                    # union of bboxes
                    box_coords = (
                        min(old_cords[0], box_coords[0]), min(old_cords[1], box_coords[1]),
                        max(old_cords[2], box_coords[2]), max(old_cords[3], box_coords[3])
                    )
                    has_duplicate_box = True
                else:
                    box_counts[bbox_name] += 1
                boxes[bbox_name] = box_coords
            if len(boxes) < 5:
                log.warning(f'Found sample with only {len(boxes)} valid boxes, Skipping.')
                samples_with_less_than_five_boxes += 1
                continue
            # ---> find attributes 
            def extract_attributes(attribute_tuples, category: str) -> List[str]:
                found_attributes = []
                for cat, relation, attr in attribute_tuples:
                    if cat == category and relation == 'yes':
                        found_attributes.append(attr)
                return found_attributes

            is_abnormal_attributes = defaultdict(lambda: False)
            disease_attributes = defaultdict(set)
            anatomical_finding_attributes = defaultdict(set)
            anat_phrases = defaultdict(list)
            for attribute_data in sg['attributes']:
                bbox_name = attribute_data['bbox_name']
                attribute_tuples = [tuple(attr.split('|')) for sent_attr in attribute_data['attributes'] for attr in sent_attr]

                is_abnormal = ('nlp', 'yes', 'abnormal') in attribute_tuples
                is_abnormal_attributes[bbox_name] = is_abnormal or is_abnormal_attributes[bbox_name]
                disease_findings = extract_attributes(attribute_tuples, 'disease')
                disease_attributes[bbox_name].update(disease_findings)
                anatomical_findings = extract_attributes(attribute_tuples, 'anatomicalfinding')
                anatomical_finding_attributes[bbox_name].update(anatomical_findings)
                phrases = attribute_data['phrases']
                anat_phrases[bbox_name].extend(phrases)

            all_diseases = set(diseases for bbox_diseases in disease_attributes.values() for diseases in bbox_diseases)
            all_anatomicalfindings = set(anatfind for bbox_anatfind in anatomical_finding_attributes.values() for anatfind in bbox_anatfind)
            any_is_abnormal = any(is_abnormal for is_abnormal in is_abnormal_attributes.values())

            bbox_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                **{f'anat_bbox_{bbox_name}': bbox for bbox_name, bbox in boxes.items()},
                'anat_has_duplicate_box': has_duplicate_box, 'anat_has_bboxes': len(boxes) > 0
                })
            bbox_attribute_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                **{f'anat_{bbox_name}_abnormal': 1.0 if is_abnormal else 0.0 for bbox_name, is_abnormal in is_abnormal_attributes.items()},
                **{f'anat_{bbox_name}_normal': 0.0 if is_abnormal else 1.0 for bbox_name, is_abnormal in is_abnormal_attributes.items()},
                **{f'anat_disease_{bbox_name}_{name}': 1.0 
                    for bbox_name, diseases in disease_attributes.items() for name in list(diseases) },
                **{f'anat_anatomicalfinding_{bbox_name}_{name}': 1.0 for bbox_name, anatfind in anatomical_finding_attributes.items() for name in list(anatfind)},
                })
            classification_samples.append({
                'dicom_id': image_id, 'subject_id': patient_id, 'study_id': study_id,
                'cig_abnormal': 1.0 if any_is_abnormal else 0.0,
                'cig_normal': 0.0 if any_is_abnormal else 1.0,
                **{f'cig_disease_{disease_name}': 1.0 for disease_name in all_diseases},
                **{f'cig_anatomicalfinding_{anatfind_name}': 1.0 for anatfind_name in all_anatomicalfindings},
            })
        log.info(f'Skipped {samples_with_non_frontal_view} samples with invalid viewpoint')
        log.info(f'Skipped {samples_with_less_than_five_boxes} samples with less than 5 valid boxes')
        # Save bboxes
        pd.DataFrame(bbox_samples).to_csv(anat_bboxes_file, index=False)
        duplicate_boxes_file = os.path.join(processed_dir, 'duplicated_boxes.csv')
        pd.DataFrame(duplicate_boxes, columns=['study_id', 'subject_id', 'dicom_id', 'bbox_name'])\
            .to_csv(duplicate_boxes_file, index=False)
        invalid_boxes_file = os.path.join(processed_dir, 'invalid_boxes.csv')
        pd.DataFrame(invalid_boxes, columns=['study_id', 'subject_id', 'dicom_id', 'bbox_name'])\
            .to_csv(invalid_boxes_file, index=False)
        stats_file = os.path.join(processed_dir, 'box_stats.json')
        with open(stats_file, 'w') as f:
            box_stats = {name: float(val) / len(bbox_samples) for name, val in box_counts.items()}
            json.dump(box_stats, f, indent=2, sort_keys=True)
        log.info(f'Bboxes written to file {anat_bboxes_file}')

        # Save bbox observations
        pd.DataFrame(bbox_attribute_samples).to_csv(anat_observations_file, index=False)
        log.info(f'Bboxes observations written to {anat_observations_file}')
        pd.DataFrame(classification_samples).fillna(0.0).to_csv(observation_file, index=False)
        log.info(f'Sample observations written to {observation_file}')

    log.info('Merging MIMIC CXR with Chest ImaGenome')
    log.info(f'MIMIC-CXR images before merge with Chest ImaGenome: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_anatboxes' in dataset_tags:
        anat_bboxes = pd.read_csv(anat_bboxes_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(anat_bboxes, how='inner')
        # we ignore samples with duplicate boxes (in train there are 9)
        mimic_cxr_meta_df = mimic_cxr_meta_df[~mimic_cxr_meta_df.anat_has_duplicate_box]
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome bboxes: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_anatlabels' in dataset_tags:
        anat_bbox_labels = pd.read_csv(anat_observations_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(anat_bbox_labels, how='inner')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome bbox observations: {mimic_cxr_meta_df.shape[0]} records')
    if 'cig_labels' in dataset_tags:
        observation_labels = pd.read_csv(observation_file, index_col='dicom_id').drop(columns=['study_id', 'subject_id'])
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(observation_labels, how='inner')
        log.info(f'MIMIC-CXR images after merge with Chest ImaGenome sample observations: {mimic_cxr_meta_df.shape[0]} records')
    
    splits_dir = os.path.join(silver_dir, 'splits')
    if 'cig_nogoldleak' in dataset_tags or 'cig_noleak' in dataset_tags or 'cig_split' in dataset_tags or 'cigmimic_split' in dataset_tags:
        ignore_images = pd.read_csv(os.path.join(splits_dir, 'images_to_avoid.csv'), index_col='dicom_id')
        mimic_cxr_meta_df = mimic_cxr_meta_df[~mimic_cxr_meta_df.index.isin(ignore_images.index)]
        log.info(f'Avoiding {ignore_images.shape[0]} images from Chest ImaGenome Gold Dataset, remaining: {mimic_cxr_meta_df.shape[0]} records')
    splits = []
    for split in ['train', 'valid', 'test']:
        split_images = pd.read_csv(os.path.join(splits_dir, f'{split}.csv'), usecols=['dicom_id'], index_col='dicom_id')
        split_images['split'] = split
        splits.append(split_images)
    splits = pd.concat(splits)

    if 'cig_split' in dataset_tags:
        log.info('Splitting based on Chest ImaGenome splits')
        mimic_cxr_meta_df = mimic_cxr_meta_df.rename(columns={'split': 'mimic_split'}).join(splits, how='inner')
        # remove mimic cxr test samples from train/val
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.split != 'test') & (mimic_cxr_meta_df.mimic_split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df.drop(columns=['mimic_split'])
        log.info(f'MIMIC-CXR images using Chest ImaGenome splits: {mimic_cxr_meta_df.shape[0]} records')
    elif 'cigmimic_split' in dataset_tags:
        log.info('Splitting based on MIMIC CXR and Chest ImaGenome splits (test set: in both test sets, train/val: MIMIC CXR split if not in any test set)')
        mimic_cxr_meta_df = mimic_cxr_meta_df.join(splits.rename(columns={'split': 'cig_split'}), how='inner')
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.split != 'test') & (mimic_cxr_meta_df.cig_split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df[~((mimic_cxr_meta_df.cig_split != 'test') & (mimic_cxr_meta_df.split == 'test'))]
        mimic_cxr_meta_df = mimic_cxr_meta_df.drop(columns=['cig_split'])
        log.info(f'MIMIC-CXR images using MIMIC CXR + Chest ImaGenome splits: {mimic_cxr_meta_df.shape[0]} records')
    elif 'cig_noleak' in dataset_tags:
        log.info('Splitting based on MIMIC-CXR splits but making sure no test samples from Chest ImaGenome are used in train or val.')
        test_split_images = pd.read_csv(os.path.join(splits_dir, 'test.csv'), usecols=['dicom_id'], index_col='dicom_id')
        mimic_cxr_meta_df = mimic_cxr_meta_df[(mimic_cxr_meta_df.split == 'test') | (~mimic_cxr_meta_df.index.isin(test_split_images.index))]
        log.info(f'MIMIC-CXR images after removing Chest ImaGenome test samples from trani/val: {mimic_cxr_meta_df.shape[0]} records')
    else:
        log.info('Splitting based on MIMIC-CXR splits. Chest ImaGenome test samples may be used during training.')

    split_and_save(mimic_cxr_meta_df, processed_dir, dataset_name)

    return mimic_cxr_meta_df

def prepare_mimic_cxr(
    dataset_name, dataset_tags,
    mimic_cxr_jpg_path, processed_dir,
    physionet_user, physionet_pw,
    check_files: bool = True) -> pd.DataFrame:

    if all(os.path.exists(os.path.join(processed_dir, f'{dataset_name}.{split}.csv')) for split in ['all', 'train', 'val', 'test']):
        log.info(f'MIMIC CXR dataset ({dataset_name}) already processed. Skipping processing')
        return pd.read_csv(os.path.join(processed_dir, f'{dataset_name}.all.csv'), index_col='dicom_id')

    if not exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path):
        log.info('Downloading MIMIC-CXR-JPG metadata...')
        zip_file = os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-jpg-2.0.0.zip')
        os.makedirs(mimic_cxr_jpg_path, exist_ok=True)
        os.system(f'wget --user {physionet_user} --password {physionet_pw} -O {zip_file} https://physionet.org/content/mimic-cxr-jpg/get-zip/2.0.0/')
        images_found = False
    else:
        if check_files:
            images_found = exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path)
        else:
            images_found = True

    if not images_found:
        log.info('Downloading MIMIC-CXR-JPG images...')
        os.system(f'wget -r -nc -nH --cut-dirs 4 -c -np -P {mimic_cxr_jpg_path}/files --user {physionet_user} --password {physionet_pw} https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/')
        log.info('Downloading images done')
    else:
        log.info('MIMIC CXR images found. Skipping download')

    mimic_cxr_meta = pd.read_csv(
        os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-metadata.csv.gz'),
        compression='gzip',
        usecols=['dicom_id', 'Rows', 'Columns', 'ViewPosition'], index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.rename(columns={'Rows': 'H', 'Columns': 'W', 'ViewPosition': 'view'})

    mimic_cxr_splits = pd.read_csv(
        os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-split.csv.gz'),
        compression='gzip', index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_splits, how='inner')
    log.info(f'Total MIMIC-CXR images: {mimic_cxr_meta.shape[0]} records')
    if 'frontal' in dataset_tags:
        mimic_cxr_meta = mimic_cxr_meta[mimic_cxr_meta.view.isin(('PA', 'AP'))]
        log.info(f'Frontal MIMIC-CXR images: {mimic_cxr_meta.shape[0]} records')

    if 'report' in dataset_tags:
        raise NotImplementedError

    if 'chexpert' in dataset_tags:
        mimic_cxr_chexpert = pd.read_csv(
            os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-chexpert.csv.gz'),
            compression='gzip', index_col=['subject_id', 'study_id'])
        mimic_cxr_chexpert = mimic_cxr_chexpert.rename(columns={name: f'chexpert_{name}' for name in mimic_cxr_chexpert.columns})
        # inner join -> we only consider samples with chexpert labels)
        mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_chexpert, on=['subject_id', 'study_id'], how='inner')

        log.info(f'Prepared MIMIC-CXR images with CheXpert labels: {mimic_cxr_meta.shape[0]} records')

    split_and_save(mimic_cxr_meta, processed_dir, dataset_name)

    return mimic_cxr_meta


def downsample_and_load_mimic_cxr_images(size_mode: int) -> Tuple[np.ndarray, Dict[str, int]]:
    assert os.path.exists(MIMIC_CXR_JPG_DIR)
    downsampled_path = os.path.join(MIMIC_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal.memmap')
    downsampled_info_path = os.path.join(MIMIC_CXR_PROCESSED_DIR, f'downsampled_{size_mode}_frontal_mapping.csv')
    if os.path.exists(downsampled_path):
        log.info(f'Using downsampled data {downsampled_path}')
        file_mapping = pd.read_csv(downsampled_info_path, usecols=['sample_id', 'index'], index_col='sample_id')
        file_mapping: Dict[str, int] = file_mapping.to_dict(orient='dict')['index']
        n_rows = len(file_mapping)
        mmap_file = np.memmap(downsampled_path, mode='r', dtype='float32', shape=(n_rows, size_mode, size_mode))
        return mmap_file, file_mapping

    log.info(f'Downsampling images to {size_mode} (saving to {downsampled_path})...')
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img_dir = os.path.join(MIMIC_CXR_JPG_DIR, 'files')
    mimic_cxr_meta = pd.read_csv(
        os.path.join(MIMIC_CXR_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv.gz'),
        compression='gzip',
        usecols=['dicom_id', 'Rows', 'Columns', 'ViewPosition'], index_col='dicom_id')
    mimic_cxr_splits = pd.read_csv(
        os.path.join(MIMIC_CXR_JPG_DIR, 'mimic-cxr-2.0.0-split.csv.gz'),
        compression='gzip', index_col='dicom_id')
    mimic_cxr_meta = mimic_cxr_meta.join(mimic_cxr_splits, how='inner')
    mimic_cxr_meta = mimic_cxr_meta[mimic_cxr_meta.ViewPosition.isin(('PA', 'AP'))]
    mimic_cxr_meta = mimic_cxr_meta.reset_index(drop=False)
    mimic_cxr_meta = mimic_cxr_meta.astype({
        'subject_id': int,
        'study_id': int,
    })
    mimic_cxr_meta = mimic_cxr_meta.astype({
        'subject_id': str,
        'study_id': str,
        'dicom_id': str
    })
    mimic_cxr_meta['sample_id'] = mimic_cxr_meta.subject_id + '/' + mimic_cxr_meta.study_id + '/' + mimic_cxr_meta.dicom_id
    mimic_cxr_meta['image_path'] = img_dir \
        + '/p' + mimic_cxr_meta.subject_id.str.slice(stop=2) \
        + '/p' + mimic_cxr_meta.subject_id \
        + '/s' + mimic_cxr_meta.study_id \
        + '/' + mimic_cxr_meta.dicom_id + '.jpg'

    file_mapping = []
    n_rows = mimic_cxr_meta.shape[0]
    pad_resize_transform = A.Compose([A.SmallestMaxSize(max_size=size_mode, interpolation=cv2.INTER_AREA), A.CenterCrop(height=size_mode, width=size_mode)])
    mmap_file = np.memmap(downsampled_path, mode='w+', dtype='float32', shape=(n_rows, size_mode, size_mode))
    for i, (row_index, row) in tqdm(enumerate(mimic_cxr_meta.iterrows()), total=n_rows):
        file_mapping.append((row['sample_id'], i))
        img = load_pil_gray(row['image_path'])
        img = np.array(img, dtype=np.float32) / 255.
        img = pad_resize_transform(image=img)['image']
        assert img.shape == (size_mode, size_mode)
        mmap_file[i, :, :] = img
        mmap_file.flush()
    
    pd.DataFrame(file_mapping, columns=['sample_id', 'index']).to_csv(downsampled_info_path)

    return mmap_file, {key: value for key, value in file_mapping}


# ================================= Utility functions =================================

def split_and_save(data_df, target_dir: str, target_file_prefix: str):
    os.makedirs(target_dir, exist_ok=True)
    n_samples = data_df.shape[0]
    split_file = os.path.join(target_dir, f'{target_file_prefix}.all.csv')
    data_df.to_csv(split_file)
    log.info(f'Saved {n_samples} records to {split_file}')
    for split in ('train', 'validate', 'test'):
        split_df: pd.DataFrame = data_df[data_df.split == split].drop(columns=['split'])
        if split == 'validate':
            split = 'val'
        n_split = split_df.shape[0]
        split_file = os.path.join(target_dir, f'{target_file_prefix}.{split}.csv')
        split_df.to_csv(split_file)
        log.info(f'Saved {n_split} records to {split_file}')
        n_samples = n_samples - n_split
    if n_samples > 0:
        log.warning(f'{n_samples} have not been saved as they are in neither of the train, val, test splits')

def exist_mimic_cxr_jpg_metadata_files(mimic_cxr_jpg_path) -> bool:
    if not os.path.exists(mimic_cxr_jpg_path) or len(os.listdir(mimic_cxr_jpg_path)) == 0:
        log.info(f'No MIMIC-CXR-JPG dataset found at {mimic_cxr_jpg_path}')
        return False
    for file in ['mimic-cxr-2.0.0-metadata.csv.gz', 'mimic-cxr-2.0.0-chexpert.csv.gz', 'mimic-cxr-2.0.0-split.csv.gz']:
        if not os.path.join(mimic_cxr_jpg_path, file):
            log.warning('Metadata file not found: ' + os.path.join(mimic_cxr_jpg_path, file))
            return False
    return True

def exist_mimic_cxr_jpg_images(mimic_cxr_jpg_path) -> bool:
    log.info('Checking MIMIC CXR JPG image files...')
    meta_data_file = os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-metadata.csv.gz')
    for i, row in tqdm(pd.read_csv(meta_data_file).iterrows()):
        p = 'p' + row['subject_id']
        p_short = p[:3]
        s = 's' + row['study_id']
        jpg_file = row['dicom_id'] + '.jpg'
        path = os.path.join(mimic_cxr_jpg_path, 'files', p_short, p, s, jpg_file)
        if not os.path.exists(path):
            log.warning('Images not complete. Image not found: ' + path)
            return False
    log.info('All MIMIC CXR JPG images found')
    return True
