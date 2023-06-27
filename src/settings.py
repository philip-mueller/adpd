
import os

WANDB_PROJECT = os.environ.get('WANDBPROJECT', 'adpd')

# e.g. you W&B username
WANDB_ENTITY = os.environ.get('WANDBENTITY')

# default directory for logging, by default wandb logs, configs, and checkpoints will be stored here
MODELS_DIR = os.environ.get('LOG_DIR', os.path.expanduser("~/models/adpd"))

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

# default data directory, data will be downloaded into subfolders of this directory
DATA_DIR = os.environ.get('DATA_DIR', os.path.expanduser("~/datasets"))
# default directory for datasets related to MIMIC-CXR, by default MIMIC-CXR-JPG and Chest ImaGenome will be downloaded into subfolders of this directory
MIMIC_CXR_BASE_DIR = os.environ.get('MIMIC_CXR_BASE_DIR', os.path.join(DATA_DIR, 'MIMIC-CXR')) 
# default directory for processed MIMIC-CXR data, by default processed/cached data will be stored here
MIMIC_CXR_PROCESSED_DIR = os.environ.get('MIMIC_CXR_PROCESSED_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'processed'))

# ---> to skip downloading and use your already existing MIMIC-CXR-JPG dataset folder: set this to your mimic-cxr-jpg_2-0-0 folder <---
# download URL: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
# it should contain the following files/folders:
# - files
# - mimic-cxr-2.0.0-metadata.csv.gz
# - mimic-cxr-2.0.0-split.csv.gz
# - mimic-cxr-2.0.0-chexpert.csv.gz
MIMIC_CXR_JPG_DIR = os.environ.get('MIMIC_CXR_JPG_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'mimic-cxr-jpg_2-0-0'))

# ---> to skip downloading and use your already existing Chest ImaGenome dataset folder: set this to your chest-imagenome-dataset-1.0.0 folder <---
# download URL: https://physionet.org/content/chest-imagenome/1.0.0/
# it should contain the following folder structure:
# - "chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph.zip"  (where chest-imagenome-dataset-1.0.0 is already a subfolder)
CHEST_IMAGEGENOME_DIR = os.environ.get('CHEST_IMAGENOME_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'chest-imagenome-dataset-1.0.0'))

# ---> to skip downloading and use your already existing MS-CXR dataset folder: set this to your ms-cxr_0-1 folder <---
# download URL: https://physionet.org/content/ms-cxr/0.1.0/
# it should contain the following folder structure:
# - "ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-0.1/MS_CXR_Local_Alignment_v1.0.0.csv"
MS_CXR_DIR = os.environ.get('MS_CXR_DIR', os.path.join(MIMIC_CXR_BASE_DIR, 'ms-cxr_0-1'))

# ---> to skip downloading and use your already existing ChestXray-8 dataset: set this to your folder <---
# it should contain the following files/folders:
# - images
# - BBox_List_2017.csv
# - Data_Entry_2017.csv
CXR8_DIR = os.environ.get('CXR8_DIR', os.path.join(DATA_DIR, 'CXR8'))

# user and PW for Physionet (to download MIMIC-CXR-JPG and Chest ImaGenome)
PHYSIONET_USER = os.environ.get('PHYSIONET_USER')
PHYSIONET_PW = os.environ.get('PHYSIONET_PW')

RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')
