import dataclasses
from typing import Callable
from PIL import Image
import torch

def load_pil_gray(path: str) -> Image.Image:
    return Image.open(path).convert('L')
