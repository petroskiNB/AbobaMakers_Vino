from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, SiglipVisionConfig, SiglipVisionModel

MODEL_ID = 'google/siglip2-base-patch16-384'
MODEL_DIR = Path('artifacts/base_model')


def load_image(path):
    with Image.open(path) as source:
        rgba = ImageOps.exif_transpose(source).convert('RGBA')
    background = Image.new('RGBA', rgba.size, 'white')
    background.alpha_composite(rgba)
    return background.convert('RGB')


def foreground(image):
    """Conservative trim of a nearly white catalog background; not a detector."""
    small = image.copy()
    small.thumbnail((512, 512))
    a = np.asarray(small)
    mask = np.min(a, axis=2) < 235
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return image
    sx, sy = image.width / small.width, image.height / small.height
    box = (max(0, int(xs.min()*sx)-5), max(0, int(ys.min()*sy)-5),
           min(image.width, int((xs.max()+1)*sx)+5), min(image.height, int((ys.max()+1)*sy)+5))
    return image.crop(box)


def views(image):
    bottle = foreground(image)
    # Heuristic second view, not an annotated label detector.
    label = bottle.crop((0, int(bottle.height*.30), bottle.width, int(bottle.height*.92)))
    return [bottle, label]


def query_views(image):
    """Multi-scale central views for a phone photo aimed at one bottle."""
    w, h = image.size
    crops = [image]
    for width_fraction, top, bottom in [(.76, .04, .98), (.58, .18, .92), (.46, .28, .82)]:
        half = int(w * width_fraction / 2)
        center = w // 2
        crops.append(image.crop((max(0, center-half), int(h*top), min(w, center+half), int(h*bottom))))
    return crops


def prepare(image):
    return ImageOps.pad(image, (384, 384), color='white', method=Image.Resampling.LANCZOS)


class Encoder:
    def __init__(self, model_dir=MODEL_DIR):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        config = SiglipVisionConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = SiglipVisionModel.from_pretrained(
            model_dir, config=config, local_files_only=True).to(self.device).eval()
        if self.device == 'cuda':
            self.model.half()

    @torch.inference_mode()
    def encode(self, images):
        batch = self.processor(images=[prepare(x) for x in images], return_tensors='pt')
        pixels = batch['pixel_values'].to(self.device, dtype=self.model.dtype)
        features = self.model(pixel_values=pixels).pooler_output.float()
        return F.normalize(features, dim=-1).cpu()


class Adapter(nn.Module):
    def __init__(self, dimension=768, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)
