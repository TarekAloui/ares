from dl_models.models.imagenet.imagenet_base import imagenetBase
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dl_models.models.base import *
import timm


class imagenetViT16_224(imagenetBase):
    def __init__(self):
        super(imagenetViT16_224, self).__init__("imagenet", "ViT")
        self.layer_ids = []
        self.default_prune_factors = []

    def build_model(self, pretrained=True, faults=[]):
        self.pretrained = pretrained
        module = timm.create_model("vit_base_patch16_224", pretrained)
        self.set_model(module, self.layer_ids, self.default_prune_factors)
