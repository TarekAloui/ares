import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dl_models.models.base import *
import timm


class cifarDeIT(ModelBase):
    def __init__(self):
        super(cifarDeIT).__init__("cifar", "DeIT")
        self.layer_ids = []
        self.default_prune_factors = []

    def build_model(self, faults=[]):
        module = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.set_model(module, self.layer_ids, self.default_prune_factors)
