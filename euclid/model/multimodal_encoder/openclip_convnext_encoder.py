import torch
import torch.nn as nn
from open_clip.timm_model import TimmModel
import open_clip
from typing import Any, Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint
from PIL import Image
from transformers import CLIPImageProcessor

class TimmModel4VLM(TimmModel):
    def __init__(self, original_instance):
        # Initialize with all attributes from the original instance
        self.__dict__.update(original_instance.__dict__)

    def forward(self, x):
        x = self.trunk.forward_features(x)
        return x

    
class OpenCLIPConvneXtVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self._device = None
        self._dtype = None

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.vision_tower, _, preprocess = open_clip.create_model_and_transforms(f'hf-hub:{self.vision_tower_name}')
        self.vision_tower = TimmModel4VLM(self.vision_tower.visual)
        self.vision_tower.requires_grad_(False)

        self.image_processor = CLIPImageProcessor(
            do_resize=True,
            size={"height": 512, "width": 512},
            resample=3,  # 3 corresponds to bicubic interpolation
            do_center_crop=True,
            crop_size={"height": 512, "width": 512},  
            do_rescale=True,
            rescale_factor=1/255,  # Standard rescaling factor for pixel values
            do_normalize=True,
            image_mean=preprocess.transforms[-1].mean,
            image_std=preprocess.transforms[-1].std,
            do_convert_rgb=True,
        )
    
    def forward(self, images):
        # return [batc_size, num_patches, hidden_size]
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                batch, hidden, _, _ = image_forward_out.shape
                image_feature = image_forward_out.view(batch, hidden, -1).transpose(1, 2)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            batch, hidden, _, _ = image_forward_outs.shape
            image_features = image_forward_outs.view(batch, hidden, -1).transpose(1, 2)
        return image_features

    @property
    def hidden_size(self):
        return self.vision_tower.trunk.head.norm.normalized_shape[0]


    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        return self.vision_tower.dtype if self.is_loaded else None

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        if self.is_loaded:
            self.vision_tower.to(dtype=dtype)

    @property
    def device(self):
        if self._device is not None:
            return self._device
        return self.vision_tower.device if self.is_loaded else None

    @device.setter
    def device(self, device):
        self._device = device
        if self.is_loaded:
            self.vision_tower.to(device=device)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if 'device' in kwargs:
            self.device = kwargs['device']
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        elif len(args) > 0:
            if isinstance(args[0], (str, torch.device)):
                self.device = args[0]
            elif isinstance(args[0], torch.dtype):
                self.dtype = args[0]
        return self