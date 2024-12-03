import torch
import torch.nn as nn
from open_clip.transformer import VisionTransformer, Transformer
import open_clip
from typing import Any, Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint
from PIL import Image
from transformers import CLIPImageProcessor

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class Transformer4VLM(Transformer):
    def __init__(self, original_instance):
        # Initialize with all attributes from the original instance
        self.__dict__.update(original_instance.__dict__)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        hidden_states = [x]
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
            hidden_states.append(x)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x, hidden_states

class VisionEncoderOutput:
    def __init__(self, last_hidden_state, hidden_states):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states

class VisionTransformer4VLM(VisionTransformer):
    def __init__(self, original_instance):
        # Initialize with all attributes from the original instance
        self.__dict__.update(original_instance.__dict__)
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if type(pixel_values) == list:
            pixel_values = torch.stack(pixel_values)
        x = self.conv1(pixel_values)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        last_hidden_state, hidden_states = self.transformer(x)
        if output_hidden_states:
            return VisionEncoderOutput(last_hidden_state, hidden_states)
        return last_hidden_state


class OpenCLIPViTVisionTower(nn.Module):
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
        self.vision_tower = VisionTransformer4VLM(self.vision_tower.visual)
        self.vision_tower.transformer = Transformer4VLM(self.vision_tower.transformer)
        self.vision_tower.requires_grad_(False)
        self.image_processor = CLIPImageProcessor(
            do_resize=True,
            size={"height": preprocess.transforms[0].size, "width": preprocess.transforms[0].size},
            resample=3,
            do_center_crop=True,
            crop_size={"height": preprocess.transforms[0].size, "width": preprocess.transforms[0].size},  
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=True,
            image_mean=preprocess.transforms[-1].mean,
            image_std=preprocess.transforms[-1].std,
            do_convert_rgb=True,
        )

        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features
    
    def forward(self, images):
        # return [batc_size, num_patches, hidden_size]
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

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

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.ln_post.normalized_shape[0]

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        # return (self.config.image_size // self.config.patch_size) ** 2
        return 256