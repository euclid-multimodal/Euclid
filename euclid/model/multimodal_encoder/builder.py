import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .openclip_vit_encoder import OpenCLIPViTVisionTower
from .openclip_convnext_encoder import OpenCLIPConvneXtVisionTower
from .dino_encoder import DINOVisionTower
from .siglip_encoder import SiglipVisionTower



def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if vision_tower.startswith("openai") or "ShareGPT4V" in vision_tower or 'clip' in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if vision_tower.startswith("laion"):
        if 'ViT' in vision_tower:
            return OpenCLIPViTVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        if 'convnext' in vision_tower:
            return OpenCLIPConvneXtVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if 'dino' in vision_tower:
        return DINOVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if 'siglip' in vision_tower:
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
