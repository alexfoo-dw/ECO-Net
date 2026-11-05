"""Selectable shape, texture, and frozen DINO node descriptors.

The memory node tensor remains [descriptor_samples, 2]. Shape-only retains
the supplied shape representation. Combined blocks share the descriptor width
and are individually normalized before the identity-initialized projector.
"""
import importlib.util
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
from .graph_ids import shared_object_ids


def load_dino(config):
    """Import CutLER's feature wrapper without importing MaskCut segmentation."""
    directory = Path(config['cutler_path']).expanduser().resolve() / 'maskcut'
    path = directory / 'dino.py'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; set features.cutler_path to a CutLER checkout')
    sys.path.insert(0, str(directory))
    # Some upstream versions resolve local helper modules from the repo root.
    sys.path.insert(1, str(directory.parent))
    spec = importlib.util.spec_from_file_location('econet_cutler_dino', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    architecture = config['dino_arch']
    if architecture not in ['small', 'base']:
        raise ValueError('dino_arch must be small or base')
    dimension = 384 if architecture == 'small' else 768
    model = module.ViTFeat(config['dino_weights'], dimension, architecture, 'k', config['dino_patch_size'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class FeatureMixin:
    def configure_features(self, config):
        self.feature_config = dict(config)
        if not any(config[k] for k in ['shape', 'texture', 'dino']):
            raise ValueError('Enable at least one feature family')
        width = self.vecs_unit_z_size * 2
        self.part_projector = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.part_projector.weight.copy_(torch.eye(width))
        self.distilled_backbone = load_dino(config) if config['dino'] else None

    @torch.no_grad()
    def distilled_features(self, images):
        if self.distilled_backbone is None:
            return None
        self.distilled_backbone.eval()
        # Normalization matches the constants in the supplied encoder.
        mean = images.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = images.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        patch = self.feature_config['dino_patch_size']
        resized = F.interpolate(images, (self.H*patch, self.W*patch), mode='bilinear', align_corners=False)
        output = self.distilled_backbone((resized-mean)/std)
        if not isinstance(output, torch.Tensor):
            raise TypeError('ViTFeat must return a tensor')
        if output.ndim == 3:
            if output.shape[-1] != self.H*self.W:
                raise ValueError('Expected DINO output BxDx(H*W)')
            output = output.reshape(len(images), -1, self.H, self.W)
        if output.ndim != 4:
            raise ValueError('Expected DINO output BxDxHxW or BxDx(H*W)')
        output = F.interpolate(output, (self.H, self.W), mode='nearest')
        return output.flatten(2).transpose(1, 2)

    @staticmethod
    def pool_pixels(pixels, masks, image_ids):
        """Pool by image using matrix multiplication; never replicate dense DINO maps per part."""
        output = pixels.new_zeros((len(masks), pixels.shape[-1]))
        for image_id in image_ids.unique():
            rows = torch.where(image_ids == image_id)[0]
            weights = masks[rows].to(pixels.dtype)
            pooled = weights @ pixels[image_id]
            output = output.index_copy(0, rows, pooled / weights.sum(1, keepdim=True).clamp_min(1))
        return output

    def combine_features(self, shape, masks, image_ids):
        shape = shape.reshape(len(masks), -1)
        masks = masks.reshape(len(masks), -1).to(shape.dtype)
        config = self.feature_config
        blocks = []
        if config['shape']:
            blocks.append(shape)
        if config['texture']:
            blocks.append(self.pool_pixels(self._rgb, masks, image_ids))
        if config['dino']:
            blocks.append(self.pool_pixels(self._distilled, masks, image_ids))
        if config['shape'] and len(blocks) == 1:
            combined = shape
        else:
            width = self.vecs_unit_z_size * 2
            sizes = [width//len(blocks) + int(i < width%len(blocks)) for i in range(len(blocks))]
            blocks = [F.normalize(F.interpolate(x[:, None], size=size, mode='linear', align_corners=False)[:, 0], dim=-1)
                      for x, size in zip(blocks, sizes)]
            combined = torch.cat(blocks, -1)
        projected = self.part_projector(torch.nan_to_num(combined))
        return projected.view(-1, self.vecs_unit_z_size, 2)

    def object_nodes(self, mask, positions, graph_id, occurrence):
        sample = int(self.l1_bk_ids[graph_id, 0, occurrence].item())
        self._object_image_id = sample // self.K
        return self.convert_mask_to_nodes(mask, positions)

    def capture_part_ids(self):
        graphs = []
        for i in range(len(self.l0_counts_l1)):
            count = int(self.l0_counts_l1[i].item())
            if count:
                graphs.append(self.l1_bk_ids[i, :count].detach().cpu().numpy())
        ids = shared_object_ids(graphs, self.b*self.K)
        self.part_ids = torch.as_tensor(ids, device=self.local_device)
