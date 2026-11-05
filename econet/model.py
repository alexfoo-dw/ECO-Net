"""ECO-Net public model API."""
import torch
from torch import nn
from .encoder import GraphMemoryEncoder
from .losses import part_regularization


class ECONet(nn.Module):
    def __init__(self, config, batch_size=1):
        super().__init__()
        self.config = config
        self.input_size = config['net']['input_size']
        net = dict(config['net'])
        if not net['delta_pos_embed'] or not net['memref'] or net['memref_sel_step'] != -1:
            raise ValueError('Use delta_pos_embed=true, memref=true, memref_sel_step=-1')
        if net['pretrained']:
            raise ValueError('Enable DINO with features.dino, not net.pretrained')
        if net['djb']:
            raise ValueError('djb must be false; memory cleanup is managed by the runtime')
        if net['z_size'] % 3 or net['memref_vecs_unit_z_size'] > self.input_size[1]*self.input_size[2]:
            raise ValueError('z_size must be a multiple of 3; descriptor samples cannot exceed image pixels')
        self.encoder = GraphMemoryEncoder(batch_size=batch_size, **net)
        self.encoder.configure_features(config['features'])

    def forward(self, images, step=0, sample_ids=None):
        """Images are Bx3xHxW in [-1,1]. Output masks include background.

        Graph composition is discrete. The optional loss recomputes live part
        features under fixed masks and shared pseudo-object assignments.
        """
        if images.ndim != 4 or list(images.shape[1:]) != self.input_size:
            raise ValueError(f'Expected Bx{self.input_size}, got {list(images.shape)}')
        if not self.training and len(images) != 1:
            raise ValueError('Recognition uses batch size 1')
        with torch.no_grad():
            slots, masks, _ = self.encoder(images, None, step, 100)
        loss = self.encoder.part_projector.weight.sum()*0
        settings = self.config['regularization']
        if self.training and settings['enabled']:
            encoder = self.encoder
            encoder._rgb = encoder.encoder_pt_1((images+1)/2).flatten(2).transpose(1, 2)
            indices = torch.arange(len(images), device=images.device).repeat_interleave(encoder.K)
            descriptors = encoder.combine_features(encoder.part_shapes,
                encoder.part_masks.view(len(images)*encoder.K, -1), indices).flatten(1)
            if sample_ids is not None:
                indices = sample_ids.to(images.device)[indices]
            loss = settings['weight'] * part_regularization(descriptors,
                encoder.part_ids, indices, settings['temperature'])
        return dict(masks=masks.view(len(images), -1, 1, *self.input_size[1:]),
                    slots=slots, loss=loss, memory_count=int(self.encoder.mem_count.item()))

    def reset_memory(self):
        """Clear stored objects while retaining the current feature parameters."""
        with torch.no_grad():
            for name, buffer in self.encoder.named_buffers():
                if name != 'mem_count_max':
                    buffer.zero_()
