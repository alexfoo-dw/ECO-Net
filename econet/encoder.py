"""Graph representation, co-part discovery, and persistent object memory.

L0 denotes parts, and L1 denotes composed objects. A paired L1 graph stores
corresponding parts from two occurrences. Training integrates graphs into
memory; recognition matches current parts to stored graphs and projects masks.
"""
import gc
import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as T
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from .pegbis.egbis import segment as egbis_segment
from .features import FeatureMixin

class GraphMemoryEncoder(FeatureMixin, nn.Module):

    def __init__(self, K, z_size, input_size, batch_size, delta_pos_embed, djb, lnf, memref, K_label, attn_thres, memref_sel_step, memref_smallobj_filter_size, mem_count_max, memref_obj_dup_sim_thres, memref_objnei_dupnei_sim_thres, memref_edge_sim_thres, memref_l0_count_max, memref_l1_count_max, memref_vecs_unit_z_size, memref_kernel_size, pretrained):
        """Allocate the pointwise encoder, spatial kernels, and persistent graph memory."""
        super(GraphMemoryEncoder, self).__init__()
        self.K = K
        self.z_size = z_size
        self.scale = self.z_size ** (-0.5)
        self.eps = 1e-08
        self.attention_iters = 3
        self.b = batch_size
        self.C = input_size[0]
        self.H = input_size[1]
        self.W = input_size[2]
        self.local_rank = '0'
        self.local_device = torch.device('cpu')
        self.delta_pos_embed = delta_pos_embed
        self.lnf = lnf
        self.djb = djb
        self.memref = memref
        self.pretrained = False
        self.positional_embedding = GraphMemoryEncoder.create_positional_embedding(self.H, self.W, batch_size, z_size)
        if not self.delta_pos_embed:
            self.pos_embed_projection = nn.Linear(4, self.z_size)
        self.attn_thres = attn_thres
        self.encoder_pt_1 = nn.Sequential(nn.Conv2d(3, self.z_size, 1, 1, 0, groups=3, bias=False, padding_mode='replicate'), nn.ReLU(True))

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1.0)
        self.encoder_pt_1.apply(init_weights)
        self.semi_conv_dim = self.z_size
        if self.lnf:
            self.norm_input = nn.LayerNorm((self.H * self.W, self.z_size), elementwise_affine=False)
        else:
            self.norm_input = torch.nn.Identity()
        # Neighbor-difference kernels and shifted masks define part adjacency.
        weights_left = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.weights_left = weights_left.view(1, 1, 3, 3).repeat(self.semi_conv_dim, 1, 1, 1)
        weights_topleft = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.weights_topleft = weights_topleft.view(1, 1, 3, 3).repeat(self.semi_conv_dim, 1, 1, 1)
        weights_top = torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.weights_top = weights_top.view(1, 1, 3, 3).repeat(self.semi_conv_dim, 1, 1, 1)
        weights_topright = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.weights_topright = weights_topright.view(1, 1, 3, 3).repeat(self.semi_conv_dim, 1, 1, 1)
        self.K_ori = K
        self.K_label = K_label
        self.sel_step = memref_sel_step
        if self.memref:
            if memref_kernel_size == 5:
                weights_left_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_topleft_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0]])
                weights_top_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]])
                weights_topright_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
                weights_right_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btmright_shift = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btm_shift = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btmleft_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
            else:
                weights_left_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_topleft_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_top_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_topright_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_right_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btmright_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btm_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
                weights_btmleft_shift = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
            self.weights_left_shift_init = weights_left_shift.view(1, 1, 5, 5)
            self.weights_topleft_shift_init = weights_topleft_shift.view(1, 1, 5, 5)
            self.weights_top_shift_init = weights_top_shift.view(1, 1, 5, 5)
            self.weights_topright_shift_init = weights_topright_shift.view(1, 1, 5, 5)
            self.weights_right_shift_init = weights_right_shift.view(1, 1, 5, 5)
            self.weights_btmright_shift_init = weights_btmright_shift.view(1, 1, 5, 5)
            self.weights_btm_shift_init = weights_btm_shift.view(1, 1, 5, 5)
            self.weights_btmleft_shift_init = weights_btmleft_shift.view(1, 1, 5, 5)
        if self.memref:
            self.smallobj_filter_size = memref_smallobj_filter_size
            self.obj_dup_sim_thres = memref_obj_dup_sim_thres
            self.objnei_dupnei_sim_thres = memref_objnei_dupnei_sim_thres
            self.edge_sim_thres = memref_edge_sim_thres
            self.vecs_unit_z_size = memref_vecs_unit_z_size
            self.l0_count_max = memref_l0_count_max
            self.l1_count_max = memref_l1_count_max
            self.edges_dim = 2
            # Registered buffers hold learned memory and are saved with model parameters.
            self.register_buffer('mem_count_max', torch.tensor(mem_count_max))
            self.register_buffer('mem_count', torch.tensor(0))
            self.register_buffer('mem_nodes', torch.zeros((self.mem_count_max, self.l0_count_max, self.vecs_unit_z_size, 2)))
            self.register_buffer('mem_nodes_obj', torch.zeros((self.mem_count_max, self.vecs_unit_z_size, 2)))
            self.register_buffer('mem_edges_idx', torch.zeros((self.mem_count_max, self.l0_count_max, 2)))
            self.register_buffer('mem_edges_val', torch.zeros((self.mem_count_max, self.l0_count_max, self.edges_dim)))
            self.register_buffer('mem_masks', torch.zeros((self.mem_count_max, 1, self.H * self.W), dtype=torch.long))
            self.register_buffer('l0_counts_mem', torch.zeros(self.mem_count_max, dtype=torch.long))
            self.register_buffer('pix_counts_mem', torch.zeros(self.mem_count_max, dtype=torch.long))
            self.register_buffer('mem_repeats', torch.zeros(self.mem_count_max, dtype=torch.long))
            self.register_buffer('mem_masks_l0', torch.zeros((self.mem_count_max, self.l0_count_max, 1, self.H * self.W), dtype=torch.uint8))

    @staticmethod
    def create_positional_embedding(h, w, batch_size, z_size):
        """Construct the coordinate grid used by the original shape and edge descriptors."""
        dist_left = torch.linspace(1, w, w).view(w, 1, 1).repeat(1, h, 1)
        dist_top = torch.linspace(1, h, h).view(1, h, 1).repeat(w, 1, 1)
        return torch.cat([dist_left, dist_top], 2).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    def compare_part_graphs(self, mem_id_0, mem_id_1):
        """Align part nodes, edges, and object nodes with Hungarian assignments."""
        mem_nodes_0_count, mem_nodes_1_count = (self.l0_counts_mem[mem_id_0].item(), self.l0_counts_mem[mem_id_1].item())
        mem_nodes_count = mem_nodes_0_count if mem_nodes_0_count > mem_nodes_1_count else mem_nodes_1_count
        mem_nodes_0 = self.mem_nodes[mem_id_0, :mem_nodes_count]
        mem_nodes_1 = self.mem_nodes[mem_id_1, :mem_nodes_count]
        mem_nodes_dist_allpair = ((mem_nodes_0.unsqueeze(1).expand(-1, mem_nodes_count, -1, -1) - mem_nodes_1.unsqueeze(0)) ** 2).sum(-1).sum(-1)
        row_ind, col_ind = linear_sum_assignment(mem_nodes_dist_allpair.detach().cpu().numpy(), maximize=False)
        mem_nodes_dist = mem_nodes_dist_allpair[row_ind, col_ind].sum()
        mem_edges_0 = self.mem_edges_val[mem_id_0, :mem_nodes_count]
        mem_edges_1 = self.mem_edges_val[mem_id_1, :mem_nodes_count]
        mem_edges_dist_allpair = ((mem_edges_0.unsqueeze(1).expand(-1, mem_nodes_count, -1) - mem_edges_1.unsqueeze(0)) ** 2).sum(-1)
        row_ind_e, col_ind_e = linear_sum_assignment(mem_edges_dist_allpair.detach().cpu().numpy(), maximize=False)
        mem_edges_dist = mem_edges_dist_allpair[row_ind_e, col_ind_e].sum()
        mem_obj_nodes_count = self.vecs_unit_z_size
        mem_nodes_0 = self.mem_nodes_obj[mem_id_0]
        mem_nodes_1 = self.mem_nodes_obj[mem_id_1]
        mem_nodes_dist_allpair = ((mem_nodes_0.unsqueeze(1).expand(-1, self.vecs_unit_z_size, -1) - mem_nodes_1.unsqueeze(0)) ** 2).sum(-1)
        row_ind, col_ind = linear_sum_assignment(mem_nodes_dist_allpair.detach().cpu().numpy(), maximize=False)
        mem_nodes_obj_dist = mem_nodes_dist_allpair[row_ind, col_ind].sum()
        return (mem_nodes_dist, mem_edges_dist, mem_nodes_obj_dist)

    def convert_mask_to_nodes(self, mask_tensor_bin, pos_embed):
        """Represent an object union mask in the same node layout used for parts."""
        mask_tensor_bin = mask_tensor_bin.squeeze(-2).unsqueeze(-1)
        mask_count = mask_tensor_bin.shape[0]
        mem_mask_pos_embed = pos_embed[:1].expand(mask_count, -1, -1) * mask_tensor_bin
        mem_mask_cm = mem_mask_pos_embed.sum(dim=-2) / mask_tensor_bin.sum(dim=-2)
        mem_mask_cm_vecs = mem_mask_pos_embed - mem_mask_cm.unsqueeze(-2)
        mem_mask_cm_vecs_len = (mem_mask_cm_vecs ** 2).sum(-1, keepdim=True).sqrt() * mask_tensor_bin
        mem_mask_cm_vecs_unit = mem_mask_cm_vecs / mem_mask_cm_vecs_len
        mem_mask_cm_vecs_unit = torch.nan_to_num(mem_mask_cm_vecs_unit, nan=1.1e-05, posinf=7.7e-05, neginf=-7.7e-05)
        mem_mask_cm_vecs_len_topk = mem_mask_cm_vecs_len.topk(int(self.vecs_unit_z_size / 2), dim=-2)
        mem_mask_cm_vecs_len_topk_idx = mem_mask_cm_vecs_len_topk.indices
        mem_mask_cm_vecs_len_topk = mem_mask_cm_vecs_len_topk.values
        mem_mask_cm_vecs_len_btmk = mem_mask_cm_vecs_len.clone()
        mem_mask_cm_vecs_len_btmk[mem_mask_cm_vecs_len_btmk == 0] = self.H + self.W
        mem_mask_cm_vecs_len_btmk = mem_mask_cm_vecs_len_btmk.topk(self.vecs_unit_z_size - int(self.vecs_unit_z_size / 2), dim=-2, largest=False)
        mem_mask_cm_vecs_len_btmk_idx = mem_mask_cm_vecs_len_btmk.indices
        mem_mask_cm_vecs_len_btmk = mem_mask_cm_vecs_len_btmk.values
        mem_mask_cm_vecs_len_topk_idx = torch.cat((mem_mask_cm_vecs_len_topk_idx, mem_mask_cm_vecs_len_btmk_idx), dim=-2)
        mem_mask_cm_vecs_unit_topk = torch.gather(mem_mask_cm_vecs_unit, -2, mem_mask_cm_vecs_len_topk_idx.expand(-1, -1, 2))
        image_ids = torch.full((mask_count,), self._object_image_id, device=mask_tensor_bin.device, dtype=torch.long)
        return self.combine_features(mem_mask_cm_vecs_unit_topk, mask_tensor_bin.squeeze(-1), image_ids)

    def forward(self, inputs, rand_pixel, step=-1, tensorboard_freq=100):
        """Segment pixels, compose recurring part graphs, then store or recognize objects."""
        self.b = inputs.shape[0]
        self.local_device = inputs.device
        self._rgb = self.encoder_pt_1((inputs + 1) / 2).flatten(2).transpose(1, 2)
        self._distilled = self.distilled_features((inputs + 1) / 2)
        x_orig = inputs
        pos_embed = self.create_positional_embedding(self.H, self.W, self.b, self.z_size).to(inputs.device)
        inputs = (inputs + 1) / 2
        self.K = self.K_ori
        b = inputs.shape[0]
        inputs = self.encoder_pt_1(inputs)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        if not self.delta_pos_embed:
            pos_embed = self.pos_embed_projection(pos_embed)
            inputs += pos_embed
            inputs = inputs.view(inputs.shape[0], -1, self.z_size)
            inputs = self.encoder_pt_2(inputs)
            b, n, d = inputs.shape
            inputs = self.norm_input(inputs)
            k = self.to_k(inputs)
        else:
            inputs = inputs.view(inputs.shape[0], self.H * self.W, -1)
            b, n, d = inputs.shape
            pos_embed = pos_embed[:b].view(b, self.H * self.W, -1)
            k = self.norm_input(inputs)
            inputs_pe = torch.cat((inputs, pos_embed), dim=-1)
        # Segmentation is discrete; gradients are handled by the separate part loss.
        imgs = inputs.view(b, self.H, self.W, -1).detach().cpu().numpy()
        attn_stack_label = torch.zeros(b, self.K * self.K_label, self.H * self.W).to(inputs.device)
        K_label_max = 0
        for b_idx in range(b):
            img = imgs[b_idx] * 255
            egbis_out = egbis_segment(img, sigma=0, k=self.attn_thres, min_size=1, filename='', four_connect=False)
            egbis_labels = np.unique(egbis_out)
            num_labels = len(egbis_labels)
            if num_labels > K_label_max:
                K_label_max = num_labels
            ari_pred_masks = torch.zeros((self.H, self.W)).to(self.local_device)
            for idx, label in enumerate(egbis_labels):
                ari_pred_masks[egbis_out == label] = idx
            label_attn = F.one_hot(ari_pred_masks.view(-1).long(), num_labels).float()
            attn_stack_label[b_idx, :num_labels] = label_attn.transpose(1, 0).contiguous().clone()
        attn_stack_init = attn_stack_label.clone()
        self.part_masks = attn_stack_label.detach().clone()
        attn_stack = attn_stack_label.unsqueeze(2)
        slots = torch.rand(b, self.K * self.K_label, self.z_size).to(self.local_device)
        oc_dist_loss = 0
        if self.memref and (self.sel_step == -1 or step == self.sel_step):
            self.K = self.K * self.K_label
            self.weights_left_shift = self.weights_left_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_topleft_shift = self.weights_topleft_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_top_shift = self.weights_top_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_topright_shift = self.weights_topright_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_right_shift = self.weights_right_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_btmright_shift = self.weights_btmright_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_btm_shift = self.weights_btm_shift_init.repeat(self.K, 1, 1, 1)
            self.weights_btmleft_shift = self.weights_btmleft_shift_init.repeat(self.K, 1, 1, 1)
            attn_stack_mask = attn_stack > 0.1
            attn_stack_bin = attn_stack_mask.float().view(b, -1, self.H, self.W)
            attn_idx = torch.argmax(attn_stack_bin, dim=1)
            attn_stack_filter_mask = attn_stack_bin.sum(-1).sum(-1) < self.smallobj_filter_size
            attn_stack_singlepix_mask = attn_stack_bin.sum(-1).sum(-1) == 1.0
            attn_stack_filter_idxes = attn_stack_filter_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
            attn_stack_bin[attn_stack_filter_mask] = 0
            attn_stack_filter_mask = attn_stack_bin.sum(-1).sum(-1) > self.H * self.W * 0.5
            attn_stack_bin[attn_stack_filter_mask] = 0
            attn_stack_bin = attn_stack_bin.view(b, self.K, -1, 1)
            attn_mask_pos_embed = pos_embed.unsqueeze(1).expand(-1, self.K, -1, -1) * attn_stack_bin
            # Centroids define translation-invariant shapes and relative edge directions.
            attn_mask_cm = attn_mask_pos_embed.sum(dim=-2) / attn_stack_bin.sum(dim=-2)
            attn_mask_feat = (attn_stack_bin.expand(-1, -1, -1, self.z_size) * k.unsqueeze(1).expand(-1, self.K, -1, -1)).sum(dim=2) / attn_stack_bin.sum(dim=-2)
            attn_mask_cm_vecs = attn_mask_pos_embed - attn_mask_cm.unsqueeze(2)
            attn_mask_cm_vecs_len = (attn_mask_cm_vecs ** 2).sum(-1, keepdim=True).sqrt() * attn_stack_bin
            attn_mask_cm_vecs_unit = attn_mask_cm_vecs / attn_mask_cm_vecs_len
            attn_mask_cm_mask = torch.isnan(attn_mask_cm_vecs_unit.view(self.b, self.K, self.H * self.W, 2).sum(-1)).unsqueeze(-1)
            attn_mask_cm_vecs_unit = torch.nan_to_num(attn_mask_cm_vecs_unit, nan=1.1e-05, posinf=7.7e-05, neginf=-7.7e-05)
            attn_mask_cm_vecs_len_topk = attn_mask_cm_vecs_len.clone()
            attn_mask_cm_vecs_len_topk[attn_mask_cm_mask] = -1
            attn_mask_cm_vecs_len_topk = attn_mask_cm_vecs_len_topk.topk(int(self.vecs_unit_z_size / 2), dim=2)
            attn_mask_cm_vecs_len_topk_idx = attn_mask_cm_vecs_len_topk.indices
            attn_mask_cm_vecs_len_topk = attn_mask_cm_vecs_len_topk.values
            attn_mask_cm_vecs_len_btmk = attn_mask_cm_vecs_len.clone()
            attn_mask_cm_vecs_len_btmk[attn_mask_cm_vecs_len_btmk == 0] = self.H + self.W
            attn_mask_cm_vecs_len_btmk[attn_mask_cm_mask] = (self.H + self.W) * 2
            attn_mask_cm_vecs_len_btmk = attn_mask_cm_vecs_len_btmk.topk(self.vecs_unit_z_size - int(self.vecs_unit_z_size / 2), dim=2, largest=False)
            attn_mask_cm_vecs_len_btmk_idx = attn_mask_cm_vecs_len_btmk.indices
            attn_mask_cm_vecs_len_btmk = attn_mask_cm_vecs_len_btmk.values
            attn_mask_cm_vecs_len_topk_idx = torch.cat((attn_mask_cm_vecs_len_topk_idx, attn_mask_cm_vecs_len_btmk_idx), dim=2)
            attn_mask_cm_vecs_unit_topk = torch.gather(attn_mask_cm_vecs_unit, -2, attn_mask_cm_vecs_len_topk_idx.expand(-1, -1, -1, 2))
            attn_mask_cm_rads_topk = torch.atan2(attn_mask_cm_vecs_unit_topk.view(-1, 2)[:, 0], attn_mask_cm_vecs_unit_topk.view(-1, 2)[:, 1]).view(self.b, self.K, -1)
            attn_mask_cm_vecs_unit_topk_sortidx = attn_mask_cm_rads_topk.clone().sort(dim=-1, descending=True).indices
            attn_mask_cm_vecs_unit_topk = torch.gather(attn_mask_cm_vecs_unit_topk, -2, attn_mask_cm_vecs_unit_topk_sortidx.unsqueeze(-1).expand(-1, -1, -1, 2))
            attn_mask_cm_vecs_unit_topk[attn_stack_singlepix_mask.view(self.b, self.K)] = -7.7e-05
            self.part_shapes = attn_mask_cm_vecs_unit_topk.view(b * self.K, self.vecs_unit_z_size, 2).detach().clone()
            self.part_masks = attn_stack_bin.view(b, self.K, -1).detach().clone()
            part_image_ids = torch.arange(b, device=inputs.device).repeat_interleave(self.K)
            attn_mask_cm_vecs_unit_topk = self.combine_features(self.part_shapes, self.part_masks.view(b * self.K, -1), part_image_ids).view(b, self.K, self.vecs_unit_z_size, 2)
            adj_obj = attn_mask_cm.unsqueeze(1).expand(-1, self.K, -1, -1) - attn_mask_cm.unsqueeze(-2)
            adj_obj_len = (adj_obj ** 2).sum(-1, keepdim=True).sqrt()
            adj_obj = adj_obj / adj_obj_len
            mask_diag = torch.eye(self.K).repeat(b, 1, 1).bool()
            adj_obj[mask_diag] = torch.tensor(float('inf')).to(adj_obj.device)
            attn_stack_bin = attn_stack_bin.view(b, -1, self.H, self.W)
            attn_stack_bin_sum = attn_stack_bin.sum(-1).sum(-1)
            attn_stack_bin_T = attn_stack_bin.reshape(b, self.K, -1).transpose(1, 2)
            attn_mask_left_shift = F.conv2d(attn_stack_bin, self.weights_left_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_left_adj = torch.matmul(attn_mask_left_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_left_shift_sum = attn_mask_left_shift.sum(-1).sum(-1)
            attn_mask_left_adj_union = attn_mask_left_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_left_adj_union[attn_mask_left_adj_union == 0] = 1
            attn_mask_left_adj = attn_mask_left_adj / attn_mask_left_adj_union
            attn_mask_topleft_shift = F.conv2d(attn_stack_bin, self.weights_topleft_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_topleft_adj = torch.matmul(attn_mask_topleft_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_topleft_shift_sum = attn_mask_topleft_shift.sum(-1).sum(-1)
            attn_mask_topleft_adj_union = attn_mask_topleft_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_topleft_adj_union[attn_mask_topleft_adj_union == 0] = 1
            attn_mask_topleft_adj = attn_mask_topleft_adj / attn_mask_topleft_adj_union
            attn_mask_top_shift = F.conv2d(attn_stack_bin, self.weights_top_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_top_adj = torch.matmul(attn_mask_top_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_top_shift_sum = attn_mask_top_shift.sum(-1).sum(-1)
            attn_mask_top_adj_union = attn_mask_top_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_top_adj_union[attn_mask_top_adj_union == 0] = 1
            attn_mask_top_adj = attn_mask_top_adj / attn_mask_top_adj_union
            attn_mask_topright_shift = F.conv2d(attn_stack_bin, self.weights_topright_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_topright_adj = torch.matmul(attn_mask_topright_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_topright_shift_sum = attn_mask_topright_shift.sum(-1).sum(-1)
            attn_mask_topright_adj_union = attn_mask_topright_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_topright_adj_union[attn_mask_topright_adj_union == 0] = 1
            attn_mask_topright_adj = attn_mask_topright_adj / attn_mask_topright_adj_union
            attn_mask_right_shift = F.conv2d(attn_stack_bin, self.weights_right_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_right_adj = torch.matmul(attn_mask_right_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_right_shift_sum = attn_mask_right_shift.sum(-1).sum(-1)
            attn_mask_right_adj_union = attn_mask_right_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_right_adj_union[attn_mask_right_adj_union == 0] = 1
            attn_mask_right_adj = attn_mask_right_adj / attn_mask_right_adj_union
            attn_mask_btmright_shift = F.conv2d(attn_stack_bin, self.weights_btmright_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_btmright_adj = torch.matmul(attn_mask_btmright_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_btmright_shift_sum = attn_mask_btmright_shift.sum(-1).sum(-1)
            attn_mask_btmright_adj_union = attn_mask_btmright_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_btmright_adj_union[attn_mask_btmright_adj_union == 0] = 1
            attn_mask_btmright_adj = attn_mask_btmright_adj / attn_mask_btmright_adj_union
            attn_mask_btm_shift = F.conv2d(attn_stack_bin, self.weights_btm_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_btm_adj = torch.matmul(attn_mask_btm_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_btm_shift_sum = attn_mask_btm_shift.sum(-1).sum(-1)
            attn_mask_btm_adj_union = attn_mask_btm_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_btm_adj_union[attn_mask_btm_adj_union == 0] = 1
            attn_mask_btm_adj = attn_mask_btm_adj / attn_mask_btm_adj_union
            attn_mask_btmleft_shift = F.conv2d(attn_stack_bin, self.weights_btmleft_shift.to(attn_stack_bin.device), padding=2, groups=self.K)
            attn_mask_btmleft_adj = torch.matmul(attn_mask_btmleft_shift.reshape(b, self.K, -1), attn_stack_bin_T)
            attn_mask_btmleft_shift_sum = attn_mask_btmleft_shift.sum(-1).sum(-1)
            attn_mask_btmleft_adj_union = attn_mask_btmleft_shift_sum.unsqueeze(-1).expand(-1, -1, self.K) + attn_stack_bin_sum.unsqueeze(-1)
            attn_mask_btmleft_adj_union[attn_mask_btmleft_adj_union == 0] = 1
            attn_mask_btmleft_adj = attn_mask_btmleft_adj / attn_mask_btmleft_adj_union
            adj_obj_iou = torch.stack([attn_mask_left_adj, attn_mask_topleft_adj, attn_mask_top_adj, attn_mask_topright_adj, attn_mask_right_adj, attn_mask_btmright_adj, attn_mask_btm_adj, attn_mask_btmleft_adj], dim=-1)
            mask_diag = torch.eye(self.K).view(1, self.K, self.K, 1).repeat(b, 1, 1, adj_obj_iou.shape[-1]).bool()
            adj_obj_iou[mask_diag] = torch.tensor(float(0)).to(adj_obj.device)
            adj_obj[adj_obj_iou.sum(-1) == 0] = torch.tensor(float('inf')).to(adj_obj.device)
            attn_stack_bin = attn_stack_bin.long().view(self.b * self.K, -1)
            adj_obj = adj_obj.view(b * self.K, self.K, -1)
            attn_mask_cm_vecs_unit_topk = attn_mask_cm_vecs_unit_topk.view(self.b * self.K, -1, 2)
            if self.training:
                # Scratch L1 graphs pair recurring part occurrences across this image batch.
                self.l0_counts_l1 = torch.zeros(self.l1_count_max, dtype=torch.long).to(self.local_device)
                self.pix_counts_l1 = torch.zeros(self.l1_count_max, dtype=torch.long).to(self.local_device)
                self.l1_nodes = torch.zeros((self.l1_count_max, self.l0_count_max, self.vecs_unit_z_size, 2)).to(self.local_device)
                self.l1_edges_idx = torch.zeros((self.l1_count_max, self.l0_count_max, 2)).to(self.local_device)
                self.l1_edges_val = torch.zeros((self.l1_count_max, self.l0_count_max, self.edges_dim)).to(self.local_device)
                self.l1_bk_ids = (torch.ones((self.l1_count_max, self.l0_count_max, 2), dtype=torch.long) * -1).to(self.local_device)
                self.l1_masks = torch.zeros((self.l1_count_max, 2, self.H * self.W), dtype=torch.long).to(self.local_device)
                self.visited_edges = torch.zeros((self.b * self.K, self.b * self.K), dtype=torch.long).to(self.local_device)
                self.bk_ids_assign = torch.zeros(self.b * self.K, dtype=torch.long).to(self.local_device)
                attn_mask_cm_vecs_unit_topk_flat_feat = attn_mask_cm_vecs_unit_topk[:, -self.z_size:, 0].view(self.b * self.K, -1)
                attn_mask_cm_vecs_unit_topk_flat = attn_mask_cm_vecs_unit_topk.view(self.b * self.K, -1)
                obj_sim = torch.zeros((self.b * self.K, self.b * self.K)).to(self.local_device)
                obj_sim_feat = torch.zeros((self.b * self.K, self.b * self.K)).to(self.local_device)
                for bk_id in range(self.b * self.K):
                    obj_sim[bk_id] = F.cosine_similarity(attn_mask_cm_vecs_unit_topk_flat[bk_id].unsqueeze(0).expand(b * self.K, -1), attn_mask_cm_vecs_unit_topk_flat, dim=-1)
                    obj_sim_feat[bk_id] = F.cosine_similarity(attn_mask_cm_vecs_unit_topk_flat_feat[bk_id].unsqueeze(0).expand(b * self.K, -1), attn_mask_cm_vecs_unit_topk_flat_feat, dim=-1)
                diag_mask = torch.eye(b * self.K).bool()
                obj_sim[diag_mask] = 0
                obj_sim_feat[diag_mask] = 0
                adj_obj_dup = adj_obj
            else:
                # Recognition reads populated memory records, not unused capacity.
                topnum = min(10, int(self.mem_count.item()))
                if topnum == 0:
                    raise RuntimeError('Object memory is empty; load a trained checkpoint')
                pix_counts_mem_extract = self.pix_counts_mem[:int(self.mem_count.item())]
                topnum_idxs = pix_counts_mem_extract.sort(descending=False).indices[-topnum:]
                obj_dist_lst = []
                for bk_id in range(self.b * self.K):
                    obj_dist_lst.append(torch.sqrt(((attn_mask_cm_vecs_unit_topk[bk_id].unsqueeze(0).unsqueeze(0).expand(-1, topnum * self.l0_count_max, -1, -1) - self.mem_nodes[topnum_idxs].view(topnum * self.l0_count_max, self.vecs_unit_z_size, -1).unsqueeze(0)) ** 2).sum(-1).sum(-1)))
                obj_dist = torch.stack(obj_dist_lst, dim=0).squeeze(1)
                obj_sim = torch.exp(-obj_dist)
                mem_edges_idx = self.mem_edges_idx[topnum_idxs]
                mem_edges_val = self.mem_edges_val[topnum_idxs]
                mem_masks = self.mem_masks[topnum_idxs]
                mem_nodes_obj = self.mem_nodes_obj[topnum_idxs]
                l0_counts_mem = self.l0_counts_mem[topnum_idxs]
                if hasattr(self, 'mem_masks_l0'):
                    mem_masks_l0 = self.mem_masks_l0[topnum_idxs]
                bk_id_mem_output = torch.zeros((topnum, 1, self.H * self.W), dtype=torch.long).to(self.local_device)
                mem_masks_l0_bk = mem_masks_l0.view(-1, self.H * self.W)
                for mem_id in range(topnum):
                    bk_id_mem_start = mem_id * self.l0_count_max
                    for l0_count_j in range(l0_counts_mem[mem_id].item()):
                        bk_id_mem = bk_id_mem_start + l0_count_j
                        bk_id_mem_output[mem_id, 0, mem_masks_l0_bk[bk_id_mem].bool()] = bk_id_mem
                memref_count_max = 5000
                l0_counts_memref = torch.zeros(memref_count_max, dtype=torch.long).to(self.local_device)
                pix_counts_memref = torch.zeros(memref_count_max, dtype=torch.long).to(self.local_device)
                pix_counts_memref_prev = pix_counts_memref.clone()
                memref_count = 0
                memref_bk_ids = (torch.ones((memref_count_max, self.l0_count_max, 2), dtype=torch.long) * -1).to(self.local_device)
                memref_masks = torch.zeros((memref_count_max, 1, self.H * self.W), dtype=torch.long).to(self.local_device)
                memref_nodes = torch.zeros((memref_count_max, self.l0_count_max, self.vecs_unit_z_size, 2)).to(self.local_device)
                memref_edges_idx = torch.zeros((memref_count_max, self.l0_count_max, 2)).to(self.local_device)
                memref_edges_val = torch.zeros((memref_count_max, self.l0_count_max, self.edges_dim)).to(self.local_device)
            if self.training:
                self.visited_edges = torch.zeros((self.b * self.K, self.b * self.K), dtype=torch.long).to(self.local_device)
            obj_sim_feat_thres = 0.997
            # Compose objects from matching parts with matching neighboring relations.
            for memref_iter in range(1):
                if self.training:
                    if memref_iter == 1:
                        break
                else:
                    memref_change = (pix_counts_memref[:memref_count] - pix_counts_memref_prev[:memref_count]).sum() / pix_counts_memref_prev[:memref_count].sum()
                    if memref_change < 0.05:
                        break
                    pix_counts_memref_prev = pix_counts_memref.clone()
                for obj_bk_id, obj_sim_row in tqdm(enumerate(obj_sim), total=len(obj_sim)):
                    if obj_bk_id in attn_stack_filter_idxes:
                        if self.training:
                            self.visited_edges[obj_bk_id] = 1
                        continue
                    obj_b_id, obj_k_id = (obj_bk_id // self.K, obj_bk_id % self.K)
                    dup_bk_ids = (obj_sim_row >= self.obj_dup_sim_thres).nonzero(as_tuple=False).squeeze(-1)
                    for dup_bk_id in dup_bk_ids:
                        dup_bk_id = dup_bk_id.item()
                        if self.training:
                            if self.visited_edges[obj_bk_id, dup_bk_id]:
                                continue
                            elif dup_bk_id in attn_stack_filter_idxes:
                                self.visited_edges[obj_bk_id, dup_bk_id], self.visited_edges[dup_bk_id, obj_bk_id] = (1, 1)
                                continue
                            else:
                                self.visited_edges[obj_bk_id, dup_bk_id], self.visited_edges[dup_bk_id, obj_bk_id] = (1, 1)
                            dup_b_id, dup_k_id = (dup_bk_id // self.K, dup_bk_id % self.K)
                        else:
                            dup_b_id, dup_k_id = (dup_bk_id // self.l0_count_max, dup_bk_id % self.l0_count_max)
                        if dup_b_id == obj_b_id:
                            continue
                        adj_obj_row = adj_obj[obj_bk_id]
                        if self.training:
                            adj_dup_row = adj_obj_dup[dup_bk_id]
                        else:
                            dup_edges_idx = (self.mem_edges_idx[dup_b_id, :, 0] == dup_k_id).nonzero().squeeze(-1)
                            dup_edges_idx_neg = (mem_edges_idx[dup_b_id, :, 0] != dup_k_id).nonzero().squeeze(-1)
                            adj_dup_row = mem_edges_val[dup_b_id].clone()
                            adj_dup_row[dup_edges_idx_neg] = torch.tensor(float('inf')).to(self.local_device)
                        for dup_nei_k_id, dup_nei_edge_val in enumerate(adj_dup_row):
                            if dup_nei_edge_val.sum() == torch.tensor(float('inf')).to(self.local_device):
                                continue
                            obj_dup_edge_sim = F.cosine_similarity(adj_obj_row, dup_nei_edge_val.unsqueeze(0).expand(self.K, -1), dim=-1)
                            obj_nei_k_ids = (obj_dup_edge_sim >= self.edge_sim_thres).nonzero(as_tuple=False).squeeze(-1)
                            for obj_nei_k_id in obj_nei_k_ids:
                                obj_nei_bk_id = (self.K * obj_b_id + obj_nei_k_id).item()
                                dup_nei_bk_id = self.K * dup_b_id + dup_nei_k_id if self.training else (self.l0_count_max * dup_b_id + mem_edges_idx[dup_b_id, dup_nei_k_id, 1]).to(torch.int)
                                if self.training:
                                    obj_nei_b_id, dup_nei_b_id = (obj_nei_bk_id // self.K, dup_nei_bk_id // self.K)
                                else:
                                    obj_nei_b_id, dup_nei_b_id = (obj_nei_bk_id // self.l0_count_max, dup_nei_bk_id // self.l0_count_max)
                                if self.training and (obj_bk_id == dup_nei_bk_id and obj_nei_bk_id == dup_bk_id):
                                    continue
                                obj_neighbor_sim = obj_sim[obj_nei_bk_id, dup_nei_bk_id]
                                if obj_neighbor_sim >= self.objnei_dupnei_sim_thres:
                                    if self.training:
                                        l1_obj_mask, l1_dup_mask, l1_obj_nei_mask, l1_dup_nei_mask = (obj_bk_id == self.l1_bk_ids, dup_bk_id == self.l1_bk_ids, obj_nei_bk_id == self.l1_bk_ids, dup_nei_bk_id == self.l1_bk_ids)
                                        l1_valid_mask = self.l0_counts_l1 > 0
                                        l1_valid_mask_expand = l1_valid_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, self.l0_count_max, 2)
                                        l1_obj_mask, l1_dup_mask, l1_obj_nei_mask, l1_dup_nei_mask = (l1_obj_mask & l1_valid_mask_expand, l1_dup_mask & l1_valid_mask_expand, l1_obj_nei_mask & l1_valid_mask_expand, l1_dup_nei_mask & l1_valid_mask_expand)
                                    else:
                                        l1_obj_mask, l1_dup_mask, l1_obj_nei_mask, l1_dup_nei_mask = (obj_bk_id == memref_bk_ids[..., :1], dup_bk_id == memref_bk_ids[..., -1:], obj_nei_bk_id == memref_bk_ids[..., :1], dup_nei_bk_id == memref_bk_ids[..., -1:])
                                    if self.training and ((l1_obj_mask.any(-1) & l1_dup_mask.any(-1)).any(-1) & (l1_obj_nei_mask.any(-1) & l1_dup_nei_mask.any(-1)).any(-1)).any():
                                        continue
                                    exists = False
                                    l1_id_sel = -1
                                    if (l1_obj_mask.any(-1) & l1_dup_mask.any(-1)).any():
                                        l1_l0_ids = (l1_obj_mask.any(dim=-1) & l1_dup_mask.any(dim=-1)).nonzero()
                                        if self.training:
                                            l1_id_sel, l0_id = l1_l0_ids[0]
                                            l0_count = self.l0_counts_l1[l1_id_sel].item()
                                            if l0_count == self.l0_count_max:
                                                continue
                                            l0_id_sel = l0_count
                                            self.l1_nodes[l1_id_sel, l0_count] = attn_mask_cm_vecs_unit_topk[obj_nei_bk_id]
                                            l1_edges_idx_count = (self.l1_edges_idx[l1_id_sel].sum(-1) != 0).sum(-1)
                                            self.l1_edges_idx[l1_id_sel, l1_edges_idx_count] = torch.tensor([l0_id, l0_count]).to(self.local_device)
                                            self.l1_edges_val[l1_id_sel, l1_edges_idx_count] = dup_nei_edge_val
                                            obj_l0_slotid = l1_obj_mask[l1_id_sel, l0_id].nonzero().squeeze(-1)[0]
                                            self.l1_bk_ids[l1_id_sel, l0_count, obj_l0_slotid] = obj_nei_bk_id
                                            self.l1_bk_ids[l1_id_sel, l0_count, (obj_l0_slotid + 1) % 2] = dup_nei_bk_id
                                            self.l1_masks[l1_id_sel, obj_l0_slotid] += attn_stack_bin[obj_nei_bk_id]
                                            self.l1_masks[l1_id_sel, (obj_l0_slotid + 1) % 2] += attn_stack_bin[dup_nei_bk_id]
                                            self.l0_counts_l1[l1_id_sel] += 1
                                            self.pix_counts_l1[l1_id_sel] = (self.l1_masks[l1_id_sel] != 0).sum(-1).min(-1).values
                                            obj_l0_slotid_sel = obj_l0_slotid
                                            for l1_id, l0_id in l1_l0_ids[1:]:
                                                l0_count_sel = self.l0_counts_l1[l1_id_sel]
                                                l0_count = self.l0_counts_l1[l1_id].item()
                                                l0_count_sum = l0_count_sel + l0_count - 1
                                                if l0_count_sum > self.l0_count_max:
                                                    continue
                                                self.l1_nodes[l1_id_sel, l0_count_sel:l0_count_sum] = torch.cat((self.l1_nodes[l1_id, :l0_id], self.l1_nodes[l1_id, l0_id + 1:l0_count]), dim=0).clone()
                                                l1_edges_idx_tmp = self.l1_edges_idx[l1_id].clone()
                                                l1_edges_idx_count = (l1_edges_idx_tmp.sum(-1) != 0).sum(-1)
                                                l1_sel_edges_idx_count = (self.l1_edges_idx[l1_id_sel].sum(-1) != 0).sum(-1)
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp < l0_id)[:l1_edges_idx_count]] += l0_count_sel
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp == l0_id)[:l1_edges_idx_count]] = l0_id_sel
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp > l0_id)[:l1_edges_idx_count]] += l0_count_sel - 1
                                                self.l1_edges_idx[l1_id_sel, l1_sel_edges_idx_count:l1_sel_edges_idx_count + l1_edges_idx_count] = self.l1_edges_idx[l1_id, :l1_edges_idx_count].clone()
                                                self.l1_edges_val[l1_id_sel, l1_sel_edges_idx_count:l1_sel_edges_idx_count + l1_edges_idx_count] = self.l1_edges_val[l1_id, :l1_edges_idx_count].clone()
                                                obj_l0_slotid = l1_obj_mask[l1_id, l0_id].nonzero().squeeze(-1)[0]
                                                self.l1_bk_ids[l1_id_sel, l0_count_sel:l0_count_sum, obj_l0_slotid_sel] = torch.cat((self.l1_bk_ids[l1_id, :l0_id, obj_l0_slotid], self.l1_bk_ids[l1_id, l0_id + 1:l0_count, obj_l0_slotid]), dim=0).clone()
                                                self.l1_bk_ids[l1_id_sel, l0_count_sel:l0_count_sum, (obj_l0_slotid_sel + 1) % 2] = torch.cat((self.l1_bk_ids[l1_id, :l0_id, (obj_l0_slotid + 1) % 2], self.l1_bk_ids[l1_id, l0_id + 1:l0_count, (obj_l0_slotid + 1) % 2]), dim=0).clone()
                                                self.l1_masks[l1_id_sel, obj_l0_slotid_sel] += self.l1_masks[l1_id, obj_l0_slotid].clone()
                                                self.l1_masks[l1_id_sel, (obj_l0_slotid_sel + 1) % 2] += self.l1_masks[l1_id, (obj_l0_slotid + 1) % 2].clone()
                                                self.l0_counts_l1[l1_id] = -1
                                                self.pix_counts_l1[l1_id] = -1
                                                self.l0_counts_l1[l1_id_sel] = l0_count_sum
                                                self.pix_counts_l1[l1_id_sel] = (self.l1_masks[l1_id_sel] != 0).sum(-1).min(-1).values
                                        else:
                                            for l1_id, l0_id in l1_l0_ids:
                                                if (l1_obj_nei_mask[l1_id].any(-1) & l1_dup_nei_mask[l1_id].any(-1)).any():
                                                    continue
                                                l0_count = l0_counts_memref[l1_id].item()
                                                memref_bk_ids[l1_id, l0_count, 0] = obj_nei_bk_id
                                                memref_bk_ids[l1_id, l0_count, 1] = dup_nei_bk_id
                                                memref_masks[l1_id] += attn_stack_bin[obj_nei_bk_id]
                                                memref_nodes[l1_id, l0_count] = attn_mask_cm_vecs_unit_topk[obj_nei_bk_id]
                                                memref_edges_idx[l1_id, l0_count] = torch.tensor([l0_id, l0_count]).to(self.local_device)
                                                memref_edges_val[l1_id, l0_count] = dup_nei_edge_val
                                                l0_counts_memref[l1_id] += 1
                                                pix_counts_memref[l1_id] = (memref_masks[l1_id] != 0).sum()
                                        exists = True
                                    if (l1_obj_nei_mask.any(-1) & l1_dup_nei_mask.any(-1)).any():
                                        l1_l0_ids = (l1_obj_nei_mask.any(dim=-1) & l1_dup_nei_mask.any(dim=-1)).nonzero()
                                        if self.training:
                                            if not exists:
                                                l1_id_sel, l0_id = l1_l0_ids[0]
                                                l0_count = self.l0_counts_l1[l1_id_sel].item()
                                                if l0_count == self.l0_count_max:
                                                    continue
                                                l0_id_sel = l0_count
                                                self.l1_nodes[l1_id_sel, l0_count] = attn_mask_cm_vecs_unit_topk[obj_bk_id]
                                                l1_edges_idx_count = (self.l1_edges_idx[l1_id_sel].sum(-1) != 0).sum(-1)
                                                self.l1_edges_idx[l1_id_sel, l1_edges_idx_count] = torch.tensor([l0_count, l0_id]).to(self.local_device)
                                                self.l1_edges_val[l1_id_sel, l1_edges_idx_count] = dup_nei_edge_val
                                                obj_l0_slotid = l1_obj_nei_mask[l1_id_sel, l0_id].nonzero().squeeze(-1)[0]
                                                self.l1_bk_ids[l1_id_sel, l0_count, obj_l0_slotid] = obj_bk_id
                                                self.l1_bk_ids[l1_id_sel, l0_count, (obj_l0_slotid + 1) % 2] = dup_bk_id
                                                self.l1_masks[l1_id_sel, obj_l0_slotid] += attn_stack_bin[obj_bk_id]
                                                self.l1_masks[l1_id_sel, (obj_l0_slotid + 1) % 2] += attn_stack_bin[dup_bk_id]
                                                self.l0_counts_l1[l1_id_sel] += 1
                                                self.pix_counts_l1[l1_id_sel] = (self.l1_masks[l1_id_sel] != 0).sum(-1).min(-1).values
                                            obj_l0_slotid_sel = obj_l0_slotid
                                            l1_l0_ids_filter = l1_l0_ids[1:] if not exists else l1_l0_ids
                                            for l1_id, l0_id in l1_l0_ids_filter:
                                                l0_count_sel = self.l0_counts_l1[l1_id_sel]
                                                l0_count = self.l0_counts_l1[l1_id].item()
                                                l0_count_sum = l0_count_sel + l0_count - 1
                                                if l0_count_sum > self.l0_count_max:
                                                    continue
                                                self.l1_nodes[l1_id_sel, l0_count_sel:l0_count_sum] = torch.cat((self.l1_nodes[l1_id, :l0_id], self.l1_nodes[l1_id, l0_id + 1:l0_count]), dim=0).clone()
                                                l1_edges_idx_tmp = self.l1_edges_idx[l1_id].clone()
                                                l1_edges_idx_count = (l1_edges_idx_tmp.sum(-1) != 0).sum(-1)
                                                l1_sel_edges_idx_count = (self.l1_edges_idx[l1_id_sel].sum(-1) != 0).sum(-1)
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp < l0_id)[:l1_edges_idx_count]] += l0_count_sel
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp == l0_id)[:l1_edges_idx_count]] = l0_id_sel
                                                self.l1_edges_idx[l1_id, :l1_edges_idx_count][(l1_edges_idx_tmp > l0_id)[:l1_edges_idx_count]] += l0_count_sel - 1
                                                self.l1_edges_idx[l1_id_sel, l1_sel_edges_idx_count:l1_sel_edges_idx_count + l1_edges_idx_count] = self.l1_edges_idx[l1_id, :l1_edges_idx_count].clone()
                                                self.l1_edges_val[l1_id_sel, l1_sel_edges_idx_count:l1_sel_edges_idx_count + l1_edges_idx_count] = self.l1_edges_val[l1_id, :l1_edges_idx_count].clone()
                                                obj_l0_slotid = l1_obj_nei_mask[l1_id, l0_id].nonzero().squeeze(-1)[0]
                                                self.l1_bk_ids[l1_id_sel, l0_count_sel:l0_count_sum, obj_l0_slotid_sel] = torch.cat((self.l1_bk_ids[l1_id, :l0_id, obj_l0_slotid], self.l1_bk_ids[l1_id, l0_id + 1:l0_count, obj_l0_slotid]), dim=0).clone()
                                                self.l1_bk_ids[l1_id_sel, l0_count_sel:l0_count_sum, (obj_l0_slotid_sel + 1) % 2] = torch.cat((self.l1_bk_ids[l1_id, :l0_id, (obj_l0_slotid + 1) % 2], self.l1_bk_ids[l1_id, l0_id + 1:l0_count, (obj_l0_slotid + 1) % 2]), dim=0).clone()
                                                self.l1_masks[l1_id_sel, obj_l0_slotid_sel] += self.l1_masks[l1_id, obj_l0_slotid].clone()
                                                self.l1_masks[l1_id_sel, (obj_l0_slotid_sel + 1) % 2] += self.l1_masks[l1_id, (obj_l0_slotid + 1) % 2].clone()
                                                self.l0_counts_l1[l1_id] = -1
                                                self.pix_counts_l1[l1_id] = -1
                                                self.l0_counts_l1[l1_id_sel] = l0_count_sum
                                                self.pix_counts_l1[l1_id_sel] = (self.l1_masks[l1_id_sel] != 0).sum(-1).min(-1).values
                                        else:
                                            for l1_id, l0_id in l1_l0_ids:
                                                if (l1_obj_mask[l1_id].any(-1) & l1_dup_mask[l1_id].any(-1)).any():
                                                    continue
                                                l0_count = l0_counts_memref[l1_id].item()
                                                memref_bk_ids[l1_id, l0_count, 0] = obj_bk_id
                                                memref_bk_ids[l1_id, l0_count, 1] = dup_bk_id
                                                memref_masks[l1_id] += attn_stack_bin[obj_bk_id]
                                                memref_nodes[l1_id, l0_count] = attn_mask_cm_vecs_unit_topk[obj_bk_id]
                                                memref_edges_idx[l1_id, l0_count] = torch.tensor([l0_count, l0_id]).to(self.local_device)
                                                memref_edges_val[l1_id, l0_count] = dup_nei_edge_val
                                                l0_counts_memref[l1_id] += 1
                                                pix_counts_memref[l1_id] = (memref_masks[l1_id] != 0).sum()
                                        exists = True
                                    if not exists:
                                        if not self.training:
                                            if memref_count == memref_count_max:
                                                continue
                                            memref_bk_ids[memref_count, 0] = torch.tensor([obj_bk_id, dup_bk_id]).to(self.local_device)
                                            memref_bk_ids[memref_count, 1] = torch.tensor([obj_nei_bk_id, dup_nei_bk_id]).to(self.local_device)
                                            memref_nodes[memref_count, 0:2] = attn_mask_cm_vecs_unit_topk[[obj_bk_id, obj_nei_bk_id]]
                                            memref_edges_idx[memref_count, 0] = torch.tensor([0, 1]).to(self.local_device)
                                            memref_edges_val[memref_count, 0] = dup_nei_edge_val
                                            memref_masks[memref_count] = attn_stack_bin[obj_bk_id] + attn_stack_bin[obj_nei_bk_id]
                                            l0_counts_memref[memref_count] += 2
                                            pix_counts_memref[memref_count] = (memref_masks[memref_count] != 0).sum()
                                        else:
                                            l1_id = self.l0_counts_l1 == 0
                                            if not l1_id.any():
                                                continue
                                            l1_id = l1_id.nonzero()[0, 0]
                                            self.l1_nodes[l1_id, 0:2] = attn_mask_cm_vecs_unit_topk[[obj_bk_id, obj_nei_bk_id]]
                                            self.l1_edges_idx[l1_id, 0] = torch.tensor([0, 1]).to(self.l1_bk_ids.device)
                                            self.l1_edges_val[l1_id, 0] = dup_nei_edge_val
                                            self.l1_bk_ids[l1_id, 0] = torch.tensor([obj_bk_id, dup_bk_id]).to(self.l1_bk_ids.device)
                                            self.l1_bk_ids[l1_id, 1] = torch.tensor([obj_nei_bk_id, dup_nei_bk_id]).to(self.l1_bk_ids.device)
                                            self.l1_masks[l1_id, 0] = attn_stack_bin[obj_bk_id] + attn_stack_bin[obj_nei_bk_id]
                                            self.l1_masks[l1_id, 1] = attn_stack_bin[dup_bk_id] + attn_stack_bin[dup_nei_bk_id]
                                            self.l0_counts_l1[l1_id] += 2
                                            self.pix_counts_l1[l1_id] = (self.l1_masks[l1_id] != 0).sum(-1).min(-1).values
                                        l1_id = l1_id if self.training else memref_count
                                        if not self.training:
                                            memref_count += 1
            if self.training:
                # Integrate composed graphs and masks into the persistent object memory.
                l1_id_filtered = []
                l1_valid_mask = self.l0_counts_l1 > 0
                l1_valid_mask_expand = l1_valid_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, self.l0_count_max, 2)
                for bk_id in tqdm(range(self.b * self.K), total=self.b * self.K):
                    b_id = bk_id // self.K
                    l1_bk_ids_mask = (self.l1_bk_ids == bk_id) & l1_valid_mask_expand
                    l1_mask = l1_bk_ids_mask.any(-1).any(-1)
                    l1_ids = l1_mask.nonzero().squeeze(-1)
                    if len(l1_ids) == 0:
                        continue
                    pix_counts_l1 = self.pix_counts_l1.clone()
                    pix_counts_l1[~l1_mask] = -1
                    l1_id_max = pix_counts_l1.argmax()
                    l1_id_max_slotid = l1_bk_ids_mask[l1_id_max].nonzero()[:, 1][0]
                    attn_stack_empty_idx_mask = attn_stack[b_id, :, 0].sum(-1) == 0
                    if attn_stack_empty_idx_mask.sum(-1) != 0:
                        attn_stack_empty_idx = attn_stack_empty_idx_mask.nonzero()[0, 0]
                        attn_stack[b_id, attn_stack_empty_idx, 0] = self.l1_masks[l1_id_max, l1_id_max_slotid].clone()
                    if l1_id_max not in l1_id_filtered:
                        l1_id_filtered.append(l1_id_max)
                        bk_ids_pairs = self.l1_bk_ids[l1_id_max, :self.l0_counts_l1[l1_id_max]]
                        for bk_ids_pairs_outer_idx, bk_id_outer in enumerate(bk_ids_pairs):
                            obj_bk_id, dup_bk_id = bk_id_outer
                            for bk_ids_pairs_inner_idx, bk_id_inner in enumerate(bk_ids_pairs):
                                obj_nei_bk_id, dup_nei_bk_id = bk_id_inner
                                obj_nei_k_id, dup_nei_k_id = (obj_nei_bk_id % self.K, dup_nei_bk_id % self.K)
                                if ((self.l1_edges_idx[l1_id_max, :, 0] == bk_ids_pairs_outer_idx) & (self.l1_edges_idx[l1_id_max, :, 1] == bk_ids_pairs_inner_idx)).any():
                                    continue
                                if adj_obj[obj_bk_id, obj_nei_k_id].sum() == torch.tensor(float('inf')):
                                    continue
                                if F.cosine_similarity(adj_obj[obj_bk_id, obj_nei_k_id], adj_obj_dup[dup_bk_id, dup_nei_k_id], dim=-1) >= self.edge_sim_thres:
                                    l1_sel_edges_idx_count = (self.l1_edges_idx[l1_id_max].sum(-1) != 0).sum(-1)
                                    if l1_sel_edges_idx_count == self.l0_count_max:
                                        continue
                                    self.l1_edges_idx[l1_id_max, l1_sel_edges_idx_count] = torch.tensor([bk_ids_pairs_outer_idx, bk_ids_pairs_inner_idx]).to(self.l1_bk_ids.device)
                                    self.l1_edges_val[l1_id_max, l1_sel_edges_idx_count] = adj_obj_dup[dup_bk_id, dup_nei_k_id]
                        if self.mem_count == self.mem_count_max:
                            mem_repeats_min_idx = (self.mem_repeats == self.mem_repeats.min()).nonzero().squeeze(-1)
                            pix_counts_mem_min = self.pix_counts_mem[mem_repeats_min_idx].min()
                            if self.pix_counts_l1[l1_id_max] > pix_counts_mem_min:
                                l0_counts_mem_min = mem_repeats_min_idx[self.pix_counts_mem[mem_repeats_min_idx].argmin()]
                                self.mem_nodes[l0_counts_mem_min] = self.l1_nodes[l1_id_max].clone()
                                self.mem_edges_idx[l0_counts_mem_min] = self.l1_edges_idx[l1_id_max].clone()
                                self.mem_edges_val[l0_counts_mem_min] = self.l1_edges_val[l1_id_max].clone()
                                self.l0_counts_mem[l0_counts_mem_min] = self.l0_counts_l1[l1_id_max].clone()
                                self.pix_counts_mem[l0_counts_mem_min] = self.pix_counts_l1[l1_id_max].clone()
                                self.mem_masks[l0_counts_mem_min] = self.l1_masks[l1_id_max, l1_id_max_slotid].clone()
                                mask_tensor_bin = (self.mem_masks[l0_counts_mem_min:l0_counts_mem_min + 1] > 0).float()
                                self.mem_nodes_obj[l0_counts_mem_min] = self.object_nodes(mask_tensor_bin, pos_embed, l1_id_max, l1_id_max_slotid)
                                self.mem_repeats[l0_counts_mem_min] = 0
                                if hasattr(self, 'mem_masks_l0'):
                                    bk_ids = self.l1_bk_ids[l1_id_max, :self.l0_counts_mem[l0_counts_mem_min], l1_id_max_slotid]
                                    self.mem_masks_l0[l0_counts_mem_min, :self.l0_counts_mem[l0_counts_mem_min]] = attn_stack_bin[bk_ids].view(-1, 1, self.H * self.W).clone().to(torch.uint8)
                                for mem_id in range(self.mem_count):
                                    if mem_id == l0_counts_mem_min:
                                        continue
                                    mem_nodes_dist, mem_edges_dist, mem_nodes_obj_dist = self.compare_part_graphs(mem_id, l0_counts_mem_min)
                                    if torch.exp(-mem_nodes_obj_dist) >= 1 and torch.exp(-mem_nodes_dist) >= 1:
                                        self.mem_repeats[mem_id] += 1
                                        self.l0_counts_mem[l0_counts_mem_min] = 0
                                        self.pix_counts_mem[l0_counts_mem_min] = 0
                                        break
                        else:
                            self.mem_nodes[self.mem_count] = self.l1_nodes[l1_id_max].clone()
                            self.mem_edges_idx[self.mem_count] = self.l1_edges_idx[l1_id_max].clone()
                            self.mem_edges_val[self.mem_count] = self.l1_edges_val[l1_id_max].clone()
                            self.l0_counts_mem[self.mem_count] = self.l0_counts_l1[l1_id_max].clone()
                            self.pix_counts_mem[self.mem_count] = self.pix_counts_l1[l1_id_max].clone()
                            self.mem_masks[self.mem_count] = self.l1_masks[l1_id_max, l1_id_max_slotid].clone()
                            mask_tensor_bin = (self.mem_masks[self.mem_count:self.mem_count + 1] > 0).float()
                            self.mem_nodes_obj[self.mem_count] = self.object_nodes(mask_tensor_bin, pos_embed, l1_id_max, l1_id_max_slotid)
                            if hasattr(self, 'mem_masks_l0'):
                                bk_ids = self.l1_bk_ids[l1_id_max, :self.l0_counts_mem[self.mem_count], l1_id_max_slotid]
                                self.mem_masks_l0[self.mem_count, :self.l0_counts_mem[self.mem_count]] = attn_stack_bin[bk_ids].view(-1, 1, self.H * self.W).clone().to(torch.uint8)
                            mem_found_match = False
                            for mem_id in range(self.mem_count):
                                mem_nodes_dist, mem_edges_dist, mem_nodes_obj_dist = self.compare_part_graphs(mem_id, self.mem_count)
                                if torch.exp(-mem_nodes_obj_dist) >= 1 and torch.exp(-mem_nodes_dist) >= 1:
                                    mem_found_match = True
                                    self.mem_repeats[mem_id] += 1
                                    break
                            if mem_found_match:
                                self.l0_counts_mem[self.mem_count] = 0
                            else:
                                self.mem_count += 1
                topnum_l1 = 1
                # Capture shared graph identities before deleting temporary graph tensors.
                self.capture_part_ids()
                del self.l0_counts_l1, self.l1_nodes, self.l1_edges_idx, self.l1_edges_val, self.l1_bk_ids, self.l1_masks, self.visited_edges, self.bk_ids_assign
                torch.cuda.empty_cache()
            else:
                seg_output = torch.zeros((self.b, 1, self.H * self.W), dtype=torch.long).to(self.local_device)
                bk_id_output = torch.zeros((self.b, 1, self.H * self.W), dtype=torch.long).to(self.local_device)
                l1_id_output = (torch.ones((self.b, 1, self.H * self.W), dtype=torch.long) * -1).to(self.local_device)
                seg_output_nclass = torch.ones(self.b, dtype=torch.long).to(self.local_device)
                seg_memref_map = [{} for _ in range(self.b)]
                no_memref_count = 0
                attn_stack_init = attn_stack_init.unsqueeze(2)
                self.part_masks = attn_stack_label.detach().clone()
                for bk_id in range(self.b * self.K):
                    l1_bk_ids_mask = memref_bk_ids[:memref_count, :, :1] == bk_id
                    l1_mask = l1_bk_ids_mask.any(-1).any(-1)
                    l1_ids = l1_mask.nonzero().squeeze(-1)
                    b_id = bk_id // self.K
                    bk_id_output[b_id, 0, attn_stack_bin[bk_id].bool()] = bk_id
                    if len(l1_ids) == 0:
                        no_memref_count += 1
                        _, K_label_init, _, _ = attn_stack_init.shape
                        bk_id_presence_sum = (attn_stack_init[b_id, :, 0].bool() & attn_stack_bin[bk_id].unsqueeze(0).expand(K_label_init, -1).bool()).sum(-1)
                        l1_id_max = bk_id_presence_sum.argmax() + memref_count
                    else:
                        pix_counts_l1 = pix_counts_memref[:memref_count].clone()
                        pix_counts_l1[~l1_mask] = -1
                        l1_id_max = pix_counts_l1.argmax()
                    if l1_id_max.item() not in seg_memref_map[b_id]:
                        seg_memref_map[b_id][l1_id_max.item()] = seg_output_nclass[b_id].item()
                        seg_output_nclass[b_id] += 1
                    seg_output[b_id, 0, attn_stack_bin[bk_id].bool()] = seg_memref_map[b_id][l1_id_max.item()]
                    l1_id_output[b_id, 0, attn_stack_bin[bk_id].bool()] = l1_id_max.item()
                seg_output_nclass_max = seg_output_nclass.max().item()
                attn_stack = torch.zeros((self.b, seg_output_nclass_max, 1, self.H * self.W)).to(self.local_device)
                for b_idx in range(self.b):
                    attn_stack[b_idx, :seg_output_nclass[b_idx]] = F.one_hot(seg_output[b_idx, 0], num_classes=seg_output_nclass[b_idx]).float().transpose(1, 0).contiguous().unsqueeze(1)
                attn_stack_memproj = torch.zeros((self.b, seg_output_nclass_max, 1, self.H * self.W)).to(self.local_device)
                bk_ids_count_memproj = torch.zeros((self.b, seg_output_nclass_max)).to(self.local_device)
                slots = torch.zeros((self.b, seg_output_nclass_max, self.vecs_unit_z_size * 2 + 2)).to(self.local_device)
                for b_idx in range(self.b):
                    for l1_id, seg_output_classidx in seg_memref_map[b_id].items():
                        memref_bk_ids_cur = memref_bk_ids[l1_id, :l0_counts_memref[l1_id].item()]
                        if l1_id >= memref_count:
                            attn_stack_memproj[b_id, seg_output_classidx, 0] = attn_stack_init[b_idx, l1_id - memref_count]
                            bk_ids_count_memproj[b_id, seg_output_classidx] = 0
                        else:
                            bk_id_samp, mem_bk_id_samp = memref_bk_ids_cur[0]
                            b_id, mem_id = (bk_id_samp // self.K, mem_bk_id_samp // self.l0_count_max)
                            mem_id_samp_posmask = mem_masks_l0_bk[mem_bk_id_samp].unsqueeze(-1).expand(-1, 2).long() * pos_embed[0]
                            mem_id_samp_cm = mem_id_samp_posmask[mem_id_samp_posmask.sum(-1) != 0].mean(0)
                            bk_id_samp_cm = attn_mask_cm.view(-1, 2)[bk_id_samp]
                            l0_mem_samp_cm_edge = bk_id_samp_cm - mem_id_samp_cm
                            mem_mask_shift = T.affine((mem_masks[mem_id, 0] != 0).long().view(1, self.H, self.W), angle=0, translate=[l0_mem_samp_cm_edge[1], l0_mem_samp_cm_edge[0]], scale=1, shear=0).view(-1)
                            attn_stack_memproj[b_id, seg_output_classidx, 0] = mem_mask_shift
                            bk_ids_count_memproj[b_id, seg_output_classidx] = l0_counts_memref[l1_id].item()
                            mask_tensor_bin = (mem_mask_shift.view(1, 1, self.H * self.W) > 0).float()
                            mask_tensor_bin = mask_tensor_bin.squeeze(-2).unsqueeze(-1)
                            mask_count = mask_tensor_bin.shape[0]
                            mem_mask_pos_embed = pos_embed[:1].expand(mask_count, -1, -1) * mask_tensor_bin
                            mem_mask_cm = mem_mask_pos_embed.sum(dim=-2) / mask_tensor_bin.sum(dim=-2)
                            slots[b_id, seg_output_classidx] = torch.cat((mem_nodes_obj[mem_id].view(-1), mem_mask_cm.view(-1)))
                attn_stack_memproj_overlap = attn_stack_memproj.sum(1)
                attn_stack_memproj_essen = torch.zeros((self.b, seg_output_nclass_max)).to(self.local_device)
                attn_stack_memproj_chosen_mask = torch.zeros((self.b, seg_output_nclass_max)).to(self.local_device)
                for b_idx in range(self.b):
                    for l1_id, seg_output_classidx in seg_memref_map[b_id].items():
                        attn_stack_memproj_cur = attn_stack_memproj[b_id, seg_output_classidx, 0]
                        memproj_remove_mask = (attn_stack_memproj_overlap[b_idx, 0] - attn_stack_memproj_cur) * attn_stack_memproj_cur
                        if (memproj_remove_mask - attn_stack[b_idx, seg_output_classidx] < 0).any():
                            attn_stack_memproj_essen[b_id, seg_output_classidx] += 1
                seg_output_memproj = torch.zeros((self.b, 1, self.H * self.W), dtype=torch.long).to(self.local_device)
                seg_output_nclass_memproj = torch.ones(self.b, dtype=torch.long).to(self.local_device)
                seg_memref_map_memproj = [{} for _ in range(self.b)]
                seg_memref_map_memproj_valid = [{} for _ in range(self.b)]
                no_memref_count_memproj = 0
                pix_counts_memproj = attn_stack_memproj.sum(-1).squeeze(-1)
                l1_id_memproj_list = list(seg_memref_map[b_id].keys())
                for b_id in range(self.b):
                    for hw in range(self.H * self.W):
                        attn_stack_memproj_mask = attn_stack_memproj[b_id, :, 0, hw] > 0
                        attn_stack_memproj_ids = attn_stack_memproj_mask.nonzero().squeeze(-1)
                        if len(attn_stack_memproj_ids) == 0:
                            no_memref_count_memproj += 1
                            continue
                        if len(attn_stack_memproj_ids) > 1:
                            memproj_essen_cur = attn_stack_memproj_essen[b_id].clone()
                            memproj_essen_cur[~attn_stack_memproj_mask] = 0
                            if memproj_essen_cur.sum() == 1:
                                memproj_id_max = memproj_essen_cur.argmax()
                            else:
                                bk_ids_count_memproj_cur = bk_ids_count_memproj[b_id].clone()
                                bk_ids_count_memproj_cur[~attn_stack_memproj_mask] = -1
                                memproj_id_maxval = bk_ids_count_memproj_cur.max()
                                if (bk_ids_count_memproj_cur == memproj_id_maxval).sum() == 1:
                                    memproj_id_max = bk_ids_count_memproj_cur.argmax()
                                else:
                                    pix_counts_memproj_cur = pix_counts_memproj[b_id].clone()
                                    pix_counts_memproj_cur[~attn_stack_memproj_mask] = -1
                                    pix_counts_memproj_cur[~(bk_ids_count_memproj_cur == memproj_id_maxval)] = -1
                                    memproj_id_max = pix_counts_memproj_cur.argmax()
                        else:
                            memproj_id_max = attn_stack_memproj_ids[0]
                        attn_stack_memproj_chosen_mask[b_id, memproj_id_max] = 1
                        if memproj_id_max.item() not in seg_memref_map_memproj[b_id]:
                            seg_memref_map_memproj[b_id][memproj_id_max.item()] = seg_output_nclass_memproj[b_id].item()
                            seg_memref_map_memproj_valid[b_id][memproj_id_max.item()] = 0
                            seg_output_nclass_memproj[b_id] += 1
                        seg_output_memproj[b_id, 0, hw] = seg_memref_map_memproj[b_id][memproj_id_max.item()]
                        if (memref_bk_ids[l1_id_memproj_list, :, :1] == bk_id_output[b_id, 0, hw]).any():
                            seg_memref_map_memproj_valid[b_id][memproj_id_max.item()] = 1
                if seg_output.sum() != 0:
                    seg_output_nclass_memproj_max = seg_output_nclass_memproj.max().item()
                    attn_stack_memproj_filtered = torch.zeros((self.b, seg_output_nclass_memproj_max, 1, self.H * self.W)).to(self.local_device)
                    for b_idx in range(self.b):
                        seg_output_memproj_onehot = F.one_hot(seg_output_memproj[b_idx, 0], num_classes=seg_output_nclass_memproj[b_idx]).float().transpose(1, 0).contiguous().unsqueeze(1)
                        seg_memref_map_memproj_valid_mask = np.array(list(seg_memref_map_memproj_valid[b_idx].values()))
                        attn_stack_memproj_filtered[b_idx, 1:seg_memref_map_memproj_valid_mask.sum() + 1] = seg_output_memproj_onehot[1:][seg_memref_map_memproj_valid_mask == 1]
                        attn_stack_memproj_filtered[b_idx, 0] = (~attn_stack_memproj_filtered[b_idx].sum(0).bool()).float()
                    attn_stack_memproj_chosen_maxcount = attn_stack_memproj_chosen_mask.long().sum(-1).max().item()
                    attn_stack_memproj_chosen = torch.zeros((self.b, attn_stack_memproj_chosen_maxcount + 1, 1, self.H * self.W)).to(self.local_device)
                    slots_chosen = torch.zeros((self.b, attn_stack_memproj_chosen_maxcount + 1, self.vecs_unit_z_size * 2 + 2)).to(self.local_device)
                    for b_idx in range(self.b):
                        attn_stack_memproj_chosen[b_idx, 1:attn_stack_memproj_chosen_maxcount + 1] = attn_stack_memproj[b_idx][attn_stack_memproj_chosen_mask[b_idx].bool()]
                        attn_stack_memproj_chosen[b_idx, 0] = (attn_stack_memproj[b_idx][attn_stack_memproj_chosen_mask[b_idx].bool()].sum(0) == 0).float()
                        slots_chosen[b_idx, 1:attn_stack_memproj_chosen_maxcount + 1] = slots[b_idx][attn_stack_memproj_chosen_mask[b_idx].bool()]
                    # Return translated whole-object masks, including recovered occluded regions.
                    attn_stack = attn_stack_memproj_chosen
                    slots = slots_chosen
        if not self.training and self.djb:
            del inputs, adj, pos_embed
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        del obj
                except:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
        return (slots, attn_stack, oc_dist_loss)
