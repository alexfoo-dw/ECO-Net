"""Atomic checkpoint IO, distributed-prefix normalization, compact memory export."""
import copy
from pathlib import Path
import torch
from .model import ECONet
import inspect

MEMORY_NAMES = ['mem_nodes', 'mem_nodes_obj', 'mem_edges_idx', 'mem_edges_val',
                'mem_masks', 'l0_counts_mem', 'pix_counts_mem', 'mem_repeats', 'mem_masks_l0']
EVAL_THRESHOLDS = ['attn_thres', 'memref_obj_dup_sim_thres', 'memref_objnei_dupnei_sim_thres', 'memref_edge_sim_thres']


def compact_state(model):
    count = max(1, int(model.encoder.mem_count.item()))
    state = {}
    for key, value in model.state_dict().items():
        # Slice before copying to CPU so unused capacity is not copied first.
        if key in {'encoder.'+name for name in MEMORY_NAMES}:
            state[key] = value[:count].detach().cpu().clone()
        else:
            state[key] = value.detach().cpu().clone()
    state['encoder.mem_count_max'] = torch.tensor(count)
    return state


def save(path, model, config, step, optimizer=None, provenance=None, sampler_state=None, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    state = compact_state(model) if compact else model.state_dict()
    if compact:
        cfg['net']['mem_count_max'] = int(state['encoder.mem_count_max'])
    record = dict(format_version=1, model=state, config=cfg, step=step,
                  optimizer=optimizer.state_dict() if optimizer else None,
                  provenance=provenance or {}, sampler_state=sampler_state,
                  rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(record, temporary)
    temporary.replace(path)


def load(path, config=None, device='cpu', training=False):
    # torch.load can execute pickle instructions: use trusted checkpoints only.
    load_kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_kwargs['weights_only'] = False

    record = torch.load(path, **load_kwargs)
    if not isinstance(record, dict):
        raise ValueError('Expected a checkpoint dictionary')
    raw = record.get('model', record.get('state_dict', record))
    if not isinstance(raw, dict):
        raise ValueError('Expected model state dictionary')
    state = {
        key[len('module.'):] if key.startswith('module.') else key: value
        for key, value in raw.items()
    }
    cfg = copy.deepcopy(record.get('config', config))
    if cfg is None:
        raise ValueError('Original-format checkpoints need an external config')
    if 'encoder.mem_count' not in state:
        raise ValueError('Checkpoint does not contain ECO-Net object memory')
    if not training:
        count = max(1, int(state['encoder.mem_count']))
        for name in MEMORY_NAMES:
            key = 'encoder.'+name
            if key not in state:
                raise ValueError(f'Missing required memory buffer: {key}')
            state[key] = state[key][:count].clone()
        state['encoder.mem_count_max'] = torch.tensor(count)
        # Threshold sweeps do not alter feature switches or parameter shapes.
        if config is not None:
            for key in EVAL_THRESHOLDS:
                cfg['net'][key] = config['net'][key]
        record.pop('model', None)
        record.pop('state_dict', None)
        record.pop('model_opt', None)
        record.pop('optimizer', None)
        del raw
    nodes = state['encoder.mem_nodes']
    cfg['net'].update(mem_count_max=nodes.shape[0], memref_l0_count_max=nodes.shape[1],
                      memref_vecs_unit_z_size=nodes.shape[2])
    model = ECONet(cfg, batch_size=cfg['training']['batch_size'] if training else 1)
    expected = model.state_dict()
    optional = {'encoder.part_projector.weight'}
    missing, unexpected = set(expected)-set(state), set(state)-set(expected)
    if missing-optional or unexpected:
        raise ValueError(f'Checkpoint mismatch: missing={sorted(missing-optional)}, unexpected={sorted(unexpected)}')
    # Original checkpoints have no projector; its initial transform is identity.
    for key in missing:
        state[key] = expected[key]
    model.load_state_dict(state, strict=True)
    return model.to(device), cfg, record
