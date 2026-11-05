"""Build ECO-Net object memory and optionally refine part representations."""
import argparse
import json
from pathlib import Path
import platform
import torch
from torch.utils.data import DataLoader, DistributedSampler
from econet import ECONet
from econet.checkpoint import load, save
from econet.data import AbsScene
from econet.runtime import seed_everything, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/absscene.json')
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--out', default='runs/absscene')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--model-name',
        default='ECONet-absscene-sample-trained.pth',
        help='Filename for the compact checkpoint inside --out',
    )
    parser.add_argument('--resume')
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument('--regularize', action='store_true')
    flags.add_argument('--no-regularization', action='store_true')
 
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.regularize:
        cfg['regularization']['enabled'] = True
    if args.no_regularization:
        cfg['regularization']['enabled'] = False
    seed_everything(cfg['seed'])
    settings = cfg['training']
    ds = AbsScene(args.data, args.split)
    if settings['steps'] < 1 or settings['batch_size'] < 1 or settings['checkpoint_every'] < 1:
        raise ValueError('steps, batch_size, and checkpoint_every must be positive')
    if settings['batch_size'] > len(ds):
        raise ValueError('Training split is smaller than batch_size')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(data_sha256=sha256(args.data), split=args.split,
                      python=platform.python_version(), torch=torch.__version__)
    step, epoch, offset = 0, 0, 0
    record = None
    if args.resume:
        model, loaded, record = load(args.resume, cfg, args.device, training=True)
        if loaded != cfg or record.get('provenance') != provenance:
            raise ValueError('Resume requires identical config, training data, and environment')
        if record.get('sampler_state') is None:
            raise ValueError('Inference checkpoint cannot resume training')
        step = record['step']
        epoch, offset = record['sampler_state']
    else:
        model = ECONet(cfg, batch_size=settings['batch_size']).to(args.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['lr']) if cfg['regularization']['enabled'] else None
    if record:
        if optimizer:
            if record.get('optimizer') is None:
                raise ValueError('Checkpoint is missing optimizer state')
            optimizer.load_state_dict(record['optimizer'])
        torch.set_rng_state(record['rng'])
        if record.get('cuda_rng') and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(record['cuda_rng'])
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    # The original single-GPU DistributedSampler uses seed=0 and epoch seeding.
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True, seed=0)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(ds, batch_size=settings['batch_size'], sampler=sampler,
                        num_workers=settings['num_workers'], drop_last=True, generator=generator)
    selected = settings['selected_batches']
    if selected and not any(0 <= i < len(loader) for i in selected):
        raise ValueError('No selected batch exists; use selected_batches=[] for small datasets')
    model.train()
    with open(out/'training.jsonl', 'a' if args.resume else 'w') as log:
        while step < settings['steps']:
            sampler.set_epoch(epoch)
            for batch_id, batch in enumerate(loader):
                if batch_id < offset or (selected and batch_id not in selected):
                    continue
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                output = model(batch['images'].to(args.device), step, sample_ids=batch['index'])
                loss = output['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite part regularization loss')
                if optimizer:
                    loss.backward()
                    optimizer.step()
                step += 1
                stats = dict(step=step, epoch=epoch, batch=batch_id,
                             loss=float(loss.detach()), memory=output['memory_count'])
                log.write(json.dumps(stats)+'\n')
                log.flush()
                print(stats, flush=True)
                if step % settings['checkpoint_every'] == 0 or step == settings['steps']:
                    save(out/'last.pth', model, cfg, step, optimizer, provenance, (epoch, batch_id+1))
                if step >= settings['steps']:
                    break
            epoch += 1
            offset = 0
    # Rebuild the online memory with final frozen parameters after refinement.
    if optimizer and settings['rebuild_memory']:
        model.reset_memory()
        model.train()
        done, epoch = 0, 0
        with torch.no_grad():
            while done < settings['steps']:
                sampler.set_epoch(epoch)
                for batch_id, batch in enumerate(loader):
                    if selected and batch_id not in selected:
                        continue
                    model(batch['images'].to(args.device), done)
                    done += 1
                    if done >= settings['steps']:
                        break
                epoch += 1
    save(out / args.model_name, model, cfg, step,
     provenance=provenance, compact=True)


if __name__ == '__main__':
    main()
