"""Evaluate the frozen ECO-Net object memory on an AbsScene split."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import colorsys
from PIL import Image, ImageDraw
from econet.checkpoint import load
from econet.data import AbsScene
from econet.metrics import (
    adjusted_rand_index,
    matching_iou_ari,
    matching_full_masks,
)
from econet.runtime import seed_everything, sha256

def save_comparison_png(path, image, truth, prediction,
                        gt_ids, pred_ids, scale=4):
    """Save input, GT, and predicted instance masks side by side.

    image:      [3,H,W], normalized to [-1,1]
    truth:      [G,1,H,W], background at index 0
    prediction: [K,1,H,W], background at index 0
    gt_ids/pred_ids: matched indices from matching_iou_ari()
    """
    rgb = image.detach().cpu().numpy().transpose(1, 2, 0)
    rgb = np.rint(np.clip((rgb + 1) / 2, 0, 1) * 255).astype(np.uint8)

    gt = truth.detach().cpu().numpy()[:, 0] > 0.5
    pred = prediction.detach().cpu().numpy()[:, 0] > 0.5

    height, width = rgb.shape[:2]

    gt_view = np.zeros((height, width, 3), dtype=np.float32)
    pred_view = np.zeros((height, width, 3), dtype=np.float32)
    gt_coverage = np.zeros((height, width, 1), dtype=np.float32)
    pred_coverage = np.zeros((height, width, 1), dtype=np.float32)

    match_index = 0

    for gt_id, pred_id in zip(gt_ids, pred_ids):
        gt_id, pred_id = int(gt_id), int(pred_id)

        # Exclude background, padded assignments, and invalid indices.
        if not (0 < gt_id < len(gt) and 0 < pred_id < len(pred)):
            continue

        gt_mask = gt[gt_id]
        pred_mask = pred[pred_id]

        # Ignore zero-overlap assignments from Hungarian matching.
        if not np.any(gt_mask & pred_mask):
            continue

        match_index += 1
        color = np.array(
            colorsys.hsv_to_rgb(
                (match_index * 0.61803398875) % 1.0,
                0.75,
                1.0,
            ),
            dtype=np.float32,
        ) * 255

        # Render the full masks of each matched foreground pair.
        gt_view += gt_mask[..., None] * color
        pred_view += pred_mask[..., None] * color

        gt_coverage += gt_mask[..., None]
        pred_coverage += pred_mask[..., None]

    # Blend overlapping foreground masks; uncovered pixels remain black.
    gt_view = np.rint(
        np.clip(gt_view / np.maximum(gt_coverage, 1), 0, 255)
    ).astype(np.uint8)

    pred_view = np.rint(
        np.clip(pred_view / np.maximum(pred_coverage, 1), 0, 255)
    ).astype(np.uint8)

    panels = [
        ("Input image", rgb),
        ("GT foreground matches", gt_view),
        ("Predicted foreground matches", pred_view),
    ]

    height, width = rgb.shape[:2]
    panel_width, panel_height = width * scale, height * scale
    gap, header = 8, 24

    canvas = Image.new(
        "RGB",
        (3 * panel_width + 2 * gap, panel_height + header),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    nearest = getattr(Image, "Resampling", Image).NEAREST

    for column, (title, array) in enumerate(panels):
        x = column * (panel_width + gap)
        panel = Image.fromarray(array).resize(
            (panel_width, panel_height), resample=nearest
        )
        canvas.paste(panel, (x, header))
        draw.text((x + 5, 6), title, fill="black")

    canvas.save(str(path))

def save_occlusion_png(path, image, truth, prediction,
                       gt_ids, pred_ids, scale=4):
    """Show each matched foreground object's full masks independently."""
    rgb = image.detach().cpu().numpy().transpose(1, 2, 0)
    rgb = np.rint(
        np.clip((rgb + 1) / 2, 0, 1) * 255
    ).astype(np.uint8)

    gt = truth.detach().cpu().numpy()[:, 0] > 0.5
    pred = prediction.detach().cpu().numpy()[:, 0] > 0.5

    pairs = []

    for gt_id, pred_id in zip(gt_ids, pred_ids):
        gt_id, pred_id = int(gt_id), int(pred_id)

        # Foreground only; exclude padded assignments.
        if not (0 < gt_id < len(gt) and 0 < pred_id < len(pred)):
            continue

        # Exclude zero-overlap assignments.
        if not np.any(gt[gt_id] & pred[pred_id]):
            continue

        pairs.append((gt_id, pred_id))

    height, width = rgb.shape[:2]
    panel_width = width * scale
    panel_height = height * scale
    header, gap = 24, 8
    row_height = header + panel_height

    canvas = Image.new(
        "RGB",
        (
            3 * panel_width + 2 * gap,
            max(1, len(pairs)) * row_height
            + max(0, len(pairs) - 1) * gap,
        ),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    nearest = getattr(Image, "Resampling", Image).NEAREST

    def paste_panel(array, column, row, title):
        x = column * (panel_width + gap)
        y = row * (row_height + gap)

        draw.text((x + 5, y + 6), title, fill="white")
        panel = Image.fromarray(array).resize(
            (panel_width, panel_height),
            resample=nearest,
        )
        canvas.paste(panel, (x, y + header))

    if not pairs:
        paste_panel(rgb, 0, 0, "Input image")
        draw.text(
            (panel_width + gap + 5, 6),
            "No foreground matches",
            fill="white",
        )

    for row, (gt_id, pred_id) in enumerate(pairs):
        # Use FULL masks—not intersections, visible-only pixels, or argmax.
        gt_mask = (gt[gt_id].astype(np.uint8) * 255)
        pred_mask = (pred[pred_id].astype(np.uint8) * 255)

        paste_panel(rgb, 0, row, "Input image")
        paste_panel(
            gt_mask, 1, row,
            "Full GT object {}".format(gt_id),
        )
        paste_panel(
            pred_mask, 2, row,
            "Full prediction {}".format(pred_id),
        )

    canvas.save(str(path))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/absscene_eval.json')
    parser.add_argument('--checkpoint', default='weights/ECONet-absscene-sample-trained.pth')
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-pngs', action='store_true')
    parser.add_argument('--save-masks', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    model, cfg, _ = load(args.checkpoint, cfg, args.device)
    seed_everything(cfg['seed'])
    model.eval()
    ds = AbsScene(args.data, args.split, annotations=True)
    count = len(ds) if args.limit is None else min(args.limit, len(ds))
    if count <= 0:
        raise ValueError('Evaluation needs at least one image')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for i in range(count):
            item = ds[i]
            predicted = model(item['images'][None].to(args.device), i)['masks'].cpu()
            truth = item['masks']
            ari, _ = adjusted_rand_index(truth[None], predicted, exclude_background=True)
            gt_ids, pred_ids, _, iou, dice = matching_iou_ari(
                truth, predicted[0], occlu=args.occlusion
            )
            if args.save_pngs:
                png_path = out / 'comparison_{:05d}.png'.format(i)

            if args.occlusion:
                # Full-object foreground evaluation: preserve overlapping masks.
                gt_ids, pred_ids, iou, dice = matching_full_masks(
                    truth,
                    predicted[0],
                )

                # Standard pixel-label ARI is not an overlapping full-mask metric.
                ari_value = None
            else:
                ari, _ = adjusted_rand_index(
                    truth[None],
                    predicted,
                    exclude_background=True,
                )
                ari_value = float(ari.reshape(-1)[0])

                gt_ids, pred_ids, _, iou, dice = matching_iou_ari(
                    truth,
                    predicted[0],
                    occlu=False,
                )

            rows.append(
                dict(
                    index=i,
                    ari=ari_value,
                    miou=float(iou.mean()) if len(iou) else None,
                    mdice=float(dice.mean()) if len(dice) else None,
                )
            )
            if args.save_masks:
                np.savez_compressed(out/f'masks_{i:05d}.npz', masks=predicted[0].numpy())
    with open(out/'per_image.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {}
    for key in ['ari', 'miou', 'mdice']:
        values = np.array([row[key] for row in rows if row[key] is not None], dtype=float)
        values = values[np.isfinite(values)]
        metrics[key] = dict(mean=float(values.mean()) if len(values) else None,
                            std=float(values.std(ddof=0)) if len(values) else None, n=len(values))
    summary = dict(images=count, split=args.split, occlusion=args.occlusion, metrics=metrics,
                   data_sha256=sha256(args.data), checkpoint_sha256=sha256(args.checkpoint))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
