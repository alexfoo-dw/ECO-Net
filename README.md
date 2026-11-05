# ECO-Net

Official implementation of **Multi-Part Object Representations via Graph Structures and Co-Part Discovery**, accepted at **NeurIPS 2026**.

**Alex Foo, Wynne Hsu, and Mong-Li Lee**

[Paper / NeurIPS page](https://neurips.cc/virtual/2026/loc/sydney/poster/150151) · [Datasets and sample weights](https://www.dropbox.com/scl/fo/3vbhtmsbmy2kh43mk3ubr/ABggy_q4a1GxkzkzCxI6KaY?rlkey=44wbhdj2nm7o7nwgknux2qnwe&st=naqakinx&dl=0)

## Overview

ECO-Net discovers multi-part objects using explicit graph representations and recurring part relationships. The framework consists of:

1. Graph representation: segment images into parts and represent their features and spatial relationships.
2. Co-part discovery: compose recurring neighboring parts into objects.
3. Part-centric regularization: refine part representations using discovered object assignments.
4. Object memory: store discovered object graphs for subsequent recognition.

This repository includes an AbsScene demonstration covering regular scenes, occluded objects, and complex backgrounds.

## AbsScene demonstration

The default training configuration and dataset batches are filtered for a quick demonstration of object discovery and memory construction.

## Installation

Create a Python environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Install the PyTorch and torchvision builds appropriate for your CUDA environment.

Commands use `cuda:0` by default. To use another device, pass `--device` to the Python entrypoints or set `DEVICE` when using the training script.

## Download datasets and sample weights

Download the AbsScene datasets and sample trained checkpoint from:

[Download datasets and sample weights](https://www.dropbox.com/scl/fo/3vbhtmsbmy2kh43mk3ubr/ABggy_q4a1GxkzkzCxI6KaY?rlkey=44wbhdj2nm7o7nwgknux2qnwe&st=naqakinx&dl=0)

Password:

```text
neurips2026
```

Arrange the downloaded files as follows:

```text
ECO-Net/
├── data/
│   ├── abstract_regular.h5
│   ├── abstract_regular_occlu.h5
│   └── abstract_regular_complex.h5
└── weights/
    └── ECONet-absscene-sample-trained.pth
```

The training and evaluation entrypoints accept other file locations through `--data` and `--checkpoint`.

### HDF5 layout

The AbsScene loader expects the following structure:

```text
train/ or test/
├── image      [N, H, W, 3]
├── segment    [N, H, W]
└── masks      [N, O, H, W, 1]
```

- `imgs` is also accepted as the image-array key.
- Images are stored as RGB `uint8` arrays.
- The highest visible segment ID represents background.
- The final mask layer is the full-image canvas.
- Training reads images only; ground-truth masks are used for evaluation.

Use `--split` to select the required HDF5 group.

## Quick evaluation

Run evaluation on the first five images from each dataset:

```bash
bash scripts/eval_all.sh
```

The script evaluates:

- AbsScene: regular scenes.
- AbsScene-O: occluded objects.
- AbsScene-C: complex backgrounds.

Results are saved to:

```text
results/
├── regular_5/
├── occlusion_5/
└── complex_5/
```

Each directory contains:

```text
per_image.csv
summary.json
comparison_00000.png
comparison_00001.png
...
```

### Evaluate an individual dataset

Regular scenes:

```bash
python eval.py \
  --checkpoint weights/ECONet-absscene-sample-trained.pth \
  --data data/abstract_regular.h5 \
  --split test \
  --limit 5 \
  --out results/regular_5 \
  --save-pngs
```

Occluded objects:

```bash
python eval.py \
  --checkpoint weights/ECONet-absscene-sample-trained.pth \
  --data data/abstract_regular_occlu.h5 \
  --split test \
  --limit 5 \
  --occlusion \
  --out results/occlusion_5 \
  --save-pngs
```

Complex backgrounds:

```bash
python eval.py \
  --checkpoint weights/ECONet-absscene-sample-trained.pth \
  --data data/abstract_regular_complex.h5 \
  --split test \
  --limit 5 \
  --out results/complex_5 \
  --save-pngs
```

Change `--limit` to evaluate a different number of samples. Omit it to evaluate the complete split.

Add `--save-masks` to save the predicted mask arrays as compressed NPZ files.

## Visualizations

For regular and complex-background scenes, each comparison PNG contains:

```text
Input image | Ground-truth foreground matches | Predicted foreground matches
```

Matched objects share colors. Background and unmatched regions in the mask panels are black.

For occlusion, each matched object is displayed independently:

```text
Input image | Full ground-truth object 1 | Full predicted object 1
Input image | Full ground-truth object 2 | Full predicted object 2
...
```

The occlusion panels display complete masks, including occluded regions, without collapsing overlapping objects into a single label map.

## Evaluation metrics

Regular and complex-background evaluation report:

- Foreground Adjusted Rand Index.
- Matched mean Intersection-over-Union.
- Matched mean Dice.

Occlusion evaluation reports full-object mean IoU and mean Dice:

- Ground-truth and predicted foreground masks are compared independently.
- Both visible and occluded object regions are included.
- Background masks are excluded from matching.
- A one-to-one IoU-optimal assignment determines the corresponding Dice matches.
- Missing ground-truth matches contribute zero.

Standard pixel-label ARI is left empty for full-object occlusion evaluation because the masks can overlap.

Scores are reported as fractions. The summary contains means and population standard deviations across evaluated images, together with dataset and checkpoint SHA256 checksums.

Visualization filtering does not remove missed or zero-overlap objects from the metric calculation.

## Train the sample model

Run the filtered AbsScene demonstration:

```bash
bash scripts/train.sh
```

To specify the output directory and compact model filename:

```bash
MODEL_NAME=ECONet-absscene-demo.pth \
OUT_DIR=runs/absscene_demo \
bash scripts/train.sh
```

If the training dataset is stored separately:

```bash
TRAIN_DATA=data/abstract_train.h5 \
MODEL_NAME=ECONet-absscene-demo.pth \
OUT_DIR=runs/absscene_demo \
bash scripts/train.sh
```

The launcher supports:

| Environment variable | Purpose |
|---|---|
| `DATA_DIR` | Dataset directory |
| `TRAIN_DATA` | Training HDF5 file |
| `CONFIG` | Training configuration |
| `OUT_DIR` | Output directory |
| `MODEL_NAME` | Final compact checkpoint filename |
| `DEVICE` | Training device |

### Training outputs

```text
runs/absscene_demo/
├── config.json
├── provenance.json
├── training.jsonl
├── last.pth
└── ECONet-absscene-demo.pth
```

- `last.pth` contains the resumable training state.
- The named final checkpoint contains the encoder and compact object memory.
- `training.jsonl` records the step, epoch, dataloader batch, loss, and memory size.

Evaluate the newly trained model:

```bash
CHECKPOINT=runs/absscene_demo/ECONet-absscene-demo.pth \
bash scripts/eval_all.sh
```

### Resume training

Use the same dataset and configuration:

```bash
MODEL_NAME=ECONet-absscene-demo.pth \
OUT_DIR=runs/absscene_demo \
bash scripts/train.sh \
  --resume runs/absscene_demo/last.pth
```

Use the resumable `last.pth`, rather than the compact inference checkpoint.

## Part-centric regularization

The regularizer is implemented in:

```text
econet/losses.py
```

Its interface is:

```python
part_regularization(
    features,
    object_ids,
    image_ids,
    temperature=0.1,
    enabled=True,
)
```

Positive pairs are parts assigned to the same discovered object in different images. Negative pairs are parts assigned to different objects in different images. Ground-truth annotations are not used.

## Optional features

Feature families are controlled independently in the configuration:

```json
{
  "shape": true,
  "texture": false,
  "dino": false
}
```

At least one feature family must be enabled.

The sample configuration uses shape features. Texture and distilled DINO features can be enabled for additional experiments. Changing feature families requires rebuilding object memory.

### DINO setup

DINO extraction uses the feature wrapper from [CutLER](https://github.com/facebookresearch/CutLER/tree/main/maskcut).

Obtain a specific upstream revision:

```bash
bash scripts/setup_dino.sh <CutLER-commit-hash>
```

Then configure:

```text
features.dino
features.cutler_path
features.dino_weights
features.dino_arch
features.dino_patch_size
```

## Repository structure

```text
econet/
├── model.py          # Public ECONet model
├── encoder.py        # Graph discovery, memory update, and recognition
├── features.py       # Shape, texture, and DINO features
├── graph_ids.py      # Shared discovered-object assignments
├── losses.py         # Part-centric regularization
├── data.py           # AbsScene loader
├── checkpoint.py     # Checkpoint loading and compact export
├── metrics.py        # Segmentation metrics
└── pegbis/           # Pixel graph segmentation

configs/
├── absscene.json
├── absscene_eval.json
└── absscene_regularized.json

scripts/
├── train.sh
├── eval_all.sh
├── compact_checkpoint.py
└── setup_dino.sh

train.py
eval.py
tests/
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

If you use this code, datasets, or model in your research, please cite:

```bibtex
@inproceedings{foo2026multipart,
  title     = {Multi-Part Object Representations via Graph Structures and Co-Part Discovery},
  author    = {Foo, Alex and Hsu, Wynne and Lee, Mong-Li},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```