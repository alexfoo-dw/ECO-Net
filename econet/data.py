"""AbsScene HDF5 reader. Training never reads ground-truth masks."""
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AbsScene(Dataset):
    def __init__(self, path, split='train', annotations=False):
        self.path = str(Path(path).expanduser())
        self.split, self.annotations = split, annotations
        with h5py.File(self.path, 'r') as file:
            if split not in file:
                raise KeyError(f'Missing {split}; available groups: {list(file)}')
            group = file[split]
            self.image_key = 'imgs' if 'imgs' in group else 'image'
            self.length = len(group[self.image_key])
            if self.length == 0:
                raise ValueError('Empty dataset')
            if annotations and not all(k in group for k in ['segment', 'masks']):
                raise ValueError('Evaluation requires segment and masks')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Never share a live HDF5 handle between DataLoader workers.
        with h5py.File(self.path, 'r') as file:
            group = file[self.split]
            image = np.asarray(group[self.image_key][index])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError('Expected HWC RGB uint8 images')
            out = dict(images=torch.from_numpy(image.transpose(2, 0, 1).copy()).float()/127.5-1,
                       index=int(index))
            if self.annotations:
                segment = np.asarray(group['segment'][index]).squeeze()
                if segment.shape != image.shape[:2]:
                    raise ValueError('Image and segment shape mismatch')
                # AbsScene stores background as the highest visible segment ID.
                background = (segment == segment.max())[None, None]
                masks = np.asarray(group['masks'][index])
                if masks.ndim != 4 or masks.shape[-1] != 1 or masks.shape[1:3] != image.shape[:2]:
                    raise ValueError('Expected masks [objects,H,W,1] matching image dimensions')
                if not np.all(masks[-1] > 0):
                    raise ValueError('Expected the final full-canvas mask')
                objects = masks[:-1].transpose(0, 3, 1, 2) > 0
                out['masks'] = torch.from_numpy(np.concatenate([background, objects], axis=0).astype('float32'))
            return out
