"""Small generated-HDF5 CLI training test."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('h5py'), 'PyTorch and h5py required')
class PipelineTests(unittest.TestCase):
    def test_training_cli(self):
        import h5py
        from test_model import small_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            images = np.full((4, 16, 16, 3), 255, np.uint8)
            images[:, 3:6, 3:5] = [255, 0, 0]
            images[:, 3:6, 5:7] = [0, 255, 0]
            data = path/'tiny.h5'
            with h5py.File(data, 'w') as file:
                file.create_group('train')['image'] = images
            cfg = path/'config.json'
            cfg.write_text(json.dumps(small_config()))
            subprocess.run([sys.executable, 'train.py', '--config', str(cfg), '--data', str(data),
                            '--out', str(path/'run'), '--device', 'cpu'], check=True)
            self.assertTrue((path/'run/ECONet-absscene-sample-trained.pth').is_file())
