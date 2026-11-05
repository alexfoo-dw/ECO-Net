import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


def small_config():
    cfg = json.loads(Path('configs/absscene.json').read_text())
    cfg['net'].update(input_size=[3, 16, 16], K=32, mem_count_max=4,
                      memref_l0_count_max=8, memref_l1_count_max=16,
                      memref_vecs_unit_z_size=8)
    cfg['training'].update(steps=1, selected_batches=[], num_workers=0)
    return cfg


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ModelTests(unittest.TestCase):
    def test_regularization_gradient_and_disabled(self):
        from econet.losses import part_regularization
        features = torch.randn(4, 16, requires_grad=True)
        ids, images = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1])
        loss = part_regularization(features, ids, images)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertEqual(float(part_regularization(features, ids, images, enabled=False)), 0)
        self.assertEqual(float(part_regularization(features, ids, torch.zeros(4, dtype=torch.long))), 0)

    def test_missing_positive_or_negative(self):
        from econet.losses import part_regularization
        z = torch.randn(4, 16, requires_grad=True)
        images = torch.tensor([0, 0, 1, 1])
        self.assertEqual(float(part_regularization(z, torch.zeros(4, dtype=torch.long), images)), 0)
        self.assertEqual(float(part_regularization(z, torch.arange(4), images)), 0)

    def test_feature_switches(self):
        from econet import ECONet
        for shape, texture, dino in [(True, False, False), (False, True, False), (True, True, False), (False, False, True), (True, True, True)]:
            cfg = small_config()
            cfg['features'].update(shape=shape, texture=texture, dino=dino)
            with patch('econet.features.load_dino', return_value=torch.nn.Identity()):
                model = ECONet(cfg)
            encoder = model.encoder
            encoder._rgb = torch.randn(1, 256, 3)
            encoder._distilled = torch.randn(1, 256, 24)
            nodes = encoder.combine_features(torch.randn(3, 8, 2), torch.ones(3, 256), torch.zeros(3, dtype=torch.long))
            self.assertEqual(tuple(nodes.shape), (3, 8, 2))
            self.assertTrue(torch.isfinite(nodes).all())
        cfg['features'].update(shape=False, texture=False, dino=False)
        with self.assertRaises(ValueError):
            ECONet(cfg)

    def test_train_forward(self):
        from econet import ECONet
        model = ECONet(small_config(), batch_size=2)
        model.train()
        images = torch.ones(2, 3, 16, 16)
        images[:, :, 3:6, 3:5] = torch.tensor([1., -1., -1.])[None, :, None, None]
        images[:, :, 3:6, 5:7] = torch.tensor([-1., 1., -1.])[None, :, None, None]
        output = model(images)
        self.assertEqual(tuple(output['masks'].shape[-2:]), (16, 16))
        self.assertTrue(torch.isfinite(output['masks']).all())

    def test_checkpoint_and_distributed_prefix(self):
        from econet import ECONet
        from econet.checkpoint import save, load
        cfg = small_config()
        model = ECONet(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            save(path, model, cfg, 0, compact=True)
            loaded, _, _ = load(path, cfg)
            torch.testing.assert_close(loaded.encoder.encoder_pt_1[0].weight, model.encoder.encoder_pt_1[0].weight)
            original = {'module.'+key:value for key, value in model.state_dict().items() if key != 'encoder.part_projector.weight'}
            torch.save({'model':original}, path)
            loaded, _, _ = load(path, cfg)
            torch.testing.assert_close(loaded.encoder.part_projector.weight, torch.eye(16))
            loaded.eval()
            with self.assertRaisesRegex(RuntimeError, 'memory is empty'):
                loaded(torch.zeros(1, 3, 16, 16))

    def test_perfect_metrics(self):
        from econet.metrics import adjusted_rand_index, matching_iou_ari
        masks = torch.zeros(2, 1, 8, 8)
        masks[1, 0, 2:4, 2:4] = 1
        masks[0] = 1-masks[1]
        ari, _ = adjusted_rand_index(masks[None], masks[None])
        self.assertAlmostEqual(float(ari.reshape(-1)[0]), 1.)
        _, _, _, iou, dice = matching_iou_ari(masks, masks)
        self.assertAlmostEqual(float(iou.mean()), 1.)
        self.assertAlmostEqual(float(dice.mean()), 1.)
