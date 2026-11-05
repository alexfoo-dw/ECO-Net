"""Cross-image part-centric regularization."""
import torch
import torch.nn.functional as F


def part_regularization(features, object_ids, image_ids, temperature=.1, enabled=True):
    """One positive and all negatives from other images, averaged over valid anchors.

    Positive: a part with the same shared discovered-object ID in another image.
    Negatives: known parts with different IDs in other images. Additional
    positives are excluded from the denominator, not treated as negatives.
    Unknown IDs (-1) are ignored. No eligible pairs returns differentiable zero.
    """
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    zero = features.sum()*0
    if not enabled or len(features) < 2:
        return zero
    z = F.normalize(features, dim=-1)
    logits = z @ z.T / temperature
    cross = image_ids[:, None] != image_ids[None, :]
    known = (object_ids[:, None] >= 0) & (object_ids[None, :] >= 0)
    same = object_ids[:, None] == object_ids[None, :]
    positive, negative = cross & known & same, cross & known & ~same
    losses = []
    for i in range(len(z)):
        candidates = torch.where(positive[i])[0]
        if not len(candidates) or not negative[i].any():
            continue
        j = candidates[i % len(candidates)]
        denominator = torch.cat([logits[i, j].reshape(1), logits[i, negative[i]]])
        losses.append(torch.logsumexp(denominator, 0) - logits[i, j])
    return torch.stack(losses).mean() if losses else zero
