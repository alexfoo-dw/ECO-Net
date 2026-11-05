"""Foreground ARI and one-to-one matched IoU/Dice."""
import torch
import numpy as np
import scipy.optimize
from sklearn.metrics import adjusted_rand_score

def adjusted_rand_index(true_mask, pred_mask, exclude_background=True):
    """
    compute the ARI for a single image. N.b. ARI 
    is invariant to permutations of the cluster IDs.

    See https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index.

    true_mask: LongTensor of shape [N, num_entities, 1, H, W] onehot
        with 1 or 255 indicating entity
        or 
        LongTensor of shape [N, 1, H, W] where
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: FloatTensor of shape [N, K, 1, H, W]  (mask probs)

    Returns: ari [N]

    """
    if len(true_mask.shape) == 5:
        N, max_num_entities, _, H, W = true_mask.shape
        true_groups = max_num_entities
        true_mask = true_mask.permute(0, 2, 3, 4, 1).contiguous()
        gt_value = 1 if true_mask[..., 0].max() == 1 else 255
        for i in range(max_num_entities):
            mask = true_mask[..., i] == gt_value
            true_mask[..., i][mask] = i
        true_mask, _ = torch.max(true_mask, dim=-1)
    elif len(true_mask.shape) == 4:
        N, _, H, W = true_mask.shape
        max_num_entities = int(true_mask.max() + 1)
    true_group_ids = true_mask.view(N, H * W).long()
    true_mask_oh = torch.nn.functional.one_hot(true_group_ids).float()
    if exclude_background:
        true_mask_oh[..., 0] = 0
    if len(pred_mask.shape) == 5:
        pred_mask = pred_mask.squeeze(2)
        pred_groups = pred_mask.shape[1]
        pred_mask = torch.argmax(pred_mask, dim=1)
    elif len(pred_mask.shape) == 4:
        pred_groups = int(pred_mask.max()) + 1
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()
    n_points = H * W
    if n_points <= max_num_entities and n_points <= pred_groups:
        raise ValueError("adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint")
    n_points = torch.sum(true_mask_oh, dim=[1, 2])
    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_mask_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)
    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)
    check_single_cluster = torch.sum((nij > 0).int(), dim=[1, 2])
    check_single_cluster = (1 == check_single_cluster).int()
    ari[ari != ari] = 0
    ari = check_single_cluster * torch.ones_like(ari) + (1 - check_single_cluster) * ari
    return (ari, (nij, pred_group_oh, true_mask_oh))

def matching_iou_ari(true_mask, pred_mask, occlu=False):
    """
    true_mask: LongTensor of shape [num_entities, 1, H, W]
        background == 0
        object 1 == 1
        object 2 == 2
        ...
    pred_mask: [K,1,H,W]
    prescence: [max # of objects]  binary mask with 1's left-to-right. First index for BG (ignore).
    """
    true_mask_ori = true_mask.clone()
    pred_mask_ori = pred_mask.clone()
    if len(true_mask.shape) == 4:
        true_mask = true_mask.squeeze(1)
        max_num_entities, H, W = true_mask.shape
        true_mask = true_mask.permute(1, 2, 0).contiguous()
        presence = torch.ones(max_num_entities)
        gt_value = 1 if true_mask[..., 0].max() == 1 else 255
        for i in range(max_num_entities):
            if (true_mask[..., i] == 0).all():
                presence[i] = 0
            mask = true_mask[..., i] == gt_value
            true_mask[..., i][mask] = i
        true_mask, _ = torch.max(true_mask, dim=-1)
    elif len(true_mask.shape) == 3:
        _, H, W = true_mask.shape
        max_num_entities = int(true_mask.max() + 1)
        presence = torch.ones(max_num_entities)
        for i in range(max_num_entities):
            if i not in true_mask:
                presence[i] = 0
    if len(pred_mask.shape) == 4:
        pred_mask = pred_mask.squeeze(1)
        K, _, _ = pred_mask.shape
        pred_mask = torch.argmax(pred_mask, dim=0)
    elif len(pred_mask.shape) == 3:
        K = int(pred_mask.max()) + 1
    pred_mask = pred_mask.view(H * W).long().cpu()
    true_object_pixels = true_mask.view(H * W).long().cpu()
    pixel_ids = torch.arange(H * W)
    num_objects = presence.sum() - 1
    start_idx = 1
    iou = torch.zeros(int(num_objects), K if K >= int(num_objects) else int(num_objects))
    dice = torch.zeros(int(num_objects), K if K >= int(num_objects) else int(num_objects))
    if num_objects == 0:
        return ([], [], [], [], [])
    iou_assigned = 0
    for i in range(start_idx, presence.shape[0]):
        viz = presence[i]
        if viz == 0:
            continue
        if len(true_mask_ori.shape) == 4 and occlu:
            true_object = pixel_ids[(true_mask_ori[i] > 0).view(H * W)]
        else:
            true_object = pixel_ids[true_object_pixels == i]
        for j in range(K):
            if len(pred_mask_ori.shape) == 4 and occlu:
                pred_object = pixel_ids[(pred_mask_ori[j] > 1 / K).view(-1)]
            else:
                pred_object = pixel_ids[pred_mask == j]
            intersection = np.intersect1d(true_object, pred_object).shape[0]
            union = len(torch.cat([true_object, pred_object]).unique(sorted=False))
            if union == 0:
                iou[iou_assigned, j] = -1000000
                dice[iou_assigned, j] = -1000000
            else:
                iou[iou_assigned, j] = intersection / union
                dice[iou_assigned, j] = 2 * intersection / (len(true_object) + len(pred_object))
        iou_assigned += 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(iou.data.cpu().numpy(), maximize=True)
    assert row_ind.shape[0] == num_objects
    assert col_ind.shape[0] == num_objects
    iou_matched = iou[row_ind, col_ind]
    dice_matched = dice[row_ind, col_ind]
    row_ind += 1
    return (row_ind, col_ind, presence, iou_matched, dice_matched)

def matching_full_masks(true_mask, pred_mask):
    """Match independent full-object foreground masks using maximum total IoU.

    Inputs:
        true_mask: [G,1,H,W], background at index 0
        pred_mask: [K,1,H,W], background at index 0

    Returns:
        gt_ids:   original GT foreground indices
        pred_ids: matched prediction indices, or -1 for missing predictions
        iou:      one score per nonempty GT foreground object
        dice:     Dice scores under the same IoU-optimal assignment

    Background and empty masks are excluded.
    Missing GT matches contribute zero, including zero-overlap assignments.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    gt = true_mask.detach().cpu().numpy()[:, 0] > 0.5
    pred = pred_mask.detach().cpu().numpy()[:, 0] > 0.5

    # Preserve original indices for the PNG visualization.
    gt_ids = np.flatnonzero(gt[1:].any(axis=(1, 2))) + 1
    available_pred_ids = np.flatnonzero(
        pred[1:].any(axis=(1, 2))
    ) + 1

    n_gt = len(gt_ids)
    n_pred = len(available_pred_ids)

    pred_ids = np.full(n_gt, -1, dtype=np.int64)
    matched_iou = np.zeros(n_gt, dtype=np.float64)
    matched_dice = np.zeros(n_gt, dtype=np.float64)

    if n_gt == 0 or n_pred == 0:
        return gt_ids, pred_ids, matched_iou, matched_dice

    # Each object retains its complete mask, even where objects overlap.
    a = gt[gt_ids].reshape(n_gt, -1).astype(np.float64)
    b = pred[available_pred_ids].reshape(n_pred, -1).astype(np.float64)

    intersection = a @ b.T
    combined_area = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :]
    union = combined_area - intersection

    pair_iou = intersection / np.maximum(union, 1)
    pair_dice = 2 * intersection / np.maximum(combined_area, 1)

    # Dummy zero-score columns ensure missed GT objects remain in the mean.
    assignment_scores = np.zeros(
        (n_gt, max(n_gt, n_pred)),
        dtype=np.float64,
    )
    assignment_scores[:, :n_pred] = pair_iou

    # Negate scores for compatibility with older SciPy versions.
    rows, columns = linear_sum_assignment(-assignment_scores)

    for row, column in zip(rows, columns):
        if column < n_pred:
            pred_ids[row] = available_pred_ids[column]
            matched_iou[row] = pair_iou[row, column]
            matched_dice[row] = pair_dice[row, column]

    return gt_ids, pred_ids, matched_iou, matched_dice
    
if __name__ == '__main__':
    true_mask = torch.LongTensor([[1, 1, 1], [0, 2, 2], [0, 0, 0]])
    true_mask = true_mask.view(1, 1, 3, 3)
    pred_mask = torch.FloatTensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0], [1, 1, 1]], [[1.0, 1.0, 0.0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 0, 0]]])
    pred_mask = pred_mask.view(1, 3, 1, 3, 3)
