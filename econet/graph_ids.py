"""Shared pseudo-object identities from paired co-part graph occurrences."""
import numpy as np


def shared_object_ids(pair_graphs, part_count):
    """Union all parts in each recurrent graph, merging overlapping graphs.

    Graph entries are flattened part IDs (image_index * parts_per_image + part).
    Negative/padded IDs are ignored. Unassigned parts retain -1.
    """
    parent = list(range(part_count))
    known = set()
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for graph in pair_graphs:
        ids = [int(i) for i in np.asarray(graph).reshape(-1) if 0 <= i < part_count]
        if not ids:
            continue
        known.update(ids)
        for i in ids[1:]:
            a, b = find(ids[0]), find(i)
            parent[max(a, b)] = min(a, b)
    labels = np.full(part_count, -1, dtype=np.int64)
    for i in known:
        labels[i] = find(i)
    return labels
