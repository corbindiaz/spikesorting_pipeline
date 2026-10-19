"""
Reassign near-duplicate spikes within each "good" cluster.

If two spikes in the same "good" cluster (group == "good" in cluster_info.tsv)
occur within 0.5 ms of each other (at 30 kHz -> 15 samples), one of them is
reassigned to a new default cluster (ID 100000), chosen 50/50 at random.
No spikes are deleted.

Inputs  (in --dir): spike_times.npy, spike_clusters.npy, spike_templates.npy,
                    cluster_info.tsv
Outputs (in --dir): spike_times_new.npy, spike_clusters_new.npy,
                    spike_templates_new.npy

Usage:
    python reassign_duplicate_spikes.py --dir /path/to/kilosort_output [--seed 0]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

FS = 30000.0  # sampling rate (Hz)
WINDOW_MS = 0.5  # duplicate window (ms)
NEW_CLUSTER_ID = 100000  # default cluster for reassigned spikes
GOOD_LABEL = "good"


def load_good_clusters(tsv_path):
    """Return a set of cluster IDs whose 'group' column equals 'good'."""
    good = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        id_col = next((c for c in ("cluster_id", "id") if c in fields), None)
        if id_col is None:
            raise ValueError(
                f"No 'cluster_id' or 'id' column in {tsv_path} (found: {fields})"
            )
        if "group" not in fields:
            raise ValueError(f"No 'group' column in {tsv_path} (found: {fields})")
        for row in reader:
            grp = (row["group"] or "").strip().lower()
            if grp == GOOD_LABEL:
                good.add(int(row[id_col]))
    return good


def find_reassign_mask(spike_times, spike_clusters, good_clusters, window_samples, rng):
    """Return boolean mask (len = n_spikes) of spikes to move to the new cluster.

    Within each good cluster, spikes are processed in time order and compared
    with the most recent spike that remains in the cluster. If the gap is
    <= window_samples, one of the two is reassigned with 50/50 probability
    (the one that stays becomes the reference for the next spike, so chains of
    close spikes are thinned too).
    """
    move = np.zeros(len(spike_times), dtype=bool)

    for c in np.unique(spike_clusters):
        if int(c) not in good_clusters:
            continue

        idx = np.flatnonzero(spike_clusters == c)
        idx = idx[np.argsort(spike_times[idx], kind="stable")]
        t = spike_times[idx]

        # Skip clusters with no close pairs (fast path)
        if not np.any(np.diff(t) <= window_samples):
            continue

        last_idx, last_t = idx[0], t[0]
        for i, ti in zip(idx[1:].tolist(), t[1:].tolist()):
            if ti - last_t <= window_samples:
                if rng.random() < 0.5:
                    move[last_idx] = True  # reassign the earlier spike
                    last_idx, last_t = i, ti  # the later one stays
                else:
                    move[i] = True  # reassign the later spike
            else:
                last_idx, last_t = i, ti

    return move


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="folder containing the input files")
    p.add_argument("--seed", type=int, default=None, help="random seed (optional)")
    args = p.parse_args()

    d = Path(args.dir)
    spike_times = np.load(d / "spike_times.npy")
    spike_clusters = np.load(d / "spike_clusters.npy")
    spike_templates = np.load(d / "spike_templates.npy")
    good_clusters = load_good_clusters(d / "cluster_info.tsv")

    n = spike_times.shape[0]
    assert (
        spike_clusters.shape[0] == n and spike_templates.shape[0] == n
    ), "spike_times, spike_clusters, spike_templates must have the same length"

    n_existing = int(np.sum(spike_clusters == NEW_CLUSTER_ID))
    if n_existing:
        print(
            f"Note: cluster {NEW_CLUSTER_ID} already has {n_existing} spikes; "
            "reassigned spikes will be added to it."
        )

    window_samples = round(WINDOW_MS * 1e-3 * FS)  # 15 samples
    rng = np.random.default_rng(args.seed)

    # int64 avoids unsigned-integer underflow in differences
    times_flat = spike_times.ravel().astype(np.int64)
    clusters_flat = spike_clusters.ravel()

    move = find_reassign_mask(
        times_flat, clusters_flat, good_clusters, window_samples, rng
    )

    new_clusters = spike_clusters.copy()
    new_clusters.reshape(-1)[move] = NEW_CLUSTER_ID  # preserves dtype and shape

    # Times and templates are unchanged (no spikes removed); saved for a
    # consistent set of *_new files.
    np.save(d / "spike_times_new.npy", spike_times)
    np.save(d / "spike_clusters_new.npy", new_clusters)
    np.save(d / "spike_templates_new.npy", spike_templates)

    print(f"Window: {window_samples} samples ({WINDOW_MS} ms at {FS/1000:.0f} kHz)")
    print(f"Good clusters:      {len(good_clusters)}")
    print(f"Total spikes:       {n}")
    print(
        f"Reassigned to {NEW_CLUSTER_ID}: {int(move.sum())} ({100 * move.mean():.3f}%)"
    )


if __name__ == "__main__":
    main()
