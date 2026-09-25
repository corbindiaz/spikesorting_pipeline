import gc
import os
import shutil
import numpy as np
from pathlib import Path

REQUIRED = ["spike_times.npy", "spike_clusters.npy", "amplitudes.npy", "spike_templates.npy"]
OPTIONAL = ["pc_features.npy", "template_features.npy"]
CHUNK = 1_000_000


def trim_file(path, n_keep, flatten):
    src = np.load(path, mmap_mode="r")
    shape = (n_keep,) if flatten else (n_keep,) + src.shape[1:]
    tmp = path.with_name(path.stem + ".tmp.npy")

    dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=shape)
    for i in range(0, n_keep, CHUNK):
        j = min(i + CHUNK, n_keep)
        dst[i:j] = src[i:j].reshape(dst[i:j].shape)
    dst.flush()

    # Release the memory maps BEFORE replacing the file (required on Windows)
    del dst, src
    gc.collect()
    os.replace(tmp, path)


def check_spike_count_discrepancy(phy):
    phy = Path(phy)
    files = [f for f in REQUIRED + OPTIONAL if (phy / f).exists()]

    # Read shapes only, then drop the maps immediately
    shapes = {}
    for f in files:
        a = np.load(phy / f, mmap_mode="r")
        shapes[f] = a.shape
        del a
    gc.collect()

    counts = {f: s[0] for f, s in shapes.items()}
    flatten = {f: (f in REQUIRED and len(s) > 1 and s[1:] == (1,)) for f, s in shapes.items()}

    if len(set(counts.values())) == 1 and not any(flatten.values()):
        print("Spike counts match across all files.")
        return

    n_keep = min(counts.values())
    print("Spike counts before:")
    for f, n in counts.items():
        print(f"  {f:24s} {n}")
    print(f"Trimming all per-spike arrays to {n_keep} spikes (dropping from the end).\n")

    backup = phy / "backup_before_trim"
    backup.mkdir(exist_ok=True)
    for f in files:
        if not (backup / f).exists():
            shutil.copy2(phy / f, backup / f)

    for f in files:
        print(f"  writing {f} ...")
        trim_file(phy / f, n_keep, flatten[f])

    print("\nSpike counts after trimming:")
    for f in files:
        a = np.load(phy / f, mmap_mode="r")
        print(f"  {f:24s} {a.shape}")
        del a


check_spike_count_discrepancy(
    r"C:\Users\corbi\Documents\mirilab_stone\code\zahra_run\export\group0\phy"
)