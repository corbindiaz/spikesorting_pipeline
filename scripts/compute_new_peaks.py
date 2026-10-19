import argparse
import ast
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


def parse_params_py(path: Path) -> dict:
    """Read simple `key = value` lines from a Phy/Kilosort params.py."""
    params = {}
    if not path.exists():
        return params
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*(#.*)?$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        try:
            params[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            params[key] = val.strip("'\"")
    return params


def load_spike_labels(folder: Path) -> np.ndarray:
    for name in ("spike_clusters.npy", "spike_templates.npy"):
        f = folder / name
        if f.exists():
            if name != "spike_clusters.npy":
                print(
                    f"[warn] spike_clusters.npy not found; using {name} "
                    "for cluster labels (labels only, no waveforms)."
                )
            return np.load(f).astype(np.int64).ravel()
    sys.exit("Could not find spike_clusters.npy or spike_templates.npy")


def write_tsv(path: Path, header: str, ids, values, fmt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["cluster_id", header])
        for cid, v in zip(ids, values):
            w.writerow([int(cid), fmt(v)])


def mean_waveform(dat, spike_samples, n_pre, n_post, pad, sos, valid_chans):
    """Average filtered waveform (time x len(valid_chans)) over the given spikes.
    """
    n_total = dat.shape[0]
    total = np.zeros((n_pre + n_post + 1, len(valid_chans)), dtype=np.float64)
    n_used = 0
    for s in spike_samples:
        start = s - pad
        stop = s + n_post + pad + 1
        if start < 0 or stop > n_total:
            continue  # too close to file edges
        raw = np.asarray(dat[start:stop, :][:, valid_chans], dtype=np.float64)
        filt = sosfiltfilt(sos, raw, axis=0)
        # spike sits at index `pad`; keep [-n_pre, +n_post]
        total += filt[pad - n_pre : pad + n_post + 1, :]
        n_used += 1
    if n_used == 0:
        return None, 0
    return total / n_used, n_used


def peak_and_amplitude(mean_wf):
    """
    mean_wf: time x channels.
    Returns (channel_col_index, negative_peak, valley_value, amplitude).
    """
    # largest negative peak = most negative value on each channel
    min_per_chan = mean_wf.min(axis=0)
    ch = int(np.argmin(min_per_chan))
    trace = mean_wf[:, ch]
    peak_ind = int(np.argmin(trace))
    peak_val = float(trace[peak_ind])

    after = trace[peak_ind + 1 :]
    if after.size == 0:
        return ch, peak_val, np.nan, np.nan
    valley_val = float(after.max())  # positive rebound after the negative peak
    return ch, peak_val, valley_val, abs(peak_val - valley_val)


def main():
    ### parse arguments
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Kilosort output folder (contains recording.dat); default: current directory",
    )
    ap.add_argument("--dat", default="recording.dat", help="raw binary file name")
    ap.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="channels in the .dat (default: params.py n_channels_dat)",
    )
    ap.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="Hz (default: params.py sample_rate, else 30000)",
    )
    ap.add_argument(
        "--dtype", default=None, help="default: params.py dtype, else int16"
    )
    ap.add_argument(
        "--exclude-last",
        action="store_true",
        help="exclude the last channel (e.g. Neuropixels sync channel) "
        "when searching for the peak channel",
    )
    ap.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="added to spike times if the .dat was trimmed (like startSample in MATLAB)",
    )
    ap.add_argument(
        "--max-spikes",
        type=int,
        default=500,
        help="max spikes per cluster; randomly subsampled if exceeded, "
        "otherwise all spikes are used",
    )
    ap.add_argument("--pre", type=int, default=40, help="samples before spike")
    ap.add_argument("--post", type=int, default=60, help="samples after spike")
    ap.add_argument(
        "--band", type=float, nargs=2, default=(300, 3000), help="band-pass Hz"
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply amplitude by this (e.g. bit-to-uV gain)",
    )
    ap.add_argument("--seed", type=int, default=0, help="random seed for subsampling")
    args = ap.parse_args()

    ### loading in data
    folder = args.folder
    params = parse_params_py(folder / "params.py")

    n_chans = args.n_channels or params.get("n_channels_dat")
    if n_chans is None:
        sys.exit("Number of channels unknown: pass --n-channels or provide params.py")
    n_chans = int(n_chans)
    fs = float(args.sample_rate or params.get("sample_rate", 30000.0))
    dtype = np.dtype(args.dtype or params.get("dtype", "int16"))
    offset = int(params.get("offset", 0))

    dat_path = folder / args.dat
    if not dat_path.exists():
        sys.exit(f"{dat_path} not found")
    n_bytes = dat_path.stat().st_size - offset
    n_samples = n_bytes // (dtype.itemsize * n_chans)
    dat = np.memmap(
        dat_path, dtype=dtype, mode="r", offset=offset, shape=(n_samples, n_chans)
    )
    print(
        f"{dat_path.name}: {n_samples} samples x {n_chans} channels, "
        f"{n_samples / fs:.1f} s @ {fs:g} Hz"
    )

    cm_path = folder / "channel_map.npy"
    if cm_path.exists():
        valid_chans = np.sort(np.load(cm_path).astype(int).ravel())
        valid_chans = valid_chans[valid_chans < n_chans]
    else:
        valid_chans = np.arange(n_chans)
    if args.exclude_last:
        valid_chans = valid_chans[valid_chans != n_chans - 1]
    print(f"Searching {len(valid_chans)} channels for the peak")

    spike_times = np.load(folder / "spike_times.npy").astype(np.int64).ravel()
    spike_times = spike_times + args.start_sample
    labels = load_spike_labels(folder)
    if len(labels) != len(spike_times):
        sys.exit("spike_times and cluster labels have different lengths")
    cluster_ids = np.unique(labels)

    ### filtering and preprocessing
    sos = butter(3, np.array(args.band) / (fs / 2), btype="bandpass", output="sos")
    pad = 2 * args.pre  # extra context so filter edge effects fall outside the crop
    min_isi = round(0.001 * fs)  # 1 ms
    rng = np.random.default_rng(args.seed)

    out_ch, out_amp = [], []
    for i, cid in enumerate(cluster_ids, 1):
        st = np.sort(spike_times[labels == cid])
        # drop duplicate detections within 1 ms of the previous spike
        if st.size > 1:
            st = st[np.concatenate(([True], np.diff(st) >= min_isi))]
        # subsample only if the unit has more than max_spikes spikes
        if st.size > args.max_spikes:
            st = np.sort(rng.choice(st, args.max_spikes, replace=False))

        mwf, n_used = mean_waveform( # noqa: RUF059
            dat, st, args.pre, args.post, pad, sos, valid_chans
        )  
        if mwf is None:
            print(f"[warn] cluster {cid}: no usable spikes")
            out_ch.append(-1)
            out_amp.append(np.nan)
            continue

        col, peak_val, valley_val, amp = peak_and_amplitude(mwf)  # noqa: RUF059
        out_ch.append(int(valid_chans[col]))  # back to .dat channel index (0-based)
        out_amp.append(amp * args.scale)

        if i % 25 == 0 or i == len(cluster_ids):
            print(f"  {i}/{len(cluster_ids)} clusters done")

    write_tsv(
        folder / "cluster_peak.tsv", "peak", cluster_ids, out_ch, lambda v: str(int(v))
    )
    write_tsv(
        folder / "cluster_peakAmplitude.tsv",
        "peakAmplitude",
        cluster_ids,
        out_amp,
        lambda v: "nan" if np.isnan(v) else f"{v:.4f}",
    )
    print(f"Saved {folder / 'cluster_p.tsv'}")
    print(f"Saved {folder / 'cluster_peakAmplitude.tsv'}")


if __name__ == "__main__":
    main()
