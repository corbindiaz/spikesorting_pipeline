import json
import re
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from ibldsp.voltage import detect_bad_channels as ibl
from matplotlib.lines import Line2D
from scipy.signal import butter, sosfiltfilt, welch
from spikeinterface.sortingcomponents.motion import interpolate_motion
from tqdm.auto import tqdm

from utils import timed
from visualization import plot_peak_localization, save_probe_figure, save_widget


def preprocess(recording_path_, output_folder, params, time_master, step=0):
    params_pre = params["preprocess"]
    preprocessed_folder = output_folder / "preprocessed"
    preprocessed_folder.mkdir(parents=True, exist_ok=True)

    plots = output_folder / "plots"
    motion_plots_folder = plots / "motion"
    plots_pre = plots / "cleaning"
    plots.mkdir(parents=True, exist_ok=True)
    motion_plots_folder.mkdir(parents=True, exist_ok=True)
    plots_pre.mkdir(parents=True, exist_ok=True)

    if params_pre["bad_channels"]["debug_mode"]:
        print("WARNING:")
        print("DEBUG MODE initiated. Pipeline will terminate before motion detection.")

    use_sync = params_pre.get("sync_channel", False)

    # Load everything (385 channels if sync is requested)
    full = si.read_spikeglx(
        recording_path_, stream_id=STREAM_ID, load_sync_channel=use_sync
    )

    print("Shifting time to ensure 0sec start...")
    print(f"Old start time: {full.get_times()[0]} seconds")
    full.shift_times(-full.get_times()[0])
    print(f"New start time: {full.get_times()[0]} seconds")

    # Trim the ENTIRE recording (neural + sync together)
    params_trim = params_pre["trimming"]
    n_samples_original = full.get_num_samples()
    start_frame, end_frame = resolve_trim_frames(full, params_trim)
    if params_trim["trim_recording"]:
        fs = full.get_sampling_frequency()
        print(
            f"Trimming recording to frames {start_frame}..{end_frame} "
            f"({start_frame / fs:.3f} s to {end_frame / fs:.3f} s)..."
        )
        full = full.frame_slice(start_frame, end_frame)

    # Only now separate the sync channel
    if use_sync:
        recording, sync = split_off_sync(full, recording_path_)
        save_sync_channel(
            sync, output_folder, start_frame, end_frame, n_samples_original
        )
    else:
        recording = full
        if (output_folder / "sync").exists():
            print(
                "WARNING: an old 'sync' folder exists from a previous run and does NOT "
                "match this preprocessed recording."
            )

    print(recording)
    save_probe_figure(recording, plots_pre, "probe_map_raw")
    print(f"Probe Map saved to: {plots_pre / 'probe_map_raw.png'}")

    if params_pre["split_by_shank"]:
        groups = recording.split_by("group")
    else:
        groups = {"wholeprobe": recording}

    for i, (group_name, group) in enumerate(groups.items()):
        group_name = str(group_name)
        recording_name = f"{group_name}_group{i}"

        group_time = time_master.setdefault(f"Group{i}", {})
        preprocessing_time = group_time.setdefault("Preprocessing", {})
        group_start = time.perf_counter()

        preprocessed_path = preprocessed_folder / recording_name
        motion_folder = preprocessed_folder / "motion" / recording_name

        print("-" * 80)
        print(f"Preprocessing shank {group_name}")
        print(f"Channels: {group.get_num_channels()}")

        try:
            # Cleaning (Phase-shift, Filtering, Bad Channel Detection, CAR)
            with timed(preprocessing_time, "Cleaning"):

                w = si.plot_traces(
                    group,
                    time_range=(10, 10.5),
                    mode="map",
                    order_channel_by_depth=True,
                    return_in_uV=True,
                    backend="matplotlib",
                )

                save_widget(w, plots_pre / f"{recording_name}_traces_raw.png")
                print(
                    f"Raw traces saved to: {plots_pre / f'{recording_name}_traces_raw.png'}"
                )

                print("Phase-shift correcting...")
                group = spre.phase_shift(group)

                if not params_pre["split_by_shank"]:
                    print(
                        "NOTE: Temporarily splitting by shank to perform certain preprocessing steps."
                    )
                    temporary_groups = recording.split_by("group")
                else:
                    temporary_groups = {group_name: group}

                temporary_groups_list = []
                for i, (temp_group_name, temp_group) in enumerate(
                    temporary_groups.items()
                ):
                    if not params_pre["split_by_shank"]:
                        print("-" * 40)
                        print(f"Processing {temp_group_name}:")
                    params_bad = params_pre["bad_channels"]
                    remove_cfg = params_bad.get("remove_bad_channels", True)

                    # Check if a custom list was passed
                    is_custom_list = isinstance(remove_cfg, (list, tuple, np.ndarray))
                    generate_diag = params_bad.get("generate_diagnostics", True)

                    channel_labels = None
                    custom_channels = None

                    if is_custom_list and not generate_diag:
                        print(
                            f"Skipping bad channel detection. Removing channels {remove_cfg} instead."
                        )
                        bad_channel_ids = [
                            c for c in remove_cfg if c in temp_group.channel_ids
                        ]
                        custom_channels = bad_channel_ids

                        # Create fallback channel_labels: 'good' by default, 'bad' for custom passed channels
                        channel_labels = np.full(
                            temp_group.get_num_channels(), "good", dtype="U5"
                        )
                        custom_mask = np.isin(temp_group.channel_ids, custom_channels)
                        channel_labels[custom_mask] = "bad"

                    else:
                        if is_custom_list and generate_diag:
                            print(
                                "Performing bad channel detection to generate diagnostics..."
                            )
                        else:
                            print("Detecting bad channels...")

                        # Detect IBL bad channels & SpikeInterface MAD noise
                        bad_channel_ids, channel_labels, feats, si_mad_bad_ids = (
                            detect_bad_channels_ibl(temp_group, params_bad)
                        )
                        bad = len(temp_group.channel_ids) - len(
                            channel_labels[channel_labels == "good"]
                        )
                        print(f"Detected {bad} bad channel(s) via IBL...")

                        # Save channel labels to CSV
                        si_mad_mask = np.isin(temp_group.channel_ids, si_mad_bad_ids)
                        channel_label_csv = pd.DataFrame(
                            {
                                "channel_id": temp_group.channel_ids,
                                "channel_label": channel_labels,
                                "si_mad_noise": np.where(si_mad_mask, "noise", "good"),
                            }
                        )
                        channel_label_path = (
                            preprocessed_folder
                            / f"{temp_group_name}_channel_labels.csv"
                        )
                        channel_label_csv.to_csv(channel_label_path, index=False)
                        print(f"Channel labels saved to: {channel_label_path}")

                    # Resolve target channels to remove
                    if is_custom_list:
                        channels_to_remove = [
                            c for c in remove_cfg if c in temp_group.channel_ids
                        ]
                        custom_channels = channels_to_remove
                    else:
                        channels_to_remove = bad_channel_ids

                    # Run diagnostics if enabled
                    if generate_diag and feats is not None:
                        bad_channels_diagnostic(
                            temp_group,
                            channel_labels,
                            feats,
                            plots_pre / "bad_channel_diagnostics",
                            params_bad,
                            name=str(temp_group_name),
                            custom_star_channels=custom_channels,
                        )

                    # Save probe layout figure with thick borders on custom channels
                    save_probe_figure(
                        temp_group,
                        plots_pre,
                        f"{temp_group_name}_bad_channels_map",
                        channel_labels=channel_labels,
                        custom_channels=custom_channels,
                    )
                    print(
                        f"Probe Map with detected channel labels saved to: {plots_pre / f'{temp_group_name}_bad_channels_map.png'}"
                    )

                    # Channel Removal Logic
                    if remove_cfg:
                        if len(channels_to_remove) > 0:
                            frac = (
                                len(channels_to_remove) / temp_group.get_num_channels()
                            )
                            if frac > params_bad["bad_channel_limit"]:
                                raise RuntimeError(
                                    f"Too many bad channels targeted for removal: "
                                    f"{len(channels_to_remove)}/{temp_group.get_num_channels()} "
                                    f"({frac:.1%}), exceeding limit of "
                                    f"{params_bad['bad_channel_limit']:.1%}. "
                                    f"Please increase 'bad_channel_limit' to override this message, "
                                    f"or set 'remove_bad_channels' to False."
                                )
                            print(
                                f"Removing {len(channels_to_remove)} targeted channels..."
                            )
                            temp_group = temp_group.remove_channels(channels_to_remove)
                        else:
                            print("No channels to remove.")

                    params_band = params_pre["bandpass_filter"]
                    print("Bandpass filtering...")
                    temp_group = spre.bandpass_filter(
                        temp_group,
                        freq_min=params_band["freq_min"],
                        freq_max=params_band["freq_max"],
                    )

                    params_car = params_pre["car"]
                    print("Applying common median reference...")
                    temp_group = spre.common_reference(
                        temp_group,
                        reference=params_car["car_reference"],
                        operator=params_car["car_operator"],
                    )

                    if not params_pre["split_by_shank"]:
                        temporary_groups_list.append(temp_group)

            if not params_pre["split_by_shank"]:
                print("-" * 40)
                print("Regrouping shanks into wholeprobe...")
                group = si.aggregate_channels(
                    recording_list=list(temporary_groups_list)
                )

            w = si.plot_traces(
                group,
                time_range=(10, 10.5),
                mode="map",
                order_channel_by_depth=True,
                return_in_uV=True,
                backend="matplotlib",
            )

            save_widget(w, plots_pre / f"{recording_name}_traces_cleaned.png")
            print(
                f"Cleaned traces plot saved to: {plots_pre / f'{recording_name}_traces_cleaned.png'}"
            )
            print(
                f"Cleaning time {recording_name}: {preprocessing_time['Cleaning']:.2f} seconds\n"
            )

            if params_bad["debug_mode"]:
                print("DEBUG MODE. Preprocessing stopped.")
                return time_master

            # Motion
            group_pre_motion = group
            params_motion = params_pre["motion"]
            if params_motion["motion_overwrite"] and motion_folder.exists():
                shutil.rmtree(motion_folder)

            probe = recording.get_probe()
            pos = np.asarray(probe.contact_positions)

            if pos.shape[0] != recording.get_num_channels():
                raise ValueError("contact_positions vs n_channels mismatch")

            y = pos[:, 1].astype(float)
            dptp = float(np.ptp(y))

            y_u = np.sort(np.unique(y))
            dy = np.diff(y_u)
            dy = dy[dy > params_motion["min_pitch_difference_um"]]

            if len(dy) < 3:
                pitch_um = dptp / max(1, len(y_u) - 1)
            else:
                pitch_um = float(np.median(dy))

            win_scale_um = float(
                np.clip(
                    params_motion["n_rows_in_scale"] * pitch_um,
                    params_motion["min_win_scale_um"],
                    params_motion["max_win_scale_fraction"] * dptp,
                )
            )

            win_step_um = float(
                np.clip(
                    params_motion["step_over_scale"] * win_scale_um,
                    max(
                        params_motion["min_win_step_um"],
                        params_motion["min_win_step_pitch_fraction"] * pitch_um,
                    ),
                    params_motion["max_win_step_fraction"] * dptp,
                )
            )

            if win_step_um >= params_motion["max_step_scale_fraction"] * win_scale_um:
                win_step_um = float(
                    params_motion["fallback_step_over_scale"] * win_scale_um
                )

            estimate_motion_kwargs = {}
            estimate_motion_kwargs["win_step_um"] = win_step_um
            estimate_motion_kwargs["win_scale_um"] = win_scale_um

            print(f"Probe span: {dptp:.2f} µm")
            print(f"Electrode pitch: {pitch_um:.2f} µm")
            print(f"Motion window: {win_scale_um:.2f} µm")
            print(f"Motion window step: {win_step_um:.2f} µm")

            with timed(preprocessing_time, "Motion"):
                try:
                    print("Computing DREDge-fast motion...")
                    motion_output = spre.compute_motion(
                        group,
                        preset="dredge_fast",
                        folder=motion_folder,
                        raise_error=False,
                        output_motion_info=True,
                        estimate_motion_kwargs=estimate_motion_kwargs,
                        **params["job_kwargs"],
                    )

                    motion, motion_info = None, None
                    if isinstance(motion_output, tuple):
                        motion, motion_info = motion_output
                    elif motion_output is not None:
                        motion = motion_output

                    if motion is not None:
                        print("Applying motion correction...")
                        group = interpolate_motion(
                            group.astype("float32"), motion=motion
                        )
                    else:
                        print(
                            "Motion computation failed; saving uncorrected recording."
                        )

                    if motion_info is not None:
                        print("Saving motion plots...")
                        try:
                            motion_plot_path = (
                                motion_plots_folder
                                / f"{recording_name}_motion_correction.png"
                            )
                            probe_plot_path = (
                                motion_plots_folder
                                / f"{recording_name}_peak_locations.png"
                            )
                            peak_plot_path = (
                                motion_plots_folder
                                / f"{recording_name}_peak_activity_all.png"
                            )
                            locations = group.get_channel_locations()
                            depth_min = np.min(locations[:, 1])
                            depth_max = np.max(locations[:, 1]) * 1.1
                            fig = plt.figure(figsize=(14, 8))
                            si.plot_motion_info(
                                motion_info,
                                group,
                                figure=fig,
                                depth_lim=(depth_min, depth_max),
                                color_amplitude=True,
                                amplitude_cmap="inferno",
                                scatter_decimate=10,
                            )
                            fig.savefig(motion_plot_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"Motion plot saved to: {motion_plot_path}")

                            plot_peak_localization(
                                recording, motion_info, probe_plot_path
                            )
                            print(f"Peak locations plot saved to: {probe_plot_path}")

                            peaks = motion_info["peaks"]

                            w = si.plot_peak_activity(
                                group_pre_motion,
                                peaks,
                                bin_duration_s=None,
                                with_interpolated_map=True,
                                backend="matplotlib",
                            )
                            save_widget(w, peak_plot_path)
                            print(f"Peak Activity plot saved to:  {peak_plot_path}")

                        except Exception as e:
                            print(
                                f"Error saving motion plots for {recording_name}: {e}"
                            )
                except Exception as e:
                    print(f"Motion correction for {recording_name} failed: {e}")

            print(
                f"Motion time {recording_name}: {preprocessing_time['Motion']:.2f} seconds\n"
            )

            # Save
            with timed(preprocessing_time, "Save"):
                print("Saving preprocessed recording...")
                group.save(
                    folder=preprocessed_path, overwrite=True, **params["job_kwargs"]
                )
                if use_sync:
                    verify_sync_alignment(output_folder, preprocessed_path)

        finally:
            total = time.perf_counter() - group_start  # noqa: F841
            accounted = sum(  # noqa: F841
                v for k, v in preprocessing_time.items() if k != "Other"
            )

    return time_master


STREAM_ID = "imec0.ap"


def resolve_trim_frames(recording, params_trim):
    """Return (start_frame, end_frame) as integers; end is exclusive.
    Expects a recording whose times were already shifted to start at 0."""
    n = recording.get_num_samples()
    fs = recording.get_sampling_frequency()
    if not params_trim["trim_recording"]:
        return 0, n
    start = 0 if params_trim["start"] is None else round(params_trim["start"] * fs)
    end = n if params_trim["end"] is None else round(params_trim["end"] * fs)
    start, end = max(start, 0), min(end, n)
    if end <= start:
        raise ValueError(f"Invalid trim: frames {start}..{end} (recording has {n})")
    return start, end


def split_off_sync(full, recording_path_, stream_id=STREAM_ID):
    """Split an (already trimmed) recording loaded with the sync channel into
    (neural_recording, sync_recording)."""
    sync_id = full.channel_ids[-1]
    if "SY" not in str(sync_id).upper():
        raise RuntimeError(
            f"Last channel '{sync_id}' does not look like a sync channel."
        )

    neural = full.select_channels(full.channel_ids[:-1])
    sync = full.select_channels([sync_id])

    # The probe may not be attached when the sync channel is loaded. Re-attach it
    # from a lazy 384-channel read (same channel ids, no data is read).
    if not neural.has_probe():
        ref = si.read_spikeglx(
            recording_path_, stream_id=stream_id, load_sync_channel=False
        )
        if list(ref.channel_ids) != list(neural.channel_ids):
            raise RuntimeError(
                "Neural channel ids differ between sync and non-sync reads."
            )
        neural = neural.set_probe(ref.get_probe())
        print("Probe re-attached to neural recording.")
    return neural, sync


def save_sync_channel(
    sync, output_folder, start_frame, end_frame, n_samples_original, stream_id=STREAM_ID
):
    if sync.get_num_segments() != 1:
        raise RuntimeError("Sync saving assumes a single segment.")
    traces = sync.get_traces()[:, 0]  # raw int16 digital word, unscaled

    sync_dir = output_folder / "sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    np.save(sync_dir / f"{stream_id}_sync.npy", traces)
    (sync_dir / f"{stream_id}_sync.json").write_text(
        json.dumps(
            {
                "fs": float(sync.get_sampling_frequency()),
                "n_samples": int(traces.shape[0]),
                "start_frame_in_original": int(
                    start_frame
                ),  # original_sample = start_frame + i
                "end_frame_in_original": int(end_frame),  # exclusive
                "n_samples_original": int(n_samples_original),
                "channel_id": str(sync.channel_ids[0]),
            },
            indent=2,
        )
    )
    print(
        f"Sync channel saved to: {sync_dir} ({traces.shape[0]} samples, "
        f"original frames {start_frame}..{end_frame})"
    )


def verify_sync_alignment(output_folder, preprocessed_path, stream_id=STREAM_ID):
    meta = json.loads((output_folder / "sync" / f"{stream_id}_sync.json").read_text())
    saved = si.load(preprocessed_path)
    if saved.get_num_samples() != meta["n_samples"]:
        raise RuntimeError(
            f"Sync misaligned with {preprocessed_path.name}: preprocessed has "
            f"{saved.get_num_samples()} samples, sync has {meta['n_samples']}."
        )
    print(
        f"Sync alignment verified for {preprocessed_path.name}: "
        f"{meta['n_samples']} samples."
    )


COLORS = {"good": "tab:green", "dead": "red", "noise": "orange", "out": "blue"}


def _chunks(rec, n, dur_s, seed=0):
    """Yield n random chunks in uV, each (n_channels, n_samples). A generator, so memory stays flat."""
    m = int(dur_s * rec.get_sampling_frequency())
    starts = np.sort(
        np.random.default_rng(seed).integers(0, rec.get_num_samples() - m, n)
    )
    g, o = rec.get_channel_gains(), rec.get_channel_offsets()
    if g is None or o is None:
        g, o = 1, 0
    for s in starts:
        yield (
            rec.get_traces(start_frame=int(s), end_frame=int(s) + m).astype("float32")
            * g
            + o
        ).T


def _features(chunks, fs, order, p):
    """
    Median features across chunks, in recording channel order.
    Calculates standard IBL features alongside SpikeInterface MAD metrics.
    """
    sos = butter(3, 300, btype="highpass", fs=fs, output="sos")
    ibl_f = {k: [] for k in ("xcor_hf", "xcor_lf", "psd_hf")}
    std_hf, raw_mean, raw_var = [], [], []
    si_mads = []

    for c in chunks:
        _, f = ibl(
            c[order] / 1e6,
            fs,
            similarity_threshold=tuple(p["similarity_threshold"]),
            psd_hf_threshold=p["psd_hf_threshold"],
        )
        for key, values in ibl_f.items():
            values.append(f[key])

        filtered = sosfiltfilt(sos, c, axis=1)
        std_hf.append(filtered.std(axis=1))
        raw_mean.append(c.mean(axis=1))
        raw_var.append(c.var(axis=1))

        # Calculate Median Absolute Deviation (MAD) for each channel on high-pass filtered data
        med = np.median(filtered, axis=1, keepdims=True)
        si_mads.append(np.median(np.abs(filtered - med), axis=1) * 1.4826)

    feats = {k: np.median(v, 0)[np.argsort(order)] for k, v in ibl_f.items()}
    feats["std_hf"] = np.median(std_hf, 0)
    feats["std_raw"] = np.sqrt(np.mean(raw_var, 0) + np.var(raw_mean, 0))
    feats["si_mad"] = np.median(si_mads, 0)
    return feats


def _outliers(x, k):
    """True where x is more than k MAD-scaled deviations above the median across channels."""
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return x > med + k * mad


def _label(feats, order, p):
    """
    IBL decision rules applied to features.
    MAD noise detection is restricted strictly to non-raw (highpass filtered) data thresholds.
    """
    dead_t, noise_t = p["similarity_threshold"]
    hf, psd = feats["xcor_hf"], feats["psd_hf"]
    lf_sorted = feats["xcor_lf"][order]

    labels = np.full(len(hf), "good", dtype="U5")
    labels[hf < dead_t] = "dead"
    k_mad = p["std_mad_threshold"]

    # Noise restricted only to high-passed / non-raw thresholds
    noisy = (
        (hf > noise_t)
        | (psd > p["psd_hf_threshold"])
        | _outliers(feats["std_hf"], k_mad)
    )
    labels[noisy] = "noise"

    k = len(lf_sorted)
    while k > 0 and lf_sorted[k - 1] < p["outside_threshold"]:
        k -= 1
    labels[order[k:]] = "out"
    return labels


def detect_bad_channels_ibl(rec, p, n_chunks=100, chunk_s=0.3, seed=0):
    """Performs IBL bad channel detection & SpikeInterface MAD noise detection."""
    locs = rec.get_channel_locations()
    order = np.lexsort((locs[:, 0], locs[:, 1]))
    feats = _features(
        _chunks(rec, n_chunks, chunk_s, seed), rec.get_sampling_frequency(), order, p
    )
    labels = _label(feats, order, p)

    try:
        si_bad_ids, _ = spre.detect_bad_channels(rec, method="mad")
    except Exception:
        si_bad_ids = np.array([])

    return np.asarray(rec.channel_ids)[labels != "good"], labels, feats, si_bad_ids


def _plot_traces(chunk, fs, labels, order, path, n, ctx=2, ms=30):
    rank = np.argsort(order)
    rows = [
        (l, i) for l in ("dead", "noise", "out") for i in np.where(labels == l)[0][:n]
    ]
    if not rows:
        return
    k = int(ms * fs / 1000)
    t = np.arange(k) / fs * 1000
    step = 2 * np.percentile(np.abs(chunk[:, :k]), 99)
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.2 * len(rows)), squeeze=False)
    for ax, (l, i) in zip(axes[:, 0], rows):
        for j, ch in enumerate(order[max(rank[i] - ctx, 0) : rank[i] + ctx + 1]):
            hit = ch == i
            ax.plot(
                t,
                chunk[ch, :k] + j * step,
                c=COLORS[l] if hit else "gray",
                lw=1.2 if hit else 0.7,
            )
        ax.set_title(
            f"{l}: channel index {i} (colored) with {ctx} depth neighbors each side",
            fontsize=8,
        )
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def bad_channels_diagnostic(
    rec,
    labels,
    feats,
    out_dir,
    p,
    name="group",
    n_chunks=30,
    chunk_s=0.3,
    seeds=(1, 2, 3, 4),
    n_examples=3,
    custom_star_channels=None,
):
    """
    Generates diagnostics, incorporating SpikeInterface MAD plots.
    Annotates channels exceeding thresholds and custom-selected channels with their numeric ID.
    """
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    fs = rec.get_sampling_frequency()
    locs = rec.get_channel_locations()
    depth = locs[:, 1]
    order = np.lexsort((locs[:, 0], depth))
    labels = np.asarray(labels).astype(str)

    print("Performing bad channel diagnostics...")
    runs = [labels] + [
        detect_bad_channels_ibl(rec, p, seed=s)[1]
        for s in tqdm(seeds, desc="stability")
    ]
    frac = (np.array(runs) != "good").mean(0)

    dead_t, noise_t = p["similarity_threshold"]
    out_t = p["outside_threshold"]
    variants = []
    for d in (-0.2, -0.1, 0.1, 0.2):
        variants += [
            (
                "similarity_threshold[0]",
                dead_t + d,
                {**p, "similarity_threshold": [dead_t + d, noise_t]},
            ),
            (
                "similarity_threshold[1]",
                noise_t + d,
                {**p, "similarity_threshold": [dead_t, noise_t + d]},
            ),
            ("outside_threshold", out_t + d, {**p, "outside_threshold": out_t + d}),
        ]
    k_mad = p["std_mad_threshold"]
    variants += [
        ("std_mad_threshold", k_mad + d, {**p, "std_mad_threshold": k_mad + d})
        for d in (-2, -1, 1, 2)
    ]
    sweep = []
    for param, value, pv in variants:
        lab = _label(feats, order, pv)
        counts = pd.Series(lab).value_counts().reindex(list(COLORS), fill_value=0)
        sweep.append(
            {
                "param": param,
                "value": round(value, 3),
                **counts.to_dict(),
                "n_changed_vs_pipeline": int((lab != labels).sum()),
            }
        )
    pd.DataFrame(sweep).sort_values(["param", "value"]).to_csv(
        out / "threshold_sweep.csv", index=False
    )

    report = pd.DataFrame(
        {
            "channel_id": rec.channel_ids,
            "x_um": locs[:, 0],
            "depth_um": depth,
            "label": labels,
            **feats,
            "frac_runs_bad": frac,
        }
    )
    report.to_csv(out / "channel_report.csv", index=False)

    print(f"Saving bad channel diagnostic figures to {out}...")
    fig = plt.figure(figsize=(26, 11))
    gs = fig.add_gridspec(2, 6)
    amp_limit = lambda x: np.median(x) + k_mad * 1.4826 * np.median(
        np.abs(x - np.median(x))
    )

    # Feature key to dictionary lookup mapping
    feat_map = {
        "High-Freq Coherence": "xcor_hf",
        "Low-Freq Coherence": "xcor_lf",
        "High-Freq PSD": "psd_hf",
        "High-Passed STD.": "std_hf",
        "High-Passed MAD": "si_mad",
        "Raw STD": "std_raw",
    }

    panels = [
        ("High-Freq Coherence", list(p["similarity_threshold"])),
        ("Low-Freq Coherence", [p["outside_threshold"]]),
        ("High-Freq PSD", [p["psd_hf_threshold"]]),
        ("High-Passed STD.", [amp_limit(feats["std_hf"])]),
        ("High-Passed MAD", [amp_limit(feats["si_mad"])]),
        ("Raw STD", [amp_limit(feats["std_raw"])]),
    ]

    channel_ids = rec.channel_ids
    star_mask = (
        np.isin(channel_ids, custom_star_channels)
        if custom_star_channels is not None
        else np.zeros(len(channel_ids), dtype=bool)
    )

    # Clean channel IDs to display numeric values (e.g. 'imec0.ap#AP246' -> '246')
    clean_num_ids = []
    for cid in channel_ids:
        match = re.search(r"\d+", str(cid)[::-1])
        clean_num_ids.append(match.group(0)[::-1] if match else str(cid))
    clean_num_ids = np.asarray(clean_num_ids)

    for j, (title, lines) in enumerate(panels):
        key = feat_map[title]
        ax = fig.add_subplot(gs[0, j])
        val_arr = feats[key]

        # Determine channels beyond threshold for this panel
        if key == "xcor_hf":
            beyond_mask = (val_arr < lines[0]) | (val_arr > lines[1])
        elif key == "xcor_lf":
            beyond_mask = val_arr < lines[0]
        else:
            beyond_mask = val_arr > lines[0]

        # Combine channels to annotate (beyond threshold OR custom star)
        annotate_mask = beyond_mask | star_mask

        # Plot standard channels (circles) vs custom channels (stars)
        for lab, c in COLORS.items():
            m = (labels == lab) & (~star_mask)
            ax.scatter(
                val_arr[m], depth[m], s=12, c=c, marker="o", label=f"{lab} ({m.sum()})"
            )

            m_star = (labels == lab) & star_mask
            if m_star.any():
                ax.scatter(
                    val_arr[m_star],
                    depth[m_star],
                    s=120,
                    c=c,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )

        # Text Annotations for flagged/custom channels
        x_span = np.ptp(val_arr) if np.ptp(val_arr) > 0 else 1.0
        x_offset = x_span * 0.015

        for idx in np.where(annotate_mask)[0]:
            ax.text(
                val_arr[idx] + x_offset,
                depth[idx],
                clean_num_ids[idx],
                fontsize=6,
                va="center",
                ha="left",
                alpha=0.85,
                fontweight="bold" if star_mask[idx] else "normal",
            )

        for v in lines:
            ax.axvline(v, ls="--", c="k", lw=0.8)
        ax.set_title(title)
        if j == 0:
            ax.set_ylabel("depth (um)")
            ax.legend(fontsize=8, loc="upper left")

        if j == 1 and star_mask.any():
            shape_legend = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Detected",
                    markerfacecolor="gray",
                    markersize=6,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    label="Custom List",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=10,
                ),
            ]
            ax.legend(
                handles=shape_legend,
                title="Marker Shape",
                fontsize=8,
                loc="upper right",
            )

    chunks = list(_chunks(rec, n_chunks, chunk_s))
    f, _ = welch(chunks[0], fs=fs, nperseg=1024, axis=1)
    psd = np.mean([welch(c, fs=fs, nperseg=1024, axis=1)[1] for c in chunks], 0)
    ax = fig.add_subplot(gs[1, :3])
    g = labels == "good"
    if g.any():
        lo, med, hi = np.percentile(psd[g], [10, 50, 90], axis=0)
        ax.fill_between(f, lo, hi, color=COLORS["good"], alpha=0.2)
        ax.plot(f, med, c=COLORS["good"], lw=2, label="good median (10-90%)")
    for lab in ("dead", "noise", "out"):
        for j, i in enumerate(np.where(labels == lab)[0][:n_examples]):
            ax.plot(f, psd[i], c=COLORS[lab], lw=1, label=lab if j == 0 else None)
    ax.axvline(0.8 * fs / 2, ls="--", c="k", lw=0.8)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Hz",
        ylabel="PSD (uV^2/Hz)",
        title="PSD: flagged vs good",
    )
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 3:])
    sc = ax.scatter(
        frac[~star_mask], depth[~star_mask], c=frac[~star_mask], s=8, vmin=0, vmax=1
    )
    if star_mask.any():
        ax.scatter(
            frac[star_mask],
            depth[star_mask],
            c=frac[star_mask],
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            vmin=0,
            vmax=1,
            zorder=5,
        )
    ax.set(
        xlabel="fraction of runs flagged", ylabel="depth (um)", title="Label stability"
    )
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)

    _plot_traces(chunks[0], fs, labels, order, out / "traces.png", n_examples)

    ids = np.asarray(rec.channel_ids)
    print(f"Channel labels: {pd.Series(labels).value_counts().to_dict()}")
    for lab in sorted(set(labels) - {"good"}):
        flagged = ids[labels == lab]
        print(f"  {lab} ({len(flagged)}): {list(flagged)}")
    if (labels == "good").all():
        print("  no non-good channels")
    unstable = ids[(frac > 0) & (frac < 1)]
    print(f"Unstable channels ({len(unstable)}): {list(unstable)}")
    return report
