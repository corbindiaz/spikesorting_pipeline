
import shutil
import time
import warnings
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spikeinterface.full as si
import spikeinterface.sorters as ss
import torch
from kilosort.run_kilosort import close_logger
from sklearn.preprocessing import StandardScaler
 
from utils import timed
 
NON_KILOSORT_PARAM_KEYS = ("generate_summary_plots", "generate_phy_unit_plots", "keep_output")
 
MIN_SPIKES_FOR_UNIT_PLOTS = 10
 
N_WAVEFORMS = 500
PRE_SAMPLES = 30
POST_SAMPLES = 50
 
# Mean-waveform overlay (built from the analyzer after it is created)
OVERLAY_MS_BEFORE = 1.0
OVERLAY_MS_AFTER = 2.0
OVERLAY_SEED = 0
# The "waveforms" extension stores every sampled waveform inside the analyzer
# zarr. The means are all the plot needs, so drop it afterwards by default.
KEEP_WAVEFORMS_IN_ANALYZER = False
 
LABEL_COLORS = {"good": "tab:green", "mua": "tab:orange"}
 
 
def sorter_kilosort(recording_path_, output_folder, params, time_master, step=1):
    output_folder = Path(output_folder)
    # Use the provided path if given, otherwise fall back to <output>/preprocessed
    preprocessed_folder = (
        Path(recording_path_) if recording_path_ else output_folder / "preprocessed"
    )
    kilosort_folder = output_folder / "kilosort"
    analyzer_folder = output_folder / "analyzer"
    plots = output_folder / "plots"
    motion_plots_folder = plots / "motion"
    summary_plots_folder = plots / "kilosort"
    phy_plots_folder = summary_plots_folder / "units"
 
    kilosort_folder.mkdir(parents=True, exist_ok=True)
    analyzer_folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    motion_plots_folder.mkdir(parents=True, exist_ok=True)
 
    kilosort_params = params["kilosort"]
 
    kilosort_params["n_jobs"] = params["job_kwargs"]["n_jobs"]
    kilosort_params["progress_bar"] = params["job_kwargs"]["progress_bar"]
    kilosort_params["chunk_duration"] = params["job_kwargs"]["chunk_duration"]
 
    generate_summary_plots = bool(kilosort_params.get("generate_summary_plots", False))
    generate_phy_unit_plots = bool(
        kilosort_params.get("generate_phy_unit_plots", False)
    )
    # Everything except the plotting flags goes to the sorter
    sorter_params = {
        k: v for k, v in kilosort_params.items() if k not in NON_KILOSORT_PARAM_KEYS
    }
 
    preprocessed_recording_paths = sorted(
        p
        for p in preprocessed_folder.iterdir()
        if p.is_dir() and "_group" in p.name and p.name != "motion"
    )
 
    for i, recording_path in enumerate(preprocessed_recording_paths):
        group_name = recording_path.stem
 
        # Register timing entries first so partial results survive a crash
        group_time = time_master.setdefault(f"Group{i}", {})
        kilosort_time = group_time.setdefault("Kilosort", {})
        group_start = time.perf_counter()
 
        print("-" * 80)
 
        kilosort_group_folder = kilosort_folder / group_name
        kilosort_group_folder.mkdir(parents=True, exist_ok=True)
 
        analyzer_group_folder = analyzer_folder / f"{group_name}.zarr"
        motion_group_folder = kilosort_group_folder / "motion"
        sorter_output_folder = kilosort_group_folder / "sorter_output"
 
        try:
            # Load
            with timed(kilosort_time, "Load"):
                print(f"Importing: {recording_path}")
                recording = si.load(recording_path)
                if step == 0:
                    print(recording)
 
            # Sorting
            with timed(kilosort_time, "Sorting"):
                print()
                print("Running Kilosort4...")
                print("CUDA available:", torch.cuda.is_available())
                print("CUDA version:", torch.version.cuda)
 
                if torch.cuda.is_available():
                    print("GPU:", torch.cuda.get_device_name(0))
                sorting = ss.run_sorter(
                    "kilosort4",
                    recording,
                    folder=kilosort_group_folder,
                    verbose=False,
                    delete_output_folder=False,
                    remove_existing_folder=True,
                    **sorter_params,
                )
 
                n_original_units = len(sorting.unit_ids)
                print(f"Kilosort4 found {n_original_units} units.")
 
                sorting = _fix_object_properties(sorting)
 
            # Read Kilosort motion
            motion = None
            if kilosort_params["do_correction"]:
                with timed(kilosort_time, "Motion"):
                    motion = read_kilosort4_motion(
                        sorter_output_folder, recording=recording
                    )
 
            # Summary plots (must run BEFORE the sorter output folder is deleted)
            if generate_summary_plots or generate_phy_unit_plots:
                ks_data = _load_ks_output(sorter_output_folder)
 
            if generate_summary_plots:
                with timed(kilosort_time, "Summary Plots"):
                    try:
                        print("Generating Kilosort summary plots...")
                        save_summary_plots(
                            sorting,
                            ks_data,
                            recording.sampling_frequency,
                            sorter_output_folder,
                            summary_plots_folder,
                            group_name,
                        )
                    except Exception as e:
                        print(f"Error generating summary plots for {group_name}: {e}")
 
            # Per-unit phy + waveform plots (also needs the sorter output folder)
            if generate_phy_unit_plots:
                with timed(kilosort_time, "Phy Unit Plots"):
                    try:
                        print("Generating per-unit phy and waveform plots...")
                        save_unit_plots(
                            sorting,
                            ks_data,
                            recording,
                            sorter_output_folder,
                            phy_plots_folder / group_name,
                        )
                    except Exception as e:
                        print(f"Error generating unit plots for {group_name}: {e}")
 
            # Close Kilosort's log file before deleting sorter output
            close_logger()

            if not kilosort_params['keep_output']:
                if sorter_output_folder.exists():
                    try:
                        shutil.rmtree(sorter_output_folder)
                        print("Deleted Kilosort sorter output folder.")
                    except Exception as e:
                        print(f"Error deleting sorter output folder: {e}")
 
            # Analyzer
            with timed(kilosort_time, "Analyzer"):
                print(f"Creating SortingAnalyzer: {analyzer_group_folder}")
                analyzer = si.create_sorting_analyzer(
                    sorting=sorting,
                    recording=recording,
                    format="zarr",
                    folder=analyzer_group_folder,
                    overwrite=True,
                    **params["job_kwargs"],
                )
 
            # Mean-waveform overlay. Needs the analyzer (random spikes +
            # waveforms from the recording); labels come from the KSLabel
            # property, so the deleted sorter output is not required.
            if generate_summary_plots:
                with timed(kilosort_time, "Waveform Summary Plots"):
                    try:
                        print("Generating mean waveform overlay plot...")
                        save_waveform_overlay_plot(
                            analyzer,
                            sorter_output_folder,
                            summary_plots_folder,
                            group_name,
                            job_kwargs=params["job_kwargs"],
                        )
                    except Exception as e:
                        print(
                            f"Error generating mean waveform overlay for "
                            f"{group_name}: {e}"
                        )
 
            # Save motion + plot
            if motion is not None:
                with timed(kilosort_time, "Motion"):
                    if motion_group_folder.exists():
                        try:
                            shutil.rmtree(motion_group_folder)
                            motion.save(folder=motion_group_folder)
                        except Exception as e:
                            print(
                                f"New kilosort motion was not saved due to "
                                f"error deleting old motion folder: {e}"
                            )
                    else:
                        motion.save(folder=motion_group_folder)
 
                    print("Saving motion plot...")
                    try:
                        motion_plot_path = (
                            motion_plots_folder / f"{group_name}_kilosort_motion.png"
                        )
                        fig = plt.figure(figsize=(14, 8))
                        si.plot_motion(motion, mode="map", figure=fig)
                        fig.savefig(motion_plot_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        print(f"Motion plot saved to: {motion_plot_path}")
                    except Exception as e:
                        print(f"Error saving motion plot for {group_name}: {e}")
 
        finally:
            total = time.perf_counter() - group_start
            accounted = sum(v for k, v in kilosort_time.items() if k != "Other")
            kilosort_time["Other"] = max(0.0, total - accounted)
            print(f"Total time Kilosort {group_name}: {total:.2f} seconds")
 
    return time_master
 
 
# ======================================================================
# Kilosort output loading / labels
# ======================================================================
def _load_ks_output(sorter_output_folder):
    """Load the Kilosort/phy files needed for plotting.
 
    pc_features.npy / pc_feature_ind.npy are only written by Kilosort4 when
    save_extra_vars=True; they are returned as None if missing.
    """
    folder = Path(sorter_output_folder)
 
    def _load(name, mmap_mode=None):
        path = folder / name
        return np.load(path, mmap_mode=mmap_mode) if path.is_file() else None
 
    data = {
        "spike_times": _load("spike_times.npy"),
        "spike_clusters": _load("spike_clusters.npy"),
        "spike_templates": _load("spike_templates.npy"),
        "templates": _load("templates.npy"),
        "pc_features": _load("pc_features.npy", mmap_mode="r"),
        "pc_feature_ind": _load("pc_feature_ind.npy"),
    }
    for key in ("spike_times", "spike_clusters", "spike_templates"):
        if data[key] is not None:
            data[key] = data[key].reshape(-1)
    return data
 
 
def _get_ks_labels(sorting, sorter_output_folder):
    """Return {unit_id: 'good'/'mua'} using the KSLabel property, falling back
    to cluster_KSLabel.tsv in the sorter output."""
    if "KSLabel" in sorting.get_property_keys():
        values = sorting.get_property("KSLabel")
        return {uid: str(v).strip().lower() for uid, v in zip(sorting.unit_ids, values)}
 
    tsv = Path(sorter_output_folder) / "cluster_KSLabel.tsv"
    if tsv.is_file():
        df = pd.read_csv(tsv, sep="\t")
        mapping = dict(zip(df["cluster_id"], df["KSLabel"].astype(str).str.lower()))
        return {uid: mapping.get(int(uid), "unknown") for uid in sorting.unit_ids}
 
    warnings.warn("No Kilosort labels found; labelling all units 'unknown'.")
    return {uid: "unknown" for uid in sorting.unit_ids}
 
 
def _main_template(ks_data, unit):
    """Most common template id among the unit's spikes."""
    templates_of_unit = ks_data["spike_templates"][ks_data["spike_clusters"] == unit]
    return int(np.bincount(templates_of_unit).argmax())
 
 
# ======================================================================
# Summary plots: good/mua bar graph + overlaid templates
# ======================================================================
def save_summary_plots(
    sorting, ks_data, fs, sorter_output_folder, out_folder, group_name
):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
 
    labels = _get_ks_labels(sorting, sorter_output_folder)
 
    # ---- bar graph of unit labels ----
    label_names = sorted(set(labels.values()), key=lambda s: (s != "good", s != "mua", s))
    counts = [sum(1 for v in labels.values() if v == name) for name in label_names]
 
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(
        label_names,
        counts,
        color=[LABEL_COLORS.get(n, "tab:gray") for n in label_names],
    )
    ax.bar_label(bars)
    ax.set_ylabel("Number of units")
    ax.set_title(f"Kilosort unit labels ({group_name})")
    fig.tight_layout()
    path = out_folder / f"{group_name}_unit_label_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Unit label bar graph saved to: {path}")

    # ---- overlaid templates (good vs mua) ----
    templates = ks_data["templates"]
    if (
        templates is None
        or ks_data["spike_clusters"] is None
        or ks_data["spike_templates"] is None
    ):
        print("templates.npy / spike files not found; skipping template overlay plot.")
        return

    n_t = templates.shape[1]
    t_ms = (np.arange(n_t) - n_t // 2) / fs * 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ("good", "mua")):
        unit_list = [u for u, lab in labels.items() if lab == name]
        for unit in unit_list:
            if not np.any(ks_data["spike_clusters"] == unit):
                continue
            template = templates[_main_template(ks_data, unit)]  # (n_t, n_chan)
            peak_ch = int(np.abs(template).max(axis=0).argmax())
            trace = template[:, peak_ch]
 
            # Scale by a positive value so peak-to-peak amplitude (max - min) == 1
            p2p = trace.max() - trace.min()
            if p2p <= 0:
                continue  # flat/degenerate template; nothing to scale
            trace = trace / p2p
 
            ax.plot(t_ms, trace, color="grey", alpha=0.15, lw=1)
        ax.set_title(f"{name} templates (n={len(unit_list)})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("Normalized template amplitude (peak-to-peak = 1)")
    fig.suptitle(f"Kilosort templates ({group_name})")
    fig.tight_layout()
    path = out_folder / f"{group_name}_template_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Template overlay saved to: {path}")
 
 
# ======================================================================
# Summary plot: overlaid mean waveforms (from the SortingAnalyzer)
# ======================================================================
def _get_mean_waveforms(analyzer, job_kwargs=None):
    """Return ({unit_id: mean waveform (n_samples, n_channels)}, nbefore).
 
    Samples up to N_WAVEFORMS spikes per unit (all spikes if a unit has fewer),
    extracts their waveforms from the analyzer's recording and averages them.
    Waveforms are sparse (channels near each unit), so the channel axis can
    differ between units; that is fine since only the peak channel is plotted.
    """
    job_kwargs = dict(job_kwargs or {})
 
    analyzer.compute(
        "random_spikes",
        method="uniform",
        max_spikes_per_unit=N_WAVEFORMS,
        seed=OVERLAY_SEED,
    )
    analyzer.compute(
        "waveforms",
        ms_before=OVERLAY_MS_BEFORE,
        ms_after=OVERLAY_MS_AFTER,
        **job_kwargs,
    )
 
    ext = analyzer.get_extension("waveforms")
    nbefore = int(ext.params["ms_before"] * analyzer.sampling_frequency / 1000.0)
 
    mean_waveforms = {}
    for unit in analyzer.unit_ids:
        wfs = ext.get_waveforms_one_unit(unit)  # (n_spikes, n_samples, n_channels)
        if wfs.shape[0] == 0:
            continue
        mean_waveforms[unit] = wfs.mean(axis=0)
    return mean_waveforms, nbefore
 
 
def save_waveform_overlay_plot(
    analyzer, sorter_output_folder, out_folder, group_name, job_kwargs=None
):
    """Overlay of per-unit mean waveforms (good vs mua), each scaled by a
    positive value so its peak-to-peak amplitude is 1, y-axis fixed to -1..1.
 
    The label bar graph is already made by save_summary_plots, so only the
    overlay is produced here.
    """
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
 
    fs = analyzer.sampling_frequency
    labels = _get_ks_labels(analyzer.sorting, sorter_output_folder)
 
    mean_waveforms, nbefore = _get_mean_waveforms(analyzer, job_kwargs)
    if not KEEP_WAVEFORMS_IN_ANALYZER:
        # Means are already in memory; keep the analyzer zarr small.
        analyzer.delete_extension("waveforms")
 
    if not mean_waveforms:
        print("No waveforms extracted; skipping mean waveform overlay plot.")
        return
 
    n_t = next(iter(mean_waveforms.values())).shape[0]
    t_ms = (np.arange(n_t) - nbefore) / fs * 1000  # 0 ms = spike time
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ("good", "mua")):
        unit_list = [u for u, lab in labels.items() if lab == name]
        for unit in unit_list:
            if unit not in mean_waveforms:
                continue  # unit had no spikes
            mean_wf = mean_waveforms[unit]  # (n_t, n_chan)
            peak_ch = int(np.abs(mean_wf).max(axis=0).argmax())
            trace = mean_wf[:, peak_ch]
 
            # Scale by a positive value so peak-to-peak amplitude (max - min) == 1
            p2p = trace.max() - trace.min()
            if p2p <= 0:
                continue  # flat/degenerate waveform; nothing to scale
            trace = trace / p2p
 
            ax.plot(t_ms, trace, color="grey", alpha=0.15, lw=1)
        ax.set_title(f"{name} mean waveforms (n={len(unit_list)})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("Normalized mean waveform (peak-to-peak = 1)")
    fig.suptitle(
        f"Kilosort mean waveforms, up to {N_WAVEFORMS} spikes/unit ({group_name})"
    )
    fig.tight_layout()
    path = out_folder / f"{group_name}_mean_waveform_overlay.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Mean waveform overlay saved to: {path}")
 
 
# ======================================================================
# Per-unit phy-style PC plots and waveform plots
# ======================================================================
def save_unit_plots(sorting, ks_data, recording, sorter_output_folder, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
 
    needed = ("spike_times", "spike_clusters", "spike_templates")
    if any(ks_data[k] is None for k in needed):
        print("Kilosort spike files not found; skipping unit plots.")
        return
 
    have_pcs = ks_data["pc_features"] is not None
    have_pc_ind = ks_data["pc_feature_ind"] is not None
    if not (have_pcs and have_pc_ind):
        print(
            "pc_features.npy / pc_feature_ind.npy not found (set "
            "save_extra_vars=True for Kilosort4); skipping unit plots."
        )
        return
 
    labels = _get_ks_labels(sorting, sorter_output_folder)
    fs = recording.sampling_frequency
    rng = np.random.default_rng(0)
 
    spike_times = ks_data["spike_times"]
    spike_clusters = ks_data["spike_clusters"]
 
    for unit in sorting.unit_ids:
        label = labels.get(unit, "unknown")
        unit_int = int(unit)
        spikes_in_cluster = np.where(spike_clusters == unit)[0]
        if len(spikes_in_cluster) < MIN_SPIKES_FOR_UNIT_PLOTS:
            print(f"Unit {unit}: only {len(spikes_in_cluster)} spikes, skipping")
            continue
 
        # PCs, standardised with spike time appended as the last column
        pcs_unit = np.asarray(
            ks_data["pc_features"][spikes_in_cluster][:, :2, :2]
        )
        pc_dims = pcs_unit.reshape(len(spikes_in_cluster), -1)
        pc_dims = np.column_stack([pc_dims, spike_times[spikes_in_cluster]])
        pc_unit = StandardScaler().fit_transform(pc_dims)
 
        phy_path = out_folder / f"{label}_unit_{unit_int:03d}_phy_plot.png"
        phy_plot(pc_unit, unit_int, label, phy_path)
 
        main_template = _main_template(ks_data, unit)
        wf_channels = ks_data["pc_feature_ind"][main_template, :2]
        wf_path = out_folder / f"{label}_unit_{unit_int:03d}_waveform_plot.png"
        try:
            plot_waveforms(
                recording,
                fs,
                unit_int,
                label,
                spike_times[spikes_in_cluster],
                wf_channels,
                rng,
                wf_path,
            )
        except Exception as e:
            print(f"Error plotting waveforms for unit {unit}: {e}")
 
 
def phy_plot(pc_unit, unit, label, save_path):
    """4x4 phy-style grid: diagonal = feature vs time, off-diagonal = feature pairs."""
    indices = [0, 1, 2, 2 + 1]  # PC0/ch0, PC0/ch1, PC1/ch0, PC1/ch1
 
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    for i, ai in enumerate(indices):
        for j, aj in enumerate(indices):
            if i == j:
                ax[i, i].scatter(pc_unit[:, -1], pc_unit[:, ai], s=1, c="k")
            else:
                ax[i, j].scatter(pc_unit[:, ai], pc_unit[:, aj], s=1, c="k")
 
    fig.suptitle(f"Phy Plot for unit {unit} ({label})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
 
 
def plot_waveforms(
    recording, fs, unit, label, spike_samples, channels, rng, save_path
):
    """Individual raw waveforms (faint) plus their mean for each of the unit's
    main channels."""
    channel_ids = recording.channel_ids[np.asarray(channels, dtype=int)]
    n_ch = len(channel_ids)
    n_frames = recording.get_num_samples()
    t_ms = np.arange(-PRE_SAMPLES, POST_SAMPLES) / fs * 1000
 
    n_pick = min(len(spike_samples), N_WAVEFORMS)
    picked = rng.choice(len(spike_samples), size=n_pick, replace=False)
 
    waveforms = []
    for s in spike_samples[picked]:
        start, stop = int(s) - PRE_SAMPLES, int(s) + POST_SAMPLES
        if start < 0 or stop > n_frames:
            continue
        waveforms.append(
            np.asarray(
                recording.get_traces(
                    start_frame=start, end_frame=stop, channel_ids=channel_ids
                ),
                dtype=float,
            )
        )
    if not waveforms:
        print(f"Unit {unit}: no valid waveforms to plot")
        return
    waveforms = np.stack(waveforms)  # (n_spikes, n_samples, n_ch)
 
    color = LABEL_COLORS.get(label, "tab:blue")
    fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 5), squeeze=False, sharey=True)
    axes = axes[0]
    for c in range(n_ch):
        axes[c].plot(t_ms, waveforms[:, :, c].T, color="grey", alpha=0.05, lw=0.5)
        axes[c].plot(t_ms, waveforms[:, :, c].mean(axis=0), color=color, lw=2)
        axes[c].set_title(f"Channel {channel_ids[c]}")
        axes[c].set_xlabel("Time from spike (ms)")
    axes[0].set_ylabel("Amplitude")
    fig.suptitle(f"Waveforms for unit {unit} ({label}, n={len(waveforms)})")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
 
 
# ======================================================================
# Existing helpers
# ======================================================================
def read_kilosort4_motion(
    sorter_output_folder: str | Path, recording: si.BaseRecording | None = None
) -> si.Motion:
    """Read motion information from a Kilosort4 output folder."""
 
    sorter_output_folder = Path(sorter_output_folder)
    ops_file = sorter_output_folder / "ops.npy"
 
    if not ops_file.is_file():
        raise FileNotFoundError("'ops.npy' file not found!")
 
    ops = np.load(ops_file, allow_pickle=True).item()
 
    yblk = ops.get("yblk")
    dshift = ops.get("dshift")
 
    if yblk is None or dshift is None:
        warnings.warn("'yblk' and 'dshift' fields not found in ops file!")
        return None
 
    displacement = dshift
    spatial_bins_um = yblk
 
    batch_size = ops["batch_size"]
    fs = ops["fs"]
    t_bin = batch_size / fs
 
    if recording is not None:
        t_start = recording.get_start_time()
        t_end = recording.get_end_time()
 
        temporal_bins_s = np.linspace(
            t_start + t_bin / 2, t_end - t_bin / 2, displacement.shape[0]
        )
    else:
        temporal_bins_s = np.arange(displacement.shape[0]) * t_bin + t_bin / 2
 
    motion = si.Motion(
        displacement=displacement,
        temporal_bins_s=temporal_bins_s,
        spatial_bins_um=spatial_bins_um,
    )
 
    return motion
 
 
def _fix_object_properties(sorting):
    """Cast object-dtype unit properties (e.g. KSLabel) to fixed-width
    strings so they survive being saved into a zarr SortingAnalyzer."""
    for key in sorting.get_property_keys():
        values = sorting.get_property(key)
        if values.dtype == object:
            sorting.set_property(key, values.astype(str))
    return sorting
 