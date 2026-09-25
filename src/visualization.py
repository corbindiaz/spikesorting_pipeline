from IPython.utils.io import capture_output
import warnings
import re
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import numpy as np

from sklearn.decomposition import PCA

from probeinterface.plotting import plot_probe

import bombcell as bc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib import cm
from matplotlib.patches import Patch

def save_probe_figure(rec, out_dir, basename="probe_layout"):
    probe = rec.get_probe()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = plot_probe(probe)

    # Handle different return types across SpikeInterface versions
    if isinstance(out, tuple):
        if hasattr(out[0], "savefig"):   # (fig, ax)
            fig = out[0]
        else:                            # (ax1, ax2, ...)
            fig = out[0].figure
    else:
        fig = out.figure

    fig.tight_layout()

    fig.savefig(out_dir / f"{basename}.png", dpi=300)
    #fig.savefig(out_dir / f"{basename}.svg")
    plt.close(fig)

def save_text_summary_image(analyzer, out_dir: Path, basename: str = "summary"):
    """Save a quick, robust summary as an image (works even if no extensions exist)."""
    lines = []
    lines.append(f"Analyzer: {Path(analyzer.folder).name if getattr(analyzer, 'folder', None) else 'N/A'}")
    lines.append(f"Units: {analyzer.sorting.get_num_units()}")
    try:
        lines.append(f"Sampling rate (Hz): {analyzer.sorting.get_sampling_frequency():.2f}")
    except Exception:
        lines.append("Sampling rate (Hz): N/A")

    # Extensions present
    try:
        ext_names = list(analyzer.get_extension_names())
    except Exception:
        ext_names = []
    lines.append(f"Extensions: {', '.join(ext_names) if ext_names else 'None'}")

    # Waveform window if available
    if analyzer.has_extension("waveforms"):
        wf = analyzer.get_extension("waveforms")
        ms_before = wf.params.get("ms_before", None)
        ms_after  = wf.params.get("ms_after", None)
        lines.append(f"Waveforms window (ms): {ms_before} before, {ms_after} after")
    else:
        lines.append("Waveforms window (ms): N/A (no 'waveforms' extension)")

    # Recording info (light)
    try:
        rec = analyzer.recording or (analyzer.get_temporary_recording() if analyzer.has_temporary_recording() else None)
        if rec is not None:
            lines.append(f"Recording: {type(rec).__name__}")
            try:
                lines.append(f"Channels: {rec.get_num_channels()}")
            except Exception:
                pass
    except Exception:
        pass

    fig = plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.text(
        0.01, 0.99,
        "\n".join(lines),
        va="top", ha="left",
        fontsize=12,
        family="monospace"
    )
    fig.tight_layout()

    out_path = out_dir / f"{basename}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved summary image:", out_path)


def save_quality_histograms(analyzer, out_dir: Path, cols=("firing_rate", "snr", "isi_violations_ratio")):
    """If quality_metrics exists, save a histogram panel image for a few common columns."""
    if not analyzer.has_extension("quality_metrics"):
        print("No 'quality_metrics' extension found; skipping quality histograms.")
        return

    qm = analyzer.get_extension("quality_metrics")

    # qm.get_data() is typically a pandas DataFrame; keep it flexible
    try:
        df = qm.get_data()
    except Exception as e:
        print("Could not read quality metrics data:", e)
        return

    available = [c for c in cols if c in getattr(df, "columns", [])]
    if not available:
        print("Quality metrics present, but none of the requested columns are available.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3))
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, available):
        vals = df[c].to_numpy() if hasattr(df[c], "to_numpy") else np.asarray(df[c])
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40)
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel("count")

    fig.tight_layout()
    out_path = out_dir / "quality_histograms.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved quality histogram image:", out_path)
    
def plot_peak_localization(
    recording,
    motion_info,
    output_path,
    time_lim0=300.0,
    time_lim1=3300.0,
    subsample=5,
):
    import matplotlib.pyplot as plt
    import spikeinterface as si
    from spikeinterface.preprocessing import load_motion_info
    from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

    def _short_chan_label(ch_id):
        s = str(ch_id)
        if "#" in s:
            s = s.split("#")[-1]
        return s.replace("AP", "").strip() or s

    def plot_probe_custom_on_axis(ax, recording, label_stride=8, show_site_labels=False):
        probe = recording.get_probe()
        positions = np.asarray(probe.contact_positions)
        contact_ids = probe.contact_ids

        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        probe_center_x = (x_min + x_max) / 2
        x_offset = -probe_center_x

        shifted = positions.copy()
        shifted[:, 0] += x_offset

        x_min_s, x_max_s = shifted[:, 0].min(), shifted[:, 0].max()
        padding = 15

        ax.scatter(
            shifted[:, 0],
            shifted[:, 1],
            s=50,
            c="lightgray",
            alpha=0.45,
            marker="s",
            edgecolors="gray",
            linewidths=0.3,
        )

        if show_site_labels:
            for i in range(0, len(shifted), label_stride):
                ax.text(
                    shifted[i, 0],
                    shifted[i, 1],
                    _short_chan_label(contact_ids[i]),
                    fontsize=5,
                    ha="center",
                    va="center",
                    color="black",
                    weight="bold",
                )

        total_w = (x_max_s - x_min_s) + 2 * padding + 40

        ax.set_xlim(-total_w / 2, total_w / 2)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_facecolor("white")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")

        return x_offset


    peaks = motion_info["peaks"]
    peak_locations = motion_info["peak_locations"]
    motion = motion_info["motion"]

    print("Computing corrected peak locations...")
    corrected_locations = correct_motion_on_peaks(
        peaks,
        peak_locations,
        motion,
        recording,
    )

    sr = float(recording.get_sampling_frequency())
    dur = float(recording.get_total_duration())

    time_lim1 = min(time_lim1, dur)

    t0 = int(time_lim0 * sr)
    t1 = int(time_lim1 * sr)

    sample_idx = peaks["sample_index"]
    mask = (sample_idx >= t0) & (sample_idx < t1)

    idx = np.flatnonzero(mask)[::subsample]

    amp_key = "amplitude" if "amplitude" in peaks.dtype.names else None
    if amp_key is None:
        raise KeyError(
            f"No amplitude column in peaks. Available: {peaks.dtype.names}"
        )

    amps = np.abs(np.asarray(peaks[amp_key][idx], dtype=float))
    amps = amps / (np.quantile(amps, 0.95) + 1e-12)

    c = plt.get_cmap("inferno")(np.clip(amps, 0, 1))

    color_kwargs = dict(
        alpha=0.35,
        s=4,
        c=c,
    )

    probe = recording.get_probe()
    py = np.asarray(probe.contact_positions)[:, 1]
    y_margin = 50.0
    ylim_probe = (
        float(py.min()) - y_margin,
        float(py.max()) + y_margin,
    )

    fig, axs = plt.subplots(
        ncols=2,
        figsize=(20, 12),
        sharey=True,
    )

    ax = axs[0]
    x_offset = plot_probe_custom_on_axis(ax, recording)

    ax.scatter(
        np.asarray(peak_locations["x"][idx], dtype=float) + x_offset,
        np.asarray(peak_locations["y"][idx], dtype=float),
        **color_kwargs,
    )

    ax.set_title(
        "Peak locations before motion correction",
        fontsize=13,
        fontweight="bold",
    )

    ax = axs[1]
    x_offset = plot_probe_custom_on_axis(ax, recording)

    ax.scatter(
        np.asarray(corrected_locations["x"][idx], dtype=float) + x_offset,
        np.asarray(corrected_locations["y"][idx], dtype=float),
        **color_kwargs,
    )

    ax.set_title(
        "Peak locations after motion correction",
        fontsize=13,
        fontweight="bold",
    )

    for ax in axs:
        ax.set_ylim(ylim_probe)

    fig.suptitle(
        f"Peak localization on probe — time "
        f"{time_lim0:.1f}–{time_lim1:.1f} s "
        f"(of {dur:.1f} s total)",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Peak localization plot saved to: {output_path}")
    print("\n=== PEAK LOCALIZATION STATS ===")
    print(f"Time window: {time_lim0:.1f}–{time_lim1:.1f} s")
    print(f"Peaks in window (before subsample): {int(np.sum(mask)):,}")
    print(f"Peaks plotted (subsample x{subsample}): {idx.size:,}")
    
def save_bombcell_static_plots(phy_out, bombcell_dir, save_dir, quality_metrics, titles, param):
    output_dir = save_dir / "bombcell_unit_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving BombCell static unit plots to: {output_dir}")

    label_colors = {
        "GOOD": "#008000",
        "MUA": "#ff8b00",
        "NOISE": "#ff0000",
        "NON-SOMA": "#4069e0",
    }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FigureCanvasAgg is non-interactive"
        )

        with capture_output():
            gui = bc.unit_quality_gui(
                str(phy_out),
                quality_metrics=quality_metrics,
                param=param,
                save_path=str(bombcell_dir),
                auto_advance=False
            )
            
    # print("GUI units:", gui.n_units)
    # print("BombCell labels:", len(titles))

    # for unit_idx in range(min(gui.n_units, len(titles))):
    #     unit_data = gui.get_unit_data(unit_idx)
    #     print(
    #         unit_idx,
    #         "GUI unit_id =", unit_data["unit_id"],
    #         "BombCell label =", titles[unit_idx]
    #     )

    for unit_idx in tqdm(range(gui.n_units), desc="Saving unit plots", unit="unit"):
        try:
            gui.current_unit_idx = unit_idx

            unit_data = gui.get_unit_data(unit_idx)
            unit_id = unit_data["unit_id"]

            label = str(titles[unit_id]).strip().upper()
            border_color = label_colors.get(label, "#808080")

            fig = plt.figure(figsize=(45, 25))

            ax_location = plt.subplot2grid((100, 30), (0, 0), rowspan=100, colspan=1)
            gui.plot_unit_location(ax_location, unit_data)

            ax_template = plt.subplot2grid((100, 30), (0, 2), rowspan=20, colspan=6)
            gui.plot_template_waveform(ax_template, unit_data)

            ax_raw = plt.subplot2grid((100, 30), (0, 9), rowspan=20, colspan=6)
            gui.plot_raw_waveforms(ax_raw, unit_data)

            ax_spatial = plt.subplot2grid((100, 30), (30, 2), rowspan=20, colspan=6)
            gui.plot_spatial_decay(ax_spatial, unit_data)

            ax_acg = plt.subplot2grid((100, 30), (30, 9), rowspan=20, colspan=6)
            gui.plot_autocorrelogram(ax_acg, unit_data)

            ax_amplitude = plt.subplot2grid((100, 30), (60, 2), rowspan=20, colspan=10)
            gui.plot_amplitudes_over_time(ax_amplitude, unit_data)

            ax_bin_metrics = plt.subplot2grid(
                (100, 30),
                (85, 2),
                rowspan=10,
                colspan=10,
                sharex=ax_amplitude
            )
            gui.plot_time_bin_metrics(ax_bin_metrics, unit_data)

            ax_amp_fit = plt.subplot2grid((100, 30), (60, 13), rowspan=20, colspan=2)
            gui.plot_amplitude_fit(ax_amp_fit, unit_data)

            gui.plot_histograms_panel(fig, unit_data)

            fig.suptitle(
                f"BOMBCELL LABEL: {label}",
                fontsize=30,
                fontweight="bold",
                y=1.01
            )

            fig.subplots_adjust(
                left=0.03,
                right=0.97,
                top=0.94,
                bottom=0.05,
                wspace=0.4,
                hspace=0.8,
            )

            border = plt.Rectangle(
                (-0.05, -0.08),
                1.14,
                1.05,
                transform=fig.transFigure,
                fill=False,
                edgecolor=border_color,
                linewidth=18,
                zorder=1000,
                clip_on=False,
            )
            fig.patches.append(border)

            output_path = output_dir / f"unit_{unit_id}.png"

            fig.savefig(
                output_path,
                dpi=75,
                bbox_inches="tight",
                pad_inches=0.15,
            )

            plt.close(fig)

        except Exception as e:
            print(f"Error saving plot for unit {unit_id}: {e}")
            print(f"Skipping unit {unit_id}.")

    print(f"Finished BombCell static plots: {output_dir}")
    
def save_bombcell_detail_plots(plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    original_show = plt.show
    plot_count = 0

    def save_show(*args, **kwargs):
        nonlocal plot_count

        fig = plt.gcf()

        if fig.get_axes():
            plot_count += 1
            ax = fig.get_axes()[0]

            title = ax.get_title().strip()

            if not title:
                labels = [
                    text.get_text().strip()
                    for text in ax.get_legend().get_texts()
                ] if ax.get_legend() else []

                if labels:
                    title = "_".join(labels)

            if not title:
                title = f"plotDetails_{plot_count:04d}"

            title = re.sub(r'[<>:"/\\|?*]', '_', title)
            title = re.sub(r'\s+', '_', title)

            filename = plot_dir / f"{title}_{plot_count:04d}.png"

            fig.savefig(
                filename,
                dpi=150,
                bbox_inches="tight"
            )

        plt.close(fig)

    plt.show = save_show

    return original_show

def save_bombcell_kilosort_label_graph(bombcell_labels, kilosort_labels, plot_dir):
    bombcell_counts = Counter(bombcell_labels)
    kilosort_counts = Counter(kilosort_labels)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    for ax, counts, title in [
        (axes[0, 0], bombcell_counts, "BombCell Unit Labels"),
        (axes[0, 1], kilosort_counts, "Kilosort Unit Labels"),
    ]:
        labels = list(counts.keys())
        values = list(counts.values())

        bar_colors = [
            "green" if str.lower(str(label)) == "good" else "blue"
            for label in labels
        ]

        bars = ax.bar(labels, values, color=bar_colors)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
            )

        ax.set_title(title)
        ax.set_xlabel("Unit label")
        ax.set_ylabel("Number of units")
        ax.tick_params(axis="x", rotation=45)

    label_pairs = Counter(zip(kilosort_labels, bombcell_labels))

    kilosort_order = list(dict.fromkeys(kilosort_labels))
    bombcell_order = list(dict.fromkeys(bombcell_labels))

    heatmap = np.zeros(
        (len(kilosort_order), len(bombcell_order)),
        dtype=int,
    )

    kilosort_index = {label: i for i, label in enumerate(kilosort_order)}
    bombcell_index = {label: i for i, label in enumerate(bombcell_order)}

    for (kilosort_label, bombcell_label), count in label_pairs.items():
        heatmap[
            kilosort_index[kilosort_label],
            bombcell_index[bombcell_label],
        ] = count

    ax = axes[1, 0]
    axes[1, 1].axis("off")

    im = ax.imshow(heatmap, cmap="YlOrBr", aspect="auto")

    ax.set_xticks(range(len(bombcell_order)))
    ax.set_yticks(range(len(kilosort_order)))
    ax.set_xticklabels(bombcell_order, rotation=45, ha="right")
    ax.set_yticklabels(kilosort_order)

    ax.set_xlabel("BombCell Label")
    ax.set_ylabel("Kilosort Label")
    ax.set_title("Kilosort vs BombCell Labels")

    for i in range(len(kilosort_order)):
        for j in range(len(bombcell_order)):
            ax.text(
                j,
                i,
                str(heatmap[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, label="Number of units")

    plt.tight_layout()
    plt.savefig(
        plot_dir / "bombcell_vs_kilosort_labels.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
    
    return str(plot_dir / "bombcell_vs_kilosort_labels.png")
    
def slay_graphs(unit_a, unit_b, latent_2d, labels, correlograms, idx_a, idx_b, lags, slay_metrics, slay_plot_dir):


    auto_a = correlograms[idx_a, idx_a]
    auto_b = correlograms[idx_b, idx_b]
    cross_ab = correlograms[idx_a, idx_b]

    similarity = slay_metrics["similarity"][idx_a, idx_b]
    ccg_metric = slay_metrics["ccg_metric"][idx_a, idx_b]
    refractory_penalty = slay_metrics["refractory_penalty"][idx_a, idx_b]
    final_metric = slay_metrics["final_metric"][idx_a, idx_b]

    fig = plt.figure(figsize=(9, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 2.5, 1.5], hspace=0.4)

    ax_pca = fig.add_subplot(gs[0])
    ax_pca.scatter(latent_2d[labels == 0, 0], latent_2d[labels == 0, 1], label=f"Unit {unit_a}", color="red", alpha=0.5, s=8)
    ax_pca.scatter(latent_2d[labels == 1, 0], latent_2d[labels == 1, 1], label=f"Unit {unit_b}", color="blue", alpha=0.5, s=8)
    ax_pca.set_xlabel("PC 1")
    ax_pca.set_ylabel("PC 2")
    ax_pca.set_title(f"Latent Space PCA: Units {unit_a} + {unit_b}")
    ax_pca.legend()

    ax_corr = fig.add_subplot(gs[1])
    ax_corr.plot(lags, auto_a, color="red", label=f"Unit {unit_a} auto")
    ax_corr.plot(lags, auto_b, color="blue", label=f"Unit {unit_b} auto")
    ax_corr.plot(lags, cross_ab, color="black", label=f"{unit_a} → {unit_b} cross")
    ax_corr.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_corr.set_xlabel("Lag (ms)")
    ax_corr.set_ylabel("Count")
    ax_corr.set_title("Correlograms")
    ax_corr.legend()

    ax_metrics = fig.add_subplot(gs[2])
    ax_metrics.axis("off")

    metric_text = (
        f"Similarity:            {similarity:.4f}\n"
        f"CCG metric:            {ccg_metric:.4f}\n"
        f"Refractory penalty:    {refractory_penalty:.4f}\n"
        f"Final metric:          {final_metric:.4f}"
    )

    ax_metrics.text(0, 1, metric_text, fontsize=10, verticalalignment="top", horizontalalignment="left", family="monospace")
    ax_metrics.set_title("SLAy Metrics", loc="left")

    fig.suptitle(f"SLAy Merge: Units {unit_a} + {unit_b}", fontsize=14)
    plt.savefig(slay_plot_dir / f"slay_merge_{unit_a}_{unit_b}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_process_timing_graph(times, plot_path):
    processes = list(times.keys())
    parts = []

    for process_data in times.values():
        for part in process_data:
            if part not in parts:
                parts.append(part)

    subparts = {}
    for part in parts:
        subparts[part] = []
        for process_data in times.values():
            if part not in process_data:
                continue
            for subpart in process_data[part]:
                if subpart not in subparts[part]:
                    subparts[part].append(subpart)

    n_processes = len(processes)
    process_totals = {}

    for process, process_data in times.items():
        total = 0
        for part_data in process_data.values():
            total += sum(part_data.values())
        process_totals[process] = total

    part_subpart_totals = {
        part: {subpart: 0 for subpart in subparts[part]}
        for part in parts
    }

    part_totals = {part: 0 for part in parts}

    for process_data in times.values():
        for part in parts:
            if part not in process_data:
                continue

            for subpart, value in process_data[part].items():
                part_subpart_totals[part][subpart] += value
                part_totals[part] += value

    viridis = plt.colormaps["viridis"]

    if len(parts) == 1:
        part_color_positions = [0.5]
    else:
        part_color_positions = np.linspace(0.15, 0.85, len(parts))

    part_base_colors = {
        part: viridis(position)
        for part, position in zip(parts, part_color_positions)
    }

    subpart_colors = {}

    for part in parts:
        n_subparts = len(subparts[part])

        if n_subparts == 1:
            shade_values = [0.65]
        else:
            shade_values = np.linspace(0.35, 0.9, n_subparts)

        subpart_colors[part] = {}

        base = np.array(part_base_colors[part][:3])

        for subpart, shade in zip(subparts[part], shade_values):
            color = base * shade
            subpart_colors[part][subpart] = tuple(color)

    fig = plt.figure(figsize=(15, 8))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.5, 1.3],
        height_ratios=[5.0, 1.8],
        wspace=0.08,
        hspace=0.18,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 0])

    y_positions = np.arange(n_processes)

    for y, process in zip(y_positions, processes):
        process_data = times[process]
        total = process_totals[process]
        left = 0.0

        for part in parts:
            if part not in process_data:
                continue

            for subpart in subparts[part]:
                value = process_data[part].get(subpart, 0)

                if value == 0:
                    continue

                percentage = 100 * value / total

                ax_main.barh(
                    y,
                    percentage,
                    left=left,
                    height=0.72,
                    color=subpart_colors[part][subpart],
                    edgecolor="white",
                    linewidth=0.5,
                )

                left += percentage

    ax_main.set_xlim(0, 100)
    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(processes)
    ax_main.invert_yaxis()
    ax_main.set_xlabel("Percentage of total computation time per shank")
    ax_main.set_ylabel("Shank")
    ax_main.set_title(
        "Computation Time by Shank",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax_main.set_xticks(np.arange(0, 101, 20))
    ax_main.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
    ax_main.grid(axis="x", linestyle="--", alpha=0.25)
    ax_main.set_axisbelow(True)

    totals = [process_totals[process] for process in processes]

    ax_right.barh(
        y_positions,
        totals,
        height=0.72,
        color="black",
    )

    ax_right.set_xlabel("Total time (s)")
    ax_right.set_yticks(y_positions)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.grid(axis="x", linestyle="--", alpha=0.25)
    ax_right.set_axisbelow(True)

    max_total = max(totals) if totals else 1
    ax_right.set_xlim(0, max_total * 1.05)

    for y, total in zip(y_positions, totals):
        ax_right.text(
            total - max_total * 0.02,
            y,
            f"{total:.2f} s",
            va="center",
            ha="right",
            fontsize=9,
            color="white",
        )

    x_positions = np.arange(len(parts))

    for part in parts:
        values = np.array([
            part_subpart_totals[part][subpart]
            for subpart in subparts[part]
        ])

        x = x_positions[parts.index(part)]
        current_bottom = 0

        for subpart, value in zip(subparts[part], values):
            ax_bottom.bar(
                x,
                value,
                bottom=current_bottom,
                width=0.72,
                color=subpart_colors[part][subpart],
                edgecolor="white",
                linewidth=0.5,
            )

            current_bottom += value

    ax_bottom.set_xticks(x_positions)
    ax_bottom.set_xticklabels(parts)
    ax_bottom.set_ylabel("Total time (s)")
    ax_bottom.set_xlabel("Step")
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bottom.set_axisbelow(True)

    max_part_total = max(part_totals.values()) if part_totals else 1

    for x, part in zip(x_positions, parts):
        total = part_totals[part]

        ax_bottom.text(
            x,
            total - max_part_total * 0.04,
            f"{total:.2f} s",
            ha="center",
            va="top",
            fontsize=9,
            color="white",
        )

    ax_bottom.set_ylim(0, max_part_total * 1.15)

    legend_handles = []

    for part in parts:
        for subpart in subparts[part]:
            legend_handles.append(
                Patch(
                    facecolor=subpart_colors[part][subpart],
                    edgecolor="none",
                    label=f"{part} — {subpart}",
                )
            )

    fig.suptitle(
        "Computation Time Breakdown",
        fontsize=16,
        fontweight="bold",
        y=1.04,
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)