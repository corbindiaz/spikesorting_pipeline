import json
import shutil
import time

import numpy as np

from utils import import_analyzer, timed
from visualization import (save_probe_figure, save_text_summary_image, save_quality_histograms)

import spikeinterface.full as si
import spikeinterface.curation as sc

import matplotlib.pyplot as plt


MAX_PRINTED_UNITS = 10


def postprocess(
    recording_path_,
    output_folder,
    params,
    time_master,
    step=2
):
    if step <= 2:
        VERBOSE_INVENTORY = params['verbose_inventory']
    else:
        VERBOSE_INVENTORY = False

    analyzer_folder = output_folder / "analyzer"
    plots = output_folder / "plots"

    group_analyzer_paths = sorted(analyzer_folder.glob("*group*.zarr"))

    job_kwargs = params['job_kwargs']
    min_spikes = params.get('min_spikes_per_unit', 2)

    # (extension name, timing label, message, compute kwargs), in dependency order
    extension_plan = [
        ("random_spikes", "Random Spikes", "Computing spikes for waveform extraction...",
        dict(method="uniform", max_spikes_per_unit=500)),
        ("waveforms", "Waveforms", "Computing waveforms...",
        dict(**job_kwargs)),
        ("templates", "Templates", "Computing templates...",
        dict(**job_kwargs)),
        ("noise_levels", "Noise Levels", "Computing noise levels...",
        dict(**job_kwargs)),
        ("spike_amplitudes", "Spike Amplitudes", "Computing spike amplitudes...",
        dict(**job_kwargs)),
        ("correlograms", "Correlograms", "Computing correlograms...",
        dict(window_ms=100, bin_ms=1.0, **job_kwargs)),
        ("quality_metrics", "Quality Metrics",
        "Computing Signal-to-Noise Ratio (SNR) and Firing Rate metrics...",
        dict(metric_names=["snr", "firing_rate"], **job_kwargs)),
    ]

    for i, analyzer_path in enumerate(group_analyzer_paths):
        group_time = time_master.setdefault(f"Group{i}", {})
        postprocessing_time = group_time.setdefault("Postprocessing", {})
        group_start = time.perf_counter()

        print("-" * 80)

        plots_group = plots / analyzer_path.stem
        plots_group.mkdir(parents=True, exist_ok=True)

        removed_log = {}

        try:
            # Import
            describe = (step == 0)
            with timed(postprocessing_time, "Import"):
                analyzer, rec, raw_file, meta_file = import_analyzer(
                    analyzer_path, VERBOSE_INVENTORY, describe,
                    dtype="float32",
                )

            # Stage 1: remove empty, excess-spike and too-few-spike units
            with timed(postprocessing_time, "Unit Cleanup"):
                sorting = analyzer.sorting
                n_original_units = int(len(sorting.unit_ids))

                sorting = sorting.remove_empty_units()
                sorting = sc.remove_excess_spikes(sorting=sorting, recording=rec)

                counts = sorting.count_num_spikes_per_unit()
                low_spike = {
                    uid: dict(reasons=[f"{int(n)} spike(s) < min {min_spikes}"],
                              n_spikes=int(n), n_channels=None)
                    for uid, n in counts.items() if n < min_spikes
                }
                if low_spike:
                    sorting = sorting.select_units(
                        [u for u in sorting.unit_ids if u not in low_spike])

                n_kept = int(len(sorting.unit_ids))
                n_removed = n_original_units - n_kept

                if n_removed > 0:
                    print(f"Removed {n_removed} unit(s) (empty, excess spikes, or "
                          f"< {min_spikes} spikes); {n_kept} remaining.")
                    if low_spike:
                        report_removed(low_spike, "spike-count filter", analyzer_path.stem)
                        removed_log.update({str(u): {**v, "stage": "spike_count"}
                                            for u, v in low_spike.items()})
                    analyzer = rebuild_analyzer(
                        sorting, rec, analyzer_path, job_kwargs, output_folder)
                else:
                    print(f"No units removed; {n_kept} remaining.")

                # Stage 2: units with zero channels in the sparsity mask
                bad = find_bad_units(analyzer, min_spikes, check_templates=False)
                if bad:
                    report_removed(bad, "sparsity check", analyzer_path.stem)
                    removed_log.update({str(u): {**v, "stage": "sparsity"}
                                        for u, v in bad.items()})
                    keep = [u for u in analyzer.unit_ids if u not in bad]
                    analyzer = rebuild_analyzer(
                        analyzer.sorting.select_units(keep), rec, analyzer_path,
                        job_kwargs, output_folder)

            # Extensions
            print("Computing extensions...")
            compute_extensions(analyzer, extension_plan, postprocessing_time)
            print()

            # Stage 3: template validation (requires computed templates)
            with timed(postprocessing_time, "Template Validation"):
                bad = find_bad_units(analyzer, min_spikes, check_templates=True)
                if bad:
                    report_removed(bad, "template check", analyzer_path.stem)
                    removed_log.update({str(u): {**v, "stage": "template"}
                                        for u, v in bad.items()})
                    keep = [u for u in analyzer.unit_ids if u not in bad]
                    analyzer = rebuild_analyzer(
                        analyzer.sorting.select_units(keep), rec, analyzer_path,
                        job_kwargs, output_folder)
                    print("Recomputing extensions after unit removal...")
                    compute_extensions(analyzer, extension_plan, postprocessing_time,
                                       label_suffix=" (recompute)")

                    still_bad = find_bad_units(analyzer, min_spikes, check_templates=True)
                    if still_bad:
                        raise RuntimeError(
                            f"Invalid units remain after cleanup in {analyzer_path.name}: "
                            f"{list(still_bad.keys())}")

            # Diagnostics summary and persistent record
            n_final = int(len(analyzer.unit_ids))
            n_ch = (analyzer.sparsity.mask.sum(axis=1) if analyzer.sparsity is not None
                    else None)
            final_counts = analyzer.sorting.count_num_spikes_per_unit()
            min_spk_final = min(final_counts.values())
            msg = f"Validation OK: {n_final} units, min spikes/unit = {min_spk_final}"
            if n_ch is not None:
                msg += f", min channels/unit = {int(n_ch.min())}"
            print(msg)

            if removed_log:
                print(f"  [WARNING] {len(removed_log)} unit(s) removed in total "
                      f"from {analyzer_path.name}.")
            with open(plots_group / "removed_units.json", "w") as f:
                json.dump(dict(min_spikes_per_unit=min_spikes,
                               n_units_final=n_final,
                               removed=removed_log), f, indent=2, default=str)
                
            expected = {"KSLabel", "KSLabel_repeat", "Amplitude", "ContactPct", "original_cluster_id"}
            missing = expected - set(analyzer.sorting.get_property_keys())
            if missing:
                print(f"  [WARNING] Missing sorting properties after cleanup: {sorted(missing)}")

            # Plots
            with timed(postprocessing_time, "Plots"):
                save_probe_figure(rec, plots_group, basename="probe_layout")
                save_text_summary_image(analyzer, plots_group, basename="summary")
                save_quality_histograms(analyzer, plots_group)
                plt.close("all")

        finally:
            total = time.perf_counter() - group_start
            accounted = sum(v for k, v in postprocessing_time.items() if k != "Other")
            postprocessing_time["Other"] = max(0.0, total - accounted)
            print(f"Total time post-processing {analyzer_path.name}: {total:.2f} seconds")

    return time_master


def find_bad_units(analyzer, min_spikes, check_templates=False):
    """Return {unit_id: info} for units that would break Phy or downstream steps.

    Reasons: too few spikes, zero channels in sparsity, degenerate template.
    """
    unit_ids = analyzer.unit_ids
    counts = analyzer.sorting.count_num_spikes_per_unit()

    if analyzer.sparsity is not None:
        n_channels = analyzer.sparsity.mask.sum(axis=1)
    else:
        n_channels = np.full(len(unit_ids), analyzer.get_num_channels())

    templates = None
    if check_templates and analyzer.has_extension("templates"):
        templates = analyzer.get_extension("templates").get_data()

    bad = {}
    for idx, uid in enumerate(unit_ids):
        reasons = []
        n_spk = int(counts[uid])
        n_ch = int(n_channels[idx])

        if n_spk < min_spikes:
            reasons.append(f"{n_spk} spike(s) < min {min_spikes}")
        if n_ch == 0:
            reasons.append("0 channels in sparsity")
        if templates is not None:
            t = templates[idx]
            if not np.all(np.isfinite(t)):
                reasons.append("non-finite template")
            elif not np.any(t):
                reasons.append("all-zero template")

        if reasons:
            bad[uid] = dict(reasons=reasons, n_spikes=n_spk, n_channels=n_ch)
    return bad


def report_removed(removed, stage, group_name):
    """Compact terminal alert: one line per unit (capped) plus a summary."""
    if not removed:
        return
    print(f"  [WARNING] {group_name} / {stage}: removed {len(removed)} invalid unit(s):")
    for uid, info in list(removed.items())[:MAX_PRINTED_UNITS]:
        ch = "n/a" if info['n_channels'] is None else info['n_channels']
        print(f"    unit {uid}: {info['n_spikes']} spikes, "
              f"{ch} ch -> {'; '.join(info['reasons'])}")
    if len(removed) > MAX_PRINTED_UNITS:
        print(f"    ... and {len(removed) - MAX_PRINTED_UNITS} more (see removed_units.json)")


def backup_analyzer_once(analyzer_path, output_folder):
    """Copy the analyzer folder to analyzer_backup/ before the first overwrite.

    An existing backup is never replaced, so the pristine original survives reruns.
    The backup lives outside analyzer/ so the '*group*.zarr' glob cannot pick it up.
    """
    backup_root = output_folder / "analyzer_backup"
    backup_root.mkdir(exist_ok=True)
    dst = backup_root / analyzer_path.name
    if dst.exists():
        print(f"  Backup already exists, keeping it: {dst}")
        return dst
    print(f"  Backing up analyzer -> {dst}")
    shutil.copytree(analyzer_path, dst)
    return dst


def rebuild_analyzer(sorting, rec, analyzer_path, job_kwargs, output_folder):
    """Recreate the zarr analyzer from a filtered sorting, after backing up the original."""
    if len(sorting.unit_ids) == 0:
        raise RuntimeError(f"All units removed for {analyzer_path.name}; nothing left to process.")
    # Load the spike trains into memory first: the sorting may be lazily backed by
    # the very zarr folder that is about to be overwritten.
    sorting = si.NumpySorting.from_sorting(sorting, with_metadata=True)
    backup_analyzer_once(analyzer_path, output_folder)
    return si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec,
        format="zarr",
        folder=analyzer_path,
        overwrite=True,
        **job_kwargs,
    )


def compute_extensions(analyzer, extension_plan, postprocessing_time, label_suffix=""):
    loaded_extensions = analyzer.get_loaded_extension_names()

    # A saved quality_metrics from a run that skipped snr would otherwise be kept forever
    if "quality_metrics" in loaded_extensions:
        qm = analyzer.get_extension("quality_metrics").get_data()
        if not {"snr", "firing_rate"} <= set(qm.columns):
            print("Existing quality metrics are missing snr/firing_rate; recomputing.")
            analyzer.delete_extension("quality_metrics")
            loaded_extensions = analyzer.get_loaded_extension_names()

    all_computed = True
    for ext_name, label, message, ext_kwargs in extension_plan:
        if ext_name in loaded_extensions:
            continue
        all_computed = False
        print(message)
        full_label = label + label_suffix
        with timed(postprocessing_time, full_label):
            analyzer.compute(ext_name, save=True, **ext_kwargs)
        print(f"  {full_label} time: {postprocessing_time[full_label]:.2f} seconds")

    if all_computed:
        print("All extensions present. No computation needed.")