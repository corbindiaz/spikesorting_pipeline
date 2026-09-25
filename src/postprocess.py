import time
import shutil

from utils import import_analyzer, timed
from visualization import (save_probe_figure, save_text_summary_image, save_quality_histograms)

import spikeinterface.full as si
import spikeinterface.curation as sc

import matplotlib.pyplot as plt


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

        try:
            # Import
            describe = (step == 0)
            with timed(postprocessing_time, "Import"):
                analyzer, rec, raw_file, meta_file = import_analyzer(
                    analyzer_path, VERBOSE_INVENTORY, describe,
                    dtype="float32",
                )

            # Remove empty / excess-spike units
            with timed(postprocessing_time, "Unit Cleanup"):
                sorting = analyzer.sorting
                n_original_units = int(len(sorting.unit_ids))

                sorting = sorting.remove_empty_units()
                sorting = sc.remove_excess_spikes(sorting=sorting, recording=rec)

                n_non_empty_units = int(len(sorting.unit_ids))
                n_removed_units = n_original_units - n_non_empty_units

                if n_removed_units > 0:
                    print(f"Removed {n_removed_units} units due to zero spikes "
                          f"or excess spikes; {n_non_empty_units} units remaining.")
                    analyzer = si.create_sorting_analyzer(
                        sorting=sorting,
                        recording=rec,
                        format="zarr",
                        folder=analyzer_path,
                        overwrite=True,
                        **job_kwargs,
                    )
                else:
                    print(f"No units removed; {n_non_empty_units} units remaining.")

            # Extensions
            print("Computing extensions...")
            loaded_extensions = analyzer.get_loaded_extension_names()
            
            # A saved quality_metrics from a run that skipped snr would otherwise be kept forever
            if "quality_metrics" in loaded_extensions:
                qm = analyzer.get_extension("quality_metrics").get_data()
                if not {"snr", "firing_rate"} <= set(qm.columns):
                    print("Existing quality metrics are missing snr/firing_rate; recomputing.")
                    analyzer.delete_extension("quality_metrics")
                    loaded_extensions = analyzer.get_loaded_extension_names()

            all_extensions_computed = True

            for ext_name, label, message, ext_kwargs in extension_plan:
                if ext_name in loaded_extensions:
                    continue

                all_extensions_computed = False
                print(message)
                with timed(postprocessing_time, label):
                    analyzer.compute(ext_name, save=True, **ext_kwargs)
                print(f"  {label} time: {postprocessing_time[label]:.2f} seconds")

            if all_extensions_computed:
                print("All extensions present. No computation needed.")
            print()

            # Plots
            with timed(postprocessing_time, "Plots"):
                save_probe_figure(rec, plots_group, basename="probe_layout")
                save_text_summary_image(analyzer, plots_group, basename="summary")
                save_quality_histograms(analyzer, plots_group)
                plt.close("all")

        finally:
            # Per-group remainder
            total = time.perf_counter() - group_start
            accounted = sum(v for k, v in postprocessing_time.items() if k != "Other")
            postprocessing_time["Other"] = max(0.0, total - accounted)
            print(f"Total time post-processing {analyzer_path.name}: {total:.2f} seconds")

    return time_master