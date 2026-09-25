import time
import shutil
from pathlib import Path

import numpy as np
import warnings

from utils import timed, recording_summary

import torch
from kilosort.run_kilosort import close_logger
import spikeinterface.full as si
import spikeinterface.sorters as ss

import matplotlib.pyplot as plt

def sorter_kilosort(
    recording_path_,
    output_folder,
    params,
    time_master,
    step=1
):
    preprocessed_folder = output_folder / "preprocessed"
    kilosort_folder = output_folder / "kilosort"
    analyzer_folder = output_folder / "analyzer"
    plots = output_folder / "plots"
    motion_plots_folder = plots / "motion"

    kilosort_folder.mkdir(parents=True, exist_ok=True)
    analyzer_folder.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    motion_plots_folder.mkdir(parents=True, exist_ok=True)

    kilosort_params = params["kilosort"]

    preprocessed_recording_paths = sorted(
        p for p in preprocessed_folder.iterdir()
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
                    **kilosort_params,
                )

                n_original_units = int(len(sorting.unit_ids))
                print(f"Kilosort4 found {n_original_units} units.")

                sorting = _fix_object_properties(sorting)

            # Read Kilosort motion
            motion = None
            if kilosort_params["do_correction"]:
                with timed(kilosort_time, "Motion"):
                    motion = read_kilosort4_motion(
                        sorter_output_folder,
                        recording=recording
                    )

            # Close Kilosort's log file before deleting sorter output
            close_logger()

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
                    **params['job_kwargs']
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
                        si.plot_motion(motion, mode = "map", figure=fig)
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


def read_kilosort4_motion(
    sorter_output_folder: str | Path,
    recording: si.BaseRecording | None = None
) -> si.Motion:
    """Read motion information from a Kilosort4 output folder."""

    sorter_output_folder = Path(sorter_output_folder)
    ops_file = sorter_output_folder / "ops.npy"

    if not ops_file.is_file():
        raise FileNotFoundError(
            "'ops.npy' file not found!"
        )

    ops = np.load(
        ops_file,
        allow_pickle=True
    ).item()

    yblk = ops.get("yblk")
    dshift = ops.get("dshift")

    if yblk is None or dshift is None:
        warnings.warn(
            "'yblk' and 'dshift' fields not found in ops file!"
        )
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
            t_start + t_bin / 2,
            t_end - t_bin / 2,
            displacement.shape[0]
        )
    else:
        temporal_bins_s = (
            np.arange(displacement.shape[0]) * t_bin
            + t_bin / 2
        )

    motion = si.Motion(
        displacement=displacement,
        temporal_bins_s=temporal_bins_s,
        spatial_bins_um=spatial_bins_um
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