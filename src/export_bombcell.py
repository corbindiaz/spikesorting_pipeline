import time
import os
import sys
from IPython.utils.io import capture_output
import warnings
import traceback
import re
from collections import Counter
import argparse
from pathlib import Path
import zarr
from tqdm import tqdm

import numpy as np
import pandas as pd

import bombcell as bc
import bombcell.loading_utils as loading_utils
import bombcell.helper_functions as helper_functions


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from utils import (import_analyzer, timed)
from visualization import (save_bombcell_static_plots, save_bombcell_detail_plots, save_bombcell_kilosort_label_graph)

def export_bombcell(
    recording_path_,
    output_folder,
    params,
    time_master,
    step=3.5
):
    params_bombcell = params['export']['bombcell']

    analyzer_folder = output_folder / "analyzer"

    plots = output_folder / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    group_analyzer_paths = sorted(
        analyzer_folder.glob("*group*.zarr")
    )

    export = output_folder / "export"

    print()
    print("EXPORTING BOMBCELL CONTENTS")
    print()

    save_plot_details = params_bombcell['save_plot_details']
    save_unit_plots = params_bombcell['save_unit_plots']

    if step == 0:
        VERBOSE_INVENTORY = params['verbose_inventory']
    else:
        VERBOSE_INVENTORY = False

    describe = (step == 0)

    loading_utils.load_ephys_data = load_ephys_data_fixed
    helper_functions.load_ephys_data = load_ephys_data_fixed

    for i, analyzer_path in enumerate(group_analyzer_paths):
        # Register timing entries first so partial results survive a crash
        group_time = time_master.setdefault(f"Group{i}", {})
        export_times = group_time.setdefault("Export", {})
        group_start = time.perf_counter()

        print("-" * 80)

        group_name = analyzer_path.stem

        try:
            # ---- Import ----
            with timed(export_times, "Bombcell Import"):
                analyzer, rec, raw_file, meta_file = import_analyzer(
                    analyzer_path,
                    VERBOSE_INVENTORY,
                    describe,
                    dtype="float32",
                )

            print(f"Bombcell raw file: {raw_file} | meta file: {meta_file}")

            # ---- Output folders ----
            export_group = export / group_name
            export_group.mkdir(parents=True, exist_ok=True)

            plots_group = plots / group_name
            plots_group.mkdir(parents=True, exist_ok=True)
            
            phy = export_group / "phy"
            

            kilosort_labels = analyzer.sorting.get_property("KSLabel")

            # ---- Bombcell processing ----
            with timed(export_times, "Bombcell Processing"):
                result = process_bombcell(
                    export_group,
                    plots_group,
                    raw_file,
                    meta_file,
                    plot_details=save_plot_details
                )

            if result:
                print(f"{analyzer_path.name}: BOMBCELL SUCCESS")

                bombcell = export_group / "bombcell"

                # ---- Static plots ----
                if save_unit_plots:
                    with timed(export_times, "Bombcell Static Plots"):
                        try:
                            save_bombcell_static_plots(
                                phy_out=phy,
                                bombcell_dir=bombcell,
                                save_dir=plots_group,
                                quality_metrics=result["quality_metrics"],
                                titles=result["unit_type_string"],
                                param=result["param"],
                            )
                        except Exception as e:
                            print(f"{analyzer_path.name}: STATIC PLOTS FAILED")
                            print(f"Error: {e}")
                            traceback.print_exc()

                bombcell_labels = result["unit_type_string"]

                # ---- KS/Bombcell comparison plot ----
                with timed(export_times, "Bombcell Label Comparison"):
                    bklg_file_name = save_bombcell_kilosort_label_graph(
                        bombcell_labels,
                        kilosort_labels,
                        plots_group
                    )

                print(
                    "Saved Bombcell & Kilosort Unit Label Comparison Plot to "
                    + bklg_file_name
                )

            else:
                print(
                    f"{analyzer_path.name}: "
                    "BOMBCELL FAILED "
                    "************************************************"
                )

            plt.close("all")

        finally:
            # Per-group remainder, computed from what was actually recorded
            total = time.perf_counter() - group_start
            accounted = sum(
                v for k, v in export_times.items()
                if k != "Other"
            )
            export_times["Bombcell Other"] = max(
                0.0,
                total - accounted
            )

            print(
                f"Total time exporting Bombcell {group_name}: "
                f"{total:.2f} seconds"
            )

    return time_master

def load_ephys_data_fixed(ephys_path):
    ephys_path = Path(ephys_path)

    spike_templates = np.load(ephys_path / "spike_templates.npy").squeeze()

    if (ephys_path / "spike_times_corrected.npy").exists():
        spike_times_samples = np.load(ephys_path / "spike_times_corrected.npy").squeeze()
    else:
        spike_times_samples = np.load(ephys_path / "spike_times.npy").squeeze()

    template_amplitudes = np.load(ephys_path / "amplitudes.npy").squeeze().astype(np.float64)

    templates_waveforms_whitened = np.load(ephys_path / "templates.npy")
    template_ind = np.load(ephys_path / "template_ind.npy")
    channel_positions = np.load(ephys_path / "channel_positions.npy").squeeze()

    n_templates = templates_waveforms_whitened.shape[0]
    n_samples = templates_waveforms_whitened.shape[1]
    n_channels = len(channel_positions)

    if (ephys_path / "whitening_mat_inv.npy").exists():
        winv = np.load(ephys_path / "whitening_mat_inv.npy")
    else:
        winv = np.eye(n_channels)

    templates_waveforms = np.zeros(
        (n_templates, n_samples, n_channels),
        dtype=templates_waveforms_whitened.dtype
    )

    for t in range(n_templates):
        channels = template_ind[t]
        valid = channels >= 0
        channels = channels[valid]

        template = templates_waveforms_whitened[t][:, valid]

        if winv.shape == (n_channels, n_channels):
            winv_local = winv[np.ix_(channels, channels)]
            template_unwhitened = template @ winv_local
        else:
            template_unwhitened = template

        templates_waveforms[t][:, channels] = template_unwhitened

    if (ephys_path / "pc_features.npy").exists():
        pc_features = np.load(ephys_path / "pc_features.npy").squeeze()
        pc_features_idx = np.load(ephys_path / "pc_feature_ind.npy").squeeze()
    else:
        pc_features = np.nan
        pc_features_idx = np.nan

    spike_templates, templates_waveforms, pc_features_idx = loading_utils.handle_manual_curation(
        ephys_path,
        spike_templates,
        templates_waveforms,
        pc_features_idx
    )

    return (
        spike_times_samples,
        spike_templates,
        templates_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions
    )


def validate_phy_directory(phy_dir):
    phy_dir = Path(phy_dir)
    
    required = [
        "spike_templates.npy",
        "spike_times.npy",
        "amplitudes.npy",
        "templates.npy",
        "template_ind.npy",
        "channel_positions.npy",
    ]

    missing = [
            name
            for name in required
            if not (phy_dir / name).exists()
        ]
    
    if missing:
        print("Missing required files:")
        for name in missing:
            print(f"  {name}")

        return False

    channel_positions_path = phy_dir / "channel_positions.npy"
    channel_positions = np.load(
        channel_positions_path,
        mmap_mode="r",
    )
    num_chans = len(channel_positions)
    print(f"Number of channels: {num_chans}")

    # Whitening matrix
    whitening_mat_path = phy_dir / "whitening_mat.npy"
    if whitening_mat_path.exists():
        wm = np.load(whitening_mat_path)
        
    else:
        print("Whitening matrix not found. Adding identity whitening matrix.")
        wm = np.eye(num_chans)
        np.save(whitening_mat_path, wm)

    if wm.shape != (num_chans, num_chans):
        raise ValueError(
            f"whitening_mat.npy has shape {wm.shape}, "
            f"expected {(num_chans, num_chans)}"
        )

    # Inverse whitening matrix
    whitening_mat_inv_path = phy_dir / "whitening_mat_inv.npy"

    if whitening_mat_inv_path.exists():
        wmi = np.load(whitening_mat_inv_path)

    else:
        print("Inverse whitening matrix not found. Computing inverse whitening matrix.")
        wmi = np.linalg.inv(wm)
        np.save(whitening_mat_inv_path, wmi)

    if wmi.shape != (num_chans, num_chans):
        raise ValueError(
            f"whitening_mat_inv.npy has shape {wmi.shape}, "
            f"expected {(num_chans, num_chans)}"
        )
        
    return True


def validate_template_dimensions(phy_dir):
    templates = np.load(
        phy_dir / "templates.npy",
        mmap_mode="r",
    )

    template_ind = np.load(
        phy_dir / "template_ind.npy",
        mmap_mode="r",
    )

    winv = np.load(
        phy_dir / "whitening_mat_inv.npy",
        mmap_mode="r",
    )

    if templates.ndim != 3:
        raise ValueError(
            f"templates.npy has unexpected shape {templates.shape}"
        )

    if template_ind.ndim != 2:
        raise ValueError(
            f"template_ind.npy has unexpected shape {template_ind.shape}"
        )

    if templates.shape[0] != template_ind.shape[0]:
        raise ValueError(
            "Number of templates does not match template_ind"
        )

    if templates.shape[2] != template_ind.shape[1]:
        raise ValueError(
            "Template channel dimension does not match template_ind"
        )

    if winv.ndim != 2 or winv.shape[0] != winv.shape[1]:
        raise ValueError(
            f"whitening_mat_inv.npy has unexpected shape {winv.shape}"
        )

    valid_channels = template_ind[template_ind >= 0]

    if len(valid_channels) > 0:
        if valid_channels.max() >= winv.shape[0]:
            raise ValueError(
                "template_ind.npy contains channels outside "
                "whitening_mat_inv.npy"
            )

    return (
        templates.shape,
        template_ind.shape,
        winv.shape,
    )
    
def check_spike_count_discrepancy(phy):
    spike_times_path = phy / "spike_times.npy"
    spike_clusters_path = phy / "spike_clusters.npy"
    amplitudes_path = phy / "amplitudes.npy"

    spike_times = np.load(spike_times_path)
    spike_clusters = np.load(spike_clusters_path)
    amplitudes = np.load(amplitudes_path)

    spike_counts = {
        "spike_times.npy": spike_times.shape[0],
        "spike_clusters.npy": spike_clusters.shape[0],
        "amplitudes.npy": amplitudes.shape[0],
    }

    if len(set(spike_counts.values())) != 1:
        max_spikes = max(spike_counts.values())
        min_spikes = min(spike_counts.values())
        difference = max_spikes - min_spikes

        print()
        print("WARNING: Spike count discrepancy detected in Phy folder")
        print(f"  spike_times.npy:    {spike_counts['spike_times.npy']}")
        print(f"  spike_clusters.npy: {spike_counts['spike_clusters.npy']}")
        print(f"  amplitudes.npy:     {spike_counts['amplitudes.npy']}")
        print(f"  Difference:         {difference} spikes")
        print()
        print(
            f"Removing the last {difference} spikes from the longer "
            "spike arrays so that all spike arrays match."
        )
        print(
            "Please double check the Phy folder to determine why "
            "there is a spike count discrepancy."
        )
        print()

        target_spikes = min_spikes

        if spike_times.shape[0] > target_spikes:
            np.save(spike_times_path, spike_times[:target_spikes])

        if spike_clusters.shape[0] > target_spikes:
            np.save(spike_clusters_path, spike_clusters[:target_spikes])

        if amplitudes.shape[0] > target_spikes:
            np.save(amplitudes_path, amplitudes[:target_spikes])

        print("Spike counts after trimming:")
        print("  spike_times:", np.load(spike_times_path).shape)
        print("  spike_clusters:", np.load(spike_clusters_path).shape)
        print("  amplitudes:", np.load(amplitudes_path).shape)
        print()


def process_bombcell(group_path, plot_dir, raw_file, meta_file, plot_details):
    GAIN_TO_UV = 1
    group_path = Path(group_path)

    phy_dir = group_path / "phy"
    save_path = group_path / "bombcell"
    
    check_spike_count_discrepancy(phy_dir)

    try:
        if not validate_phy_directory(phy_dir):
            print(f"Skipping group: {group_path.name}")
            return False

        try:
            template_shape, template_ind_shape, winv_shape = (
                validate_template_dimensions(phy_dir)
            )
        except Exception as e:
            print()
            print(f"Skipping group: {e}")
            return False

        print()
        print("SortingAnalyzer data:")
        print(f"  templates.npy      {template_shape}")
        print(f"  template_ind.npy   {template_ind_shape}")
        print(f"  whitening_mat_inv  {winv_shape}")

        print()
        print("Raw data:")
        print(f"  {raw_file}")
        print(f"  {meta_file}")

        save_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(f"BombCell output: {save_path}")
        print("Creating BombCell parameters...")
        
        param = bc.get_default_parameters(
            kilosort_path=str(phy_dir),
            raw_file=str(raw_file),
            meta_file=str(meta_file) if meta_file is not None else None,
            kilosort_version=4,
        )

        if meta_file is None:
            # No SpikeGLX .meta, so set what the .meta would have supplied
            param["gain_to_uV"] = GAIN_TO_UV
            print("No .meta file. Gain set manually. Relevant param keys:",
                  [k for k in param if "chan" in k.lower() or "sync" in k.lower() or "gain" in k.lower()])

        param["plotGlobal"] = True
        param["plotDetails"] = plot_details
        param["savePlots"] = True
        param["plotsSaveDir"] = str(plot_dir)

        print("Running BombCell...")
        
        return run_bombcell_extract_graphs(group_path, phy_dir, plot_dir, save_path, plot_details, param)    
        
    except Exception as e:
        print(f"ERROR during BombCell processing {group_path.name}: {e}")
        traceback.print_exc()
        return False

def run_bombcell_extract_graphs(group_path, phy_dir, plot_dir, save_path, plot_details, param):
    original_show = None
    if plot_details:
        detail_plot_dir = plot_dir / "plotDetails"
        original_show = save_bombcell_detail_plots(detail_plot_dir)

    try:
        quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
            ks_dir=str(phy_dir),
            save_path=str(save_path),
            param=param
        )

    except Exception as e:
        print()
        print(f"ERROR processing {group_path.name}:")
        print(e)
        return False

    finally:
        if plot_details and original_show is not None:
            plt.show = original_show

    print()
    print(f"{group_path.name} completed successfully.")
    return {
        "quality_metrics": quality_metrics,
        "param": param,
        "unit_type": unit_type,
        "unit_type_string": unit_type_string,
    }

