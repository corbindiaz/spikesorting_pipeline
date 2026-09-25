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

from sklearn.decomposition import PCA
from slay import compute_slay_merges

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from utils import import_analyzer
from visualization import slay_graphs

def curation(
    recording,
    output_folder,
    params,
    time_master,
    step = 4
):
    pass

def testing():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_folder",
        type=Path,
        required=True,
        help="Root directory for the AINDs Output (contains step subfolders like 'curated', 'postprocessed', etc.)",
    )
    parser.add_argument(
        "--step",
        default="postprocessed",
        help="Step directory name (default: postprocessed)",
    )

    args = parser.parse_args()
    units_to_merge = 'kilosort'
    root = args.results_folder
    results_root = Path(root)
    step_name = args.step
    base = root / step_name

    group_analyzer_paths = sorted(
        base.glob("*_group*.zarr")
    )
    
    # Phy output location
    phy_base_out = root / "phy_out"
    phy_base_out.mkdir(parents=True, exist_ok=True)

    print("Analyzer path:", base)
    for i, analyzer_path in enumerate(group_analyzer_paths):
        print("=" * 80)
        print("Processing:", analyzer_path)
        print()
        result = None
        analyzer, rec, recording_path = import_analyzer(analyzer_path)
        if units_to_merge == "bombcell":
            if not result:
                print(f"{analyzer_path.name}: Bombcell was skipped, no Bombcell labels available for SLAy. Using Kilosort labels instead.")
                unit_labels_list = analyzer.sorting.get_property('KSLabel')
            else:
                unit_labels_list = result["unit_type_string"]

        elif units_to_merge == "kilosort":
            unit_labels_list = analyzer.sorting.get_property('KSLabel')

        else:
            raise ValueError(
                f"Unknown units_to_merge: {units_to_merge!r}. "
                "Expected 'bombcell' or 'kilosort'."
            )

        units_to_keep = [
            unit_id
            for unit_id, unit_type in zip(
                analyzer.unit_ids,
                unit_labels_list
            )
            if str.lower(unit_type) == "good"
        ]

        slay_analyzer = analyzer.select_units(unit_ids=units_to_keep)

        if "peaks" in slay_analyzer.get_saved_extension_names():
            slay_analyzer.delete_extension("peaks")

        print()
        # print(f"Running SLAy on {len(units_to_keep)} units (of {len(analyzer.unit_ids)} total), using '{units_to_merge}' labels...")
        # merges, slay_metrics = slay_and_graph(slay_analyzer, out_dir, plot_dir)

        # slay_end = time.perf_counter()
        # elapsed_slay = slay_end - bombcell_end
        # elapsed = slay_end - start
        # print(f"Total time for SLAy: {elapsed_slay:.2f} seconds")



def slay_and_graph(slay_analyzer, out_dir, plot_dir):
    good_units = slay_analyzer.unit_ids
    captured = {}

    def trace(frame, event, arg):
        if event == "return":
            if "spike_latents" in frame.f_locals:
                captured["spike_latents"] = frame.f_locals["spike_latents"]
            if "spike_labels" in frame.f_locals:
                captured["spike_labels"] = frame.f_locals["spike_labels"]
        return trace

    sys.settrace(trace)

    slay_dir = out_dir / "slay"
    slay_dir.mkdir(parents=True, exist_ok=True)

    slay_plot_dir = plot_dir / "slay_merges"
    slay_plot_dir.mkdir(parents=True, exist_ok=True)

    slay_model_path = slay_dir / "slay_model.pt"

    try:
        merges, sorting_analyzer, slay_metrics = compute_slay_merges(slay_analyzer, model_path=str(slay_model_path))
    finally:
        sys.settrace(None)

    spike_latents = captured.get("spike_latents")
    spike_labels = captured.get("spike_labels")

    if not merges:
        print("No merges found by SLAy.")
        return merges, slay_metrics

    if spike_latents is None or spike_labels is None:
        print("Could not capture spike_latents or spike_labels from compute_slay_merges. Skipping SLAy graphing.")
        return merges, slay_metrics

    unit_latents = {unit_id: spike_latents[spike_labels == unit_id] for unit_id in good_units}

    try:
        correlograms_ext = slay_analyzer.get_extension("correlograms")
        correlograms = correlograms_ext.get_data()
    except Exception as e:
        print("Could not retrieve correlograms extension. Skipping SLAy graphing:", e)
        return merges, slay_metrics

    bin_ms = correlograms_ext.params["bin_ms"]
    window_ms = correlograms_ext.params["window_ms"]
    n_bins = correlograms.shape[-1]
    lags = np.arange(n_bins) * bin_ms - window_ms / 2 + bin_ms / 2

    unit_to_index = {unit_id: i for i, unit_id in enumerate(good_units)}

    print(f"Generating SLAy comparison graphs for {len(merges)} merges...")

    for unit_a, unit_b in merges:
        try:
            idx_a = unit_to_index[unit_a]
            idx_b = unit_to_index[unit_b]
            latents_a = unit_latents[unit_a]
            latents_b = unit_latents[unit_b]
            combined_latents = np.vstack([latents_a, latents_b])
            labels = np.concatenate([np.zeros(len(latents_a), dtype=int), np.ones(len(latents_b), dtype=int)])

            latent_2d = PCA(n_components=2).fit_transform(combined_latents)

            slay_graphs(unit_a, unit_b, latent_2d, labels, correlograms, idx_a, idx_b, lags, slay_metrics, slay_plot_dir)
        except Exception as e:
            print(f"Error generating SLAy graph for units {unit_a} and {unit_b}: {e}")
            print(traceback.format_exc())
            print("Skipping graphing this pair and continuing with the next merge.")

    return merges, slay_metrics