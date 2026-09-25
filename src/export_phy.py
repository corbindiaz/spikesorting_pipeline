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

from spikeinterface.exporters import export_to_phy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from utils import (import_analyzer, timed)

def export_phy(
    recording_path_,
    output_folder,
    params,
    time_master,
    step=3
):
    if step == 0:
        VERBOSE_INVENTORY = params['verbose_inventory']
    else:
        VERBOSE_INVENTORY = False

    params_phy = params['export']['phy']
    TEST_RUN = params_phy['test_phy']

    analyzer_folder = output_folder / "analyzer"

    plots = output_folder / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    group_analyzer_paths = sorted(analyzer_folder.glob("*group*.zarr"))

    # Phy output location
    export = output_folder / "export"
    export.mkdir(parents=True, exist_ok=True)

    print()
    print("EXPORTING PHY CONTENTS")
    print()

    qm_column_order = params_phy['qm_column_order']
    tm_column_order = params_phy['tm_column_order']
    export_quality_metrics = params_phy['export_quality_metrics']
    export_template_metrics = params_phy['export_template_metrics']

    describe = (step == 0)

    for i, analyzer_path in enumerate(group_analyzer_paths):
        # Register timing entries first so partial results survive a crash
        group_time = time_master.setdefault(f"Group{i}", {})
        export_times = group_time.setdefault("Export", {})
        group_start = time.perf_counter()

        print("-" * 80)

        group_name = analyzer_path.stem  # e.g. "block0_imec0.ap_recording1_group0"
        export_group = export / group_name
        export_group.mkdir(parents=True, exist_ok=True)
        phy = export_group / "phy"

        try:
            # ---- Import (errors propagate) ----
            with timed(export_times, "Phy Import"):
                analyzer, rec, raw_file, meta_file = import_analyzer(
                    analyzer_path, VERBOSE_INVENTORY, describe,
                    dtype="float32",
                )

            extra_props = analyzer.sorting.get_property_keys()

            # ---- Phy export (errors propagate) ----
            with timed(export_times, "Phy Export"):
                if not TEST_RUN:
                    export_to_phy_props_first_then_metrics(
                        sorting_analyzer=analyzer,
                        output_folder=phy,
                        additional_properties=extra_props,
                        qm_column_order=qm_column_order,
                        tm_column_order=tm_column_order,
                        export_quality_metrics=export_quality_metrics,
                        export_template_metrics=export_template_metrics,
                        compute_pc_features=params_phy['compute_pc_features'],
                        compute_amplitudes=params_phy['compute_amplitudes'],
                        copy_binary=params_phy['copy_binary'],
                        use_relative_path=params_phy['use_relative_path'],
                        remove_if_exists=params_phy['remove_if_exists'],
                        progress_bar=params_phy['progress_bar'],
                        verbose=params_phy['verbose_phy'],
                        n_jobs=-1,
                        mp_context="spawn",
                    )
                    print("Exported Phy folder:", phy)
                else:
                    time.sleep(3)

            print(f"Phy export time {group_name}: {export_times['Phy Export']:.2f} seconds")

            # ---- Fix params.py dat_path (errors propagate) ----
            if not TEST_RUN:
                with timed(export_times, "Phy Params Update"):
                    update_phy_params_dat_path(phy)

        finally:
            # Per-group remainder, computed from what was actually recorded
            total = time.perf_counter() - group_start
            accounted = sum(v for k, v in export_times.items() if k != "Other")
            export_times["Phy Other"] = max(0.0, total - accounted)
            print(f"Total time exporting phy {group_name}: {total:.2f} seconds")

    return time_master
        
    
# Phy cluster_*.tsv write order (on disk): 1) extra_props  2) quality_metrics  3) template_metrics
# (export_to_phy is called with QM/TM disabled; then QM TSVs, then TM TSVs.)

def write_phy_qm_tsvs(
    sorting_analyzer,
    output_folder,
    unit_ids,
    *,
    qm_column_order=None,
):
    """Write cluster_<metric>.tsv for quality_metrics only (SpikeInterface-compatible)."""
    output_folder = Path(output_folder)
    n = len(unit_ids)
    cluster_ids = list(range(n))

    if not sorting_analyzer.has_extension("quality_metrics"):
        return

    qm_data = sorting_analyzer.get_extension("quality_metrics").get_data()
    if qm_column_order is not None:
        cols = [c for c in qm_column_order if c in qm_data.columns]
    else:
        cols = [c for c in qm_data.columns if c not in ["num_spikes", "firing_rate"]]
    for column_name in cols:
        metric = pd.DataFrame(
            {"cluster_id": cluster_ids, column_name: qm_data[column_name].values}
        )
        metric.to_csv(output_folder / f"cluster_{column_name}.tsv", sep="\t", index=False)

def write_phy_tm_tsvs(
    sorting_analyzer,
    output_folder,
    unit_ids,
    *,
    tm_column_order=None,
):
    """Write cluster_<metric>.tsv for template_metrics only (SpikeInterface-compatible)."""
    output_folder = Path(output_folder)
    n = len(unit_ids)
    cluster_ids = list(range(n))

    if not sorting_analyzer.has_extension("template_metrics"):
        return

    tm_data = sorting_analyzer.get_extension("template_metrics").get_data()
    if tm_column_order is not None:
        cols = [c for c in tm_column_order if c in tm_data.columns]
    else:
        cols = list(tm_data.columns)
    for column_name in cols:
        metric = pd.DataFrame(
            {"cluster_id": cluster_ids, column_name: tm_data[column_name].values}
        )
        metric.to_csv(output_folder / f"cluster_{column_name}.tsv", sep="\t", index=False)

def export_to_phy_props_first_then_metrics(
    sorting_analyzer,
    output_folder,
    additional_properties,
    qm_column_order=None,
    tm_column_order=None,
    *,
    export_quality_metrics=True,
    export_template_metrics=True,
    **export_kwargs,
):
    """
    Export order: (1) extra_props via export_to_phy, (2) quality_metrics TSVs, (3) template_metrics TSVs.
    Raw SpikeInterface export_to_phy writes QM before additional_properties; we disable QM/TM there
    and append them in this order so Phy sees: extra_props → qm → tm.

    - `additional_properties`: order of sorting-property columns among themselves.
    - `qm_column_order` / `tm_column_order`: if a list, export ONLY those columns; if None, all QM/TM
      (QM still skips num_spikes & firing_rate when exporting all). Use `[]` for no columns of that kind.
    - `export_quality_metrics` / `export_template_metrics`: set False to skip writing those cluster_*.tsv
      blocks entirely. SpikeInterface's `add_quality_metrics` / `add_template_metrics` in kwargs still
      override if passed (for compatibility).
    """
    

    add_qm = export_kwargs.pop("add_quality_metrics", export_quality_metrics)
    add_tm = export_kwargs.pop("add_template_metrics", export_template_metrics)
    export_kwargs["add_quality_metrics"] = False
    export_kwargs["add_template_metrics"] = False

    # (1) extra_props + core Phy files (no QM/TM tsv from SI)
    export_to_phy(
        sorting_analyzer=sorting_analyzer,
        output_folder=output_folder,
        additional_properties=additional_properties,
        **export_kwargs,
    )

    sorting = sorting_analyzer.sorting
    unit_ids = [u for u in sorting.unit_ids if len(sorting.get_unit_spike_train(u)) > 0]

    # (2) quality_metrics
    if add_qm:
        write_phy_qm_tsvs(
            sorting_analyzer,
            output_folder,
            unit_ids,
            qm_column_order=qm_column_order,
        )
    # (3) template_metrics
    if add_tm:
        write_phy_tm_tsvs(
            sorting_analyzer,
            output_folder,
            unit_ids,
            tm_column_order=tm_column_order,
        )
        
def update_phy_params_dat_path(phy):
    params_path = phy / "params.py"

    if not params_path.exists():
        print(f"params.py not found: {params_path}")
        return

    dat_path = (phy / "recording.dat").resolve()

    text = params_path.read_text()

    old = "dat_path = r'recording.dat'"
    new = f"dat_path = r'{dat_path}'"

    if old not in text:
        print(f"Could not find expected dat_path line in {params_path}")
        return

    text = text.replace(old, new)
    params_path.write_text(text)

    print(f"Updated params.py dat_path: {dat_path}")
