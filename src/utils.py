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
from contextlib import contextmanager

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import spikeinterface.full as si
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.exporters import export_to_phy
from probeinterface.plotting import plot_probe

import bombcell as bc
import bombcell.loading_utils as loading_utils
import bombcell.helper_functions as helper_functions

from slay import compute_slay_merges

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.ioff()

@contextmanager
def timed(timing_dict, key):
    """Add elapsed time to timing_dict[key], even if the block raises.
    Does not catch the exception."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing_dict[key] = timing_dict.get(key, 0.0) + (time.perf_counter() - t0)

from pathlib import Path
import zarr
import spikeinterface.full as si


import numpy as np
import spikeinterface.full as si

# Not used
def recording_summary(rec):
    fs = rec.get_sampling_frequency()
    n_seg = rec.get_num_segments()
    n = sum(rec.get_num_samples(segment_index=i) for i in range(n_seg))
    size_gb = n * rec.get_num_channels() * rec.get_dtype().itemsize / 1e9
    print(f"\033[4mRecorder Summary\033[0m")
    print(f"channels: {rec.get_num_channels()} | segments: {n_seg} | dtype: {rec.get_dtype()}")
    print(f"fs: {fs:.0f} Hz | samples: {n:,} | duration: {n / fs / 60:.2f} min")
    print(f"size: {size_gb:.2f} GB")

def _walk(obj):
    """Yield every dict nested anywhere inside obj (including obj itself)."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)


def _patch_recording_dict(rec_dict, dtype=None):
    """Clean up the serialized recording in place. Returns True if anything changed."""
    changed = False
    for d in _walk(rec_dict):
        if "load_sync_channel" in d:
            del d["load_sync_channel"]
            changed = True
        if dtype and "BinaryRecordingExtractor" in str(d.get("class", "")):
            kwargs = d.setdefault("kwargs", {})
            if kwargs.get("dtype") != dtype:
                kwargs["dtype"] = dtype
                changed = True
    return changed


def _stored_paths(rec_dict):
    """Path(s) stored in the recording metadata (only used for error messages)."""
    for d in _walk(rec_dict):
        for key in ("file_paths", "folder_path", "file_path"):
            if key in d:
                v = d[key]
                return v if isinstance(v, list) else [v]
    return []


def _find_meta(raw_file):
    """SpikeGLX: foo.ap.bin -> foo.ap.meta. Otherwise a lone .meta in the same folder, else None."""
    same_stem = raw_file.with_suffix(".meta")
    if same_stem.exists():
        return same_stem
    others = list(raw_file.parent.glob("*.meta"))
    return others[0] if len(others) == 1 else None


def _locate_raw_file(rec, data_path):
    """
    1. The file behind the loaded recording (if it's a binary recording).
    2. data_path if it's a file.
    3. A single .bin/.dat in data_path or its child folders (SpikeGLX layout).
    """
    if rec is not None:
        try:
            p = Path(rec.get_binary_description()["file_paths"][0])
            if p.is_file():
                return p
        except Exception:
            pass

    if data_path is None:
        raise FileNotFoundError("No recording path available and the recording is not a binary file.")
    data_path = Path(data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist:\n{data_path}")
    if data_path.is_file():
        return data_path

    search_dirs = [data_path, *(p for p in data_path.iterdir() if p.is_dir())]
    candidates = [f for d in search_dirs for pat in ("*.bin", "*.dat") for f in d.glob(pat)]

    # SpikeGLX folders often hold both .ap.bin and .lf.bin: prefer the AP band
    ap = [c for c in candidates if c.name.endswith(".ap.bin")]
    if len(candidates) > 1 and len(ap) == 1:
        candidates = ap

    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .bin/.dat file under:\n{data_path}\n"
            f"Found {len(candidates)}: {[str(c) for c in candidates]}"
        )
    return candidates[0]


def import_analyzer(analyzer_path, VERBOSE_INVENTORY=False, describe=True, *,
                    data_path=None, dtype=None):
    """
    Load a SpikeInterface analyzer from a zarr folder.

    dtype: force the dtype of any BinaryRecordingExtractor stored in the analyzer
           (e.g. "float32" for Kilosort's recording.dat). None leaves it alone.
    Returns (analyzer, rec, raw_file, meta_file). meta_file is None if no .meta exists.
    """
    analyzer_path = Path(analyzer_path)
    print("Importing:", analyzer_path)

    root = zarr.open(analyzer_path, mode="a")
    rec_dict = root["recording"][0]

    try:
        if _patch_recording_dict(rec_dict, dtype):
            root["recording"][0] = rec_dict   # only write back when something changed
    except Exception as e:
        print(f"Could not patch recording metadata: {e}")

    analyzer = si.load_sorting_analyzer(analyzer_path)

    rec = analyzer.recording
    if rec is None and analyzer.has_temporary_recording():
        rec = analyzer.get_temporary_recording()
    if rec is None:
        stored = [(analyzer_path / p).resolve() for p in _stored_paths(rec_dict)]
        raise RuntimeError(
            f"No recording attached to analyzer: {analyzer_path}.\n"
            f"Make sure the recording is available at: {stored}\n"
            f"Or attach a temporary recording before running this script."
        )

    raw_file = _locate_raw_file(rec, data_path)
    meta_file = _find_meta(raw_file)

    print()
    if describe:
        analyzer_info(analyzer, analyzer_path, VERBOSE_INVENTORY)
        print()

    return analyzer, rec, raw_file, meta_file

def analyzer_info(analyzer, analyzer_path, VERBOSE_INVENTORY):
    print("\033[4mAnalyzer Summary\033[0m")
    summarize_analyzer(analyzer)
    print_waveform_window(analyzer)
    
    if VERBOSE_INVENTORY:
        print_analyzer_inventory(analyzer, title=f"Inventory: {analyzer_path.name}")

def summarize_analyzer(analyzer):
    print("Recording attached:", analyzer.has_recording())
    print("Temporary recording attached:", analyzer.has_temporary_recording())
    print("Number of units:", analyzer.sorting.get_num_units())
    print("Sampling frequency (Hz):", analyzer.sorting.get_sampling_frequency())
    

    if analyzer.has_recording() or analyzer.has_temporary_recording():
        rec = analyzer.recording or analyzer.get_temporary_recording()
        print("Recording object:", rec)
        if hasattr(rec, "file_path"):
            print("Recording file path:", rec.file_path)

def print_waveform_window(analyzer):
    # Optional: show waveform window length if waveforms extension exists
    if analyzer.has_extension("waveforms"):
        wf = analyzer.get_extension("waveforms")
        print("Waveform window (ms):", wf.params.get("ms_before"), "before,", wf.params.get("ms_after"), "after")
    else:
        print("No 'waveforms' extension found in this analyzer.")
        
    
def print_analyzer_inventory(analyzer, title="Analyzer inventory"):
    """Print extensions on the analyzer and unit property keys on the sorting (for Phy additional_properties)."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    try:
        ext = list(analyzer.get_extension_names())
    except Exception:
        ext = []
    print("Extensions (loaded):", ext if ext else "None")

    try:
        saved = list(analyzer.get_saved_extension_names())
        if saved:
            print("Extensions (saved on disk):", saved)
    except Exception:
        pass

    sorting = analyzer.sorting
    prop_keys = []
    try:
        if hasattr(sorting, "get_property_keys"):
            prop_keys = list(sorting.get_property_keys())
        elif hasattr(sorting, "property_keys"):
            prop_keys = list(sorting.property_keys)
    except Exception:
        prop_keys = []
    print("Sorting unit property keys:", prop_keys if prop_keys else "None")

    for k in prop_keys[:8]:
        try:
            v = sorting.get_property(k)
            sh = getattr(v, "shape", None)
            dt = getattr(v, "dtype", None)
            print(f"  [{k}] len/shape={sh if sh is not None else len(v)} dtype={dt}")
        except Exception as e:
            print(f"  [{k}] (read error: {e})")
    if len(prop_keys) > 8:
        print(f"  ... and {len(prop_keys) - 8} more keys")

    if analyzer.has_extension("quality_metrics"):
        try:
            qm = analyzer.get_extension("quality_metrics")
            df = qm.get_data()
            print("quality_metrics DataFrame columns:", list(getattr(df, "columns", [])))
        except Exception as e:
            print("quality_metrics: could not read:", e)

    if analyzer.has_extension("template_metrics"):
        try:
            tm = analyzer.get_extension("template_metrics")
            df = tm.get_data()
            print("template_metrics DataFrame columns:", list(getattr(df, "columns", [])))
        except Exception as e:
            print("template_metrics: could not read:", e)

    try:
        dfm = analyzer.get_metrics_extension_data()
        if dfm is not None and hasattr(dfm, "columns") and len(dfm.columns):
            print("get_metrics_extension_data() columns:", list(dfm.columns))
    except Exception:
        pass

    print("=" * 60 + "\n")

# not used
def insert_example_sorting_property(analyzer, key="phy_note", text="from_notebook"):
    """Example: add a string property for every unit (in memory; zarr may persist if save=True)."""
    uids = list(analyzer.sorting.get_unit_ids())
    n = len(uids)
    values = np.array([text] * n, dtype=object)
    analyzer.set_sorting_property(key, values, save=True)
    print(f"Inserted sorting property {key!r} for {n} units.")
    
