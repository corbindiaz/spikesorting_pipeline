import io
import re
import sys
import json
import textwrap
import traceback
import time
from datetime import datetime
import os
import warnings
import argparse
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

from preprocess import preprocess
from sorter_kilosort import sorter_kilosort
from postprocess import postprocess
from export_phy import export_phy
from export_bombcell import export_bombcell
from curation import curation
from visualization import save_process_timing_graph

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

PARAMS_PATH = Path(__file__).resolve().parent / 'params.json'

with open(PARAMS_PATH) as f:
    params = json.load(f)

STEPS = {
    "preprocess": preprocess,
    "kilosort": sorter_kilosort,
    "postprocess": postprocess,
    "phy": export_phy,
    "bombcell": export_bombcell,
    "curation": curation,
}

DEFAULT_STEPS = [
    "preprocess",
    "kilosort",
    "postprocess",
    "export",
    "curation",
]

SUB_STEPS = [
    "preprocess",
    "kilosort",
    "postprocess",
    "phy",
    "bombcell",
    "curation",
]


def main():
    log_buffer = io.StringIO()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_buffer)
    sys.stderr = Tee(orig_stderr, log_buffer)
    
    parser = argparse.ArgumentParser(
        description="Run the ephys processing pipeline."
    )

    parser.add_argument(
        "--recording",
        type=Path,
        help="Path to the input recording.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Root output folder for the pipeline.",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "preprocess",
            "kilosort",
            "postprocess",
            "phy",
            "bombcell",
            "export",
            "curation",
            "rest"
        ],
        default=DEFAULT_STEPS,
        help=(
            "Steps to run, in the order provided. "
            "'export' runs both phy and bombcell."
        ),
    )

    width = 80

    print("=" * width)
    print("=" * width)
    print("WELCOME TO SPIKESORTING PIPELINE.".center(width))
    print("=" * width)
    print("=" * width)

    args = parser.parse_args()

    if not args.steps:
        parser.error("At least one pipeline step must be specified.")
        
    first_step = args.steps[0]
    args.steps = expand_rest(args.steps, parser)
    print(f'Steps to run: {args.steps}' )
    preprocess_folder = args.output / "preprocessed"

    if first_step == "preprocess":
        if args.recording is None:
            parser.error(
                "--recording is required when starting with preprocess."
            )
        else:
            print(f"Recording input given. Using recording at {args.recording} for preprocessing.")
    elif first_step == "kilosort":
        if not preprocess_folder.exists():
            parser.error(
                f"--Preprocessed recording must exist as {preprocess_folder} to run sorting."
            )   
        else:
            print(f"Preprocessed recording found. Using {preprocess_folder} for sorting.")

    analyzer_folder = args.output / "analyzer"

    if first_step not in ("preprocess", "kilosort"):
        if not analyzer_folder.exists():
            parser.error(
                f"Analyzer must exist at {analyzer_folder} to run step after sorting."
            )
        else:
            print(f"Analyzer found. Using {analyzer_folder} for {first_step} step.")

    export_dir = args.output / "export"
    phy_dirs = list(export_dir.glob("*/phy"))

    if "bombcell" in args.steps and not phy_dirs:
        parser.error(
            f"Phy output must exist under {export_dir}/*/phy before running bombcell."
        )

    plots = args.output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    time_master = {}

    print()
    print("Inputs valid. All systems GO!")
    print()

    failed = False
    try:
        for i, step_name in enumerate(args.steps):
            print("=" * width)
            print(f"RUNNING STEP: {step_name}".center(width))
            print("=" * width)

            if step_name == "export":
                step_function = run_export
            else:
                step_function = STEPS[step_name]

            os.environ["PYTHONWARNINGS"] = (
                "ignore:The provided margin_ms:UserWarning:spikeinterface.preprocessing.filter"
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The provided margin_ms .* is smaller than the recommended margin.*",
                category=UserWarning,
                module=r"spikeinterface\.preprocessing\.filter",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Trying to unpickle estimator IncrementalPCA from version.*",
                category=InconsistentVersionWarning,
            )

            step_start = time.perf_counter()
            try:
                result = step_function(
                    recording_path_=args.recording,
                    output_folder=args.output,
                    params=params,
                    time_master=time_master,
                    step=i,
                )
                if result is not None:
                    time_master = result
            except BaseException:
                failed = True
                print()
                print(f"STEP FAILED: {step_name} "
                      f"(after {time.perf_counter() - step_start:.2f} seconds)")
                raise
            finally:
                plt.close("all")

            step_elapsed = time.perf_counter() - step_start
            print()
            print(f"COMPLETED STEP: {step_name}")
            print(f"TOTAL COMPUTATION TIME: {step_elapsed:.2f} seconds")
        
        print()
        print("=" * width)
        print("=" * width)
        print("PIPELINE COMPLETE !".center(width))
        print("=" * width)
        print("=" * width)

    except BaseException:
        # Python prints the traceback only after `finally`, so record it now.
        # Written to the buffer only, so it isn't duplicated on the terminal.
        log_buffer.write(traceback.format_exc())
        raise

    finally:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_FAILED" if failed else ""
        
        if params['diagnostic']['print_time_computation_graph']:
            plot_path = args.output / f"computation_time_{timestamp}{suffix}.png"
            try:
                save_process_timing_graph(time_master, plot_path)
            except Exception as plot_err:
                print(f"Could not save timing graph: {plot_err}")

        # Save the PDF last so it includes everything above.
        try:
            if params['diagnostic']['print_terminal_pdf']:
                save_terminal_pdf(
                    log_buffer.getvalue(),
                    args.output / f"terminal_output_{timestamp}{suffix}.pdf",
                )
        except Exception as pdf_err:
            print(f"Could not save terminal PDF: {pdf_err}", file=orig_stderr)
        finally:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr

def expand_rest(steps, parser, SUB_STEPS = SUB_STEPS):
    """Expand a trailing 'rest' entry in --steps into the remaining steps.
 
    '<step_name> rest' means: start at <step_name> and run every step after
    it, in pipeline order. Anything listed before <step_name> is preserved
    as-is. 'rest' must be the last entry in the list.
    """
    if "rest" not in steps:
        return steps
 
    rest_idx = steps.index("rest")
 
    if rest_idx != len(steps) - 1:
        parser.error("'rest' must be the last entry in --steps.")
 
    if rest_idx == 0:
        parser.error(
            "'rest' must follow a step name indicating where to start "
            "(e.g. '--steps kilosort rest')."
        )
 
    start_step = steps[rest_idx - 1]
 
    order = SUB_STEPS if start_step in ("phy", "bombcell") else DEFAULT_STEPS
 
    if start_step not in order:
        parser.error(f"Cannot determine step order for '{start_step}'.")
 
    start_pos = order.index(start_step)
 
    # Keep everything listed before start_step untouched, then append
    # start_step and everything after it in the chosen order.
    expanded = steps[:rest_idx - 1] + order[start_pos:]
 
    return expanded

def run_export(
    recording,
    output_folder,
    params,
    time_master,
    step,
):
    time_master = export_phy(
        recording=recording,
        output_folder=output_folder,
        params=params,
        time_master=time_master,
        step=step,
    )

    time_master = export_bombcell(
        recording=recording,
        output_folder=output_folder,
        params=params,
        time_master=time_master,
        step=step+0.5,
    )

    return time_master


class Tee:
    """Mirror everything written to a stream into a shared in-memory buffer."""

    def __init__(self, stream, buffer):
        self._stream = stream
        self._buffer = buffer

    def write(self, data):
        self._stream.write(data)
        self._buffer.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_log(text):
    """Strip ANSI codes and collapse tqdm-style \\r redraws to their final state."""
    lines = []
    for raw in _ANSI.sub("", text).split("\n"):
        segments = [s for s in raw.split("\r") if s.strip()]
        lines.append(segments[-1].expandtabs(4) if segments else "")
    return lines


def save_terminal_pdf(text, pdf_path, wrap=110, lines_per_page=80):
    lines = []
    for line in _clean_log(text):
        lines.extend(
            textwrap.wrap(line, wrap, drop_whitespace=False,
                          replace_whitespace=False) or [""]
        )

    with PdfPages(pdf_path) as pdf:
        for start in range(0, max(len(lines), 1), lines_per_page):
            chunk = "\n".join(lines[start:start + lines_per_page])
            chunk = chunk.replace("$", r"\$")  # avoid matplotlib mathtext
            fig = Figure(figsize=(8.5, 11))
            fig.text(0.05, 0.97, chunk, va="top", ha="left",
                     family="monospace", fontsize=7)
            pdf.savefig(fig)

if __name__ == "__main__":
    main()