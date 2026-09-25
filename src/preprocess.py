import shutil
import time

import numpy as np
import pandas as pd

from utils import timed, recording_summary
from visualization import plot_peak_localization

import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.motion import interpolate_motion

import matplotlib.pyplot as plt

def preprocess(recording_path_, output_folder, params, time_master, step=0):
    params_pre = params['preprocess']
    preprocessed_folder = output_folder / "preprocessed"
    preprocessed_folder.mkdir(parents=True, exist_ok=True)
    
    plots = output_folder / "plots"
    motion_plots_folder = plots / "motion"
    plots.mkdir(parents=True, exist_ok=True)
    motion_plots_folder.mkdir(parents=True, exist_ok=True)

    recording = si.read_spikeglx(recording_path_, stream_id="imec0.ap")
    print(recording)
    
    # Time Shift
    print("Shifting time to ensure 0sec start...")
    print(f'Old start time: {recording.get_times()[0]} seconds')
    recording.shift_times(-recording.get_times()[0])
    print(f'New start time: {recording.get_times()[0]} seconds')
    
    # Trimming
    params_trim = params_pre['trimming']
    if params_trim['trim_recording']:
        if params_trim['start'] is None:
            start = recording.get_times()[0]
        else:
            start = params_trim['start']
        if params_trim['end'] is None:
            end = recording.get_times()[-1]
        else:
            end = params_trim['end']
        print(f"Trimming recording to be from {start} sec to {end} sec...")
        recording = recording.time_slice(start_time=start, 
                                         end_time=end)
    
    if params_pre['split_by_shank']:
        groups = recording.split_by("group")
    else:
        groups = {'wholeprobe':recording}

    for i, (group_name, group) in enumerate(groups.items()):
        group_name = str(group_name)
        recording_name = f"{group_name}_group{i}"

        # Register timing entries first so partial results survive a crash
        group_time = time_master.setdefault(f"Group{i}", {})
        preprocessing_time = group_time.setdefault("Preprocessing", {})
        group_start = time.perf_counter()

        preprocessed_path = preprocessed_folder / recording_name
        motion_folder = preprocessed_folder / "motion" / recording_name

        print("-" * 80)
        print(f"Preprocessing shank {group_name}")
        print(f"Channels: {group.get_num_channels()}")

        try:
            # Cleaning (Phase-shift, Filtering, Bad Channel Detection, CAR)
            with timed(preprocessing_time, "Cleaning"):
                print("Phase-shift correcting...")
                group = spre.phase_shift(group)

                params_band = params_pre['bandpass_filter']
                print("Bandpass filtering...")
                group = spre.bandpass_filter(
                    group,
                    freq_min=params_band['freq_min'],
                    freq_max=params_band['freq_max'],
                )

                params_bad = params_pre['bad_channels']
                print("Detecting bad channels...")
                bad_channel_ids, channel_labels = spre.detect_bad_channels(
                    group,
                    method=params_bad['detect_bad_channels_method'],
                    n_neighbors=params_bad['coh_psd_nneighbors'],
                    psd_hf_threshold=params_bad['coh_psd_hf_threshold'],
                    dead_channel_threshold=params_bad['coh_psd_dead_threshold'],
                    noisy_channel_threshold=params_bad['coh_psd_noise_threshold'],
                    outside_channel_threshold=params_bad['coh_psd_out_threshold'],
                    nyquist_threshold=params_bad['coh_psd_nyquist'],
                )
                print(f"Detected {len(bad_channel_ids)} bad channels...")

                channel_label_csv = pd.Series(channel_labels, index=np.arange(len(channel_labels)))
                channel_label_path = preprocessed_folder / f"{group_name}_channel_labels.csv"
                channel_label_csv.to_csv(channel_label_path, header=["spikeinterface_label"], index_label="channel_id")
                print(f"Channel labels saved to: {channel_label_path}\n")

                if params_bad['remove_bad_channels']:
                    if len(bad_channel_ids) > 0:
                        frac = len(bad_channel_ids) / group.get_num_channels()
                        if frac > params_pre['bad_channel_limit']:
                            raise RuntimeError(
                                f"Too many bad channels detected: "
                                f"{len(bad_channel_ids)}/{group.get_num_channels()} "
                                f"({frac:.1%}), exceeding limit of "
                                f"{params_pre['bad_channel_limit']:.1%}. "
                                f"Please increase 'bad_channel_limit' to override this message, "
                                f"or set 'remove_bad_channels' to False."
                            )
                        print("Removing bad channels...")
                        group = group.remove_channels(bad_channel_ids)
                    else:
                        print("No bad channels to remove.")

                params_car = params_pre['car']
                print("Applying common median reference...")
                group = spre.common_reference(
                    group,
                    reference=params_car['car_reference'],
                    operator=params_car['car_operator'],
                )

            print(f"Cleaning time {recording_name}: {preprocessing_time['Cleaning']:.2f} seconds\n")

            # Motion
            params_motion = params_pre["motion"]
            if params_motion['motion_overwrite'] and motion_folder.exists():
                shutil.rmtree(motion_folder)
                
            probe = recording.get_probe()
            pos = np.asarray(probe.contact_positions)

            if pos.shape[0] != recording.get_num_channels():
                raise ValueError("contact_positions vs n_channels mismatch")

            y = pos[:, 1].astype(float)
            dptp = float(np.ptp(y))

            y_u = np.sort(np.unique(y))
            dy = np.diff(y_u)
            dy = dy[dy > params_motion["min_pitch_difference_um"]]

            if len(dy) < 3:
                pitch_um = dptp / max(1, len(y_u) - 1)
            else:
                pitch_um = float(np.median(dy))

            win_scale_um = float(
                np.clip(
                    params_motion["n_rows_in_scale"] * pitch_um,
                    params_motion["min_win_scale_um"],
                    params_motion["max_win_scale_fraction"] * dptp
                )
            )

            win_step_um = float(
                np.clip(
                    params_motion["step_over_scale"] * win_scale_um,
                    max(
                        params_motion["min_win_step_um"],
                        params_motion["min_win_step_pitch_fraction"] * pitch_um
                    ),
                    params_motion["max_win_step_fraction"] * dptp
                )
            )

            if win_step_um >= params_motion["max_step_scale_fraction"] * win_scale_um:
                win_step_um = float(
                    params_motion["fallback_step_over_scale"] * win_scale_um
                )

            estimate_motion_kwargs = {}
            estimate_motion_kwargs["win_step_um"] = win_step_um
            estimate_motion_kwargs["win_scale_um"] = win_scale_um

            print(f"Probe span: {dptp:.2f} µm")
            print(f"Electrode pitch: {pitch_um:.2f} µm")
            print(f"Motion window: {win_scale_um:.2f} µm")
            print(f"Motion window step: {win_step_um:.2f} µm")

            with timed(preprocessing_time, "Motion"):
                try:
                    print("Computing DREDge-fast motion...")
                    motion_output = spre.compute_motion(
                        group,
                        preset="dredge_fast",
                        folder=motion_folder,
                        raise_error=False,
                        output_motion_info = True,
                        estimate_motion_kwargs= estimate_motion_kwargs,
                        **params['job_kwargs'],
                    )

                    motion, motion_info = None, None
                    if isinstance(motion_output, tuple):
                        motion, motion_info = motion_output
                    elif motion_output is not None:
                        motion = motion_output

                    if motion is not None:
                        print("Applying motion correction...")
                        group = interpolate_motion(group.astype("float32"), motion=motion)
                    else:
                        print("Motion computation failed; saving uncorrected recording.")

                    if motion_info is not None:
                        print("Saving motion plot...")
                        try:
                            motion_plot_path = motion_plots_folder / f"{recording_name}_motion_correction.png"
                            probe_plot_path = motion_plots_folder / f"{recording_name}_Peak_Locations.png"
                            locations = group.get_channel_locations()
                            depth_min = np.min(locations[:, 1])
                            depth_max = np.max(locations[:, 1])*1.1
                            fig = plt.figure(figsize=(14, 8))
                            si.plot_motion_info(
                                motion_info, group,
                                figure=fig,
                                depth_lim=(depth_min, depth_max),
                                color_amplitude=True,
                                amplitude_cmap="inferno",
                                scatter_decimate=10,
                            )
                            fig.savefig(motion_plot_path, dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            print(f"Motion plot saved to: {motion_plot_path}")
                            
                            plot_peak_localization(recording, motion_info, probe_plot_path)
                            print(f"Peak locations plot saved to: {probe_plot_path}")
                        except Exception as e:
                            print(f"Error saving motion plots for {recording_name}: {e}")
                except Exception as e:
                    print(f"Motion correction for {recording_name} failed: {e}")

            print(f"Motion time {recording_name}: {preprocessing_time['Motion']:.2f} seconds\n")

            # Save
            with timed(preprocessing_time, "Save"):
                print("Saving preprocessed recording...")
                group.save(folder=preprocessed_path, overwrite=True, **params['job_kwargs'])

        finally:
            total = time.perf_counter() - group_start
            accounted = sum(v for k, v in preprocessing_time.items() if k != "Other")
            #preprocessing_time["Other"] = max(0.0, total - accounted)

    return time_master