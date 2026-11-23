import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

N_SHANKS = 4

df = pd.read_csv('cluster_group.tsv', sep="\t")

# ---- Noise units: report the total, but exclude from all graphs ----
n_noise = int((df['group'] == 'noise').sum())
print(f'Total number of noise units (excluded from graphs): {n_noise}')

# ---- Bar plot of group counts (fixed order + colors) ----
order = ['good', 'mua', 'nonsoma_mua', 'nonsoma_good', 'lowcount']  # noise excluded
fallback_order = ['unsorted', 'split']

colors = {
    'good':         '#3a9d5d',
    'mua':          '#f28c28',
    'nonsoma_mua':  '#0b2a6f',  # dark blue
    'nonsoma_good': '#2f6fbf',  # blue
    'noise':        '#7a3b3b',
    'lowcount':     '#9e9e9e',
    'unsorted':     '#8e5ea2',
    'split':        '#d4a017',
}

# If none of the usual groups are present, use unsorted/split instead
if not df['group'].isin(order + ['noise']).any():
    order = fallback_order

group_counts = df['group'].value_counts().reindex(order, fill_value=0)

# Only keep 'lowcount' if it has at least 1 cluster
if 'lowcount' in group_counts.index and group_counts['lowcount'] < 1:
    group_counts = group_counts.drop('lowcount')

print(group_counts)

plt.figure(figsize=(10, 6))
ax = group_counts.plot(kind='bar', color=[colors[g] for g in group_counts.index])
plt.xlabel('Group')
plt.ylabel('Count')
plt.title('Group Counts')
plt.xticks(rotation=0)
ax.bar_label(ax.containers[0], padding=3)
ax.margins(y=0.1)
plt.show()

# ---- Histogram of spike counts per "good" cluster ----
spike_times = np.load('spike_times.npy').flatten()
spike_clusters = np.load('spike_clusters.npy').flatten()
assert len(spike_times) == len(spike_clusters)

# Number of spikes per cluster
cluster_ids, n_spikes = np.unique(spike_clusters, return_counts=True)
spikes_per_cluster = pd.Series(n_spikes, index=cluster_ids)

good_ids = df.loc[df['group'] == 'good', 'cluster_id']
good_spike_counts = spikes_per_cluster.reindex(good_ids).dropna()

if len(good_spike_counts) > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(good_spike_counts, bins=30, color=colors['good'], edgecolor='white')
    plt.xlabel('Number of spikes in cluster')
    plt.ylabel('Number of "good" clusters')
    plt.title(f'Spike Counts per Good Cluster (n = {len(good_spike_counts)})')
    plt.show()
else:
    print('No "good" clusters found; skipping spike count histogram.')
    

# ---- Firing rates for good clusters, using occupied 1-second bins only ----

# Set sampling rate to the actual spike_times.npy time units.
# For standard Kilosort/Phy data, spike_times are usually sample indices.
SAMPLE_RATE = 30000.0  # Hz -- CHANGE if your recording sampling rate differs

BIN_SIZE = 1.0  # seconds

good_ids = df.loc[df['group'] == 'good', 'cluster_id'].to_numpy()

# Convert spike times from sample indices to seconds
spike_times_sec = spike_times.astype(np.float64) / SAMPLE_RATE

# Recording duration from the latest recorded spike
recording_duration = spike_times_sec.max()

# Store firing rates for each good cluster
good_firing_rates = {}

for cluster_id in good_ids:

    # Extract spike times for this cluster
    cluster_spikes = spike_times_sec[
        spike_clusters == cluster_id
    ]

    if len(cluster_spikes) == 0:
        continue

    # Bin spikes into 1-second bins
    bin_edges = np.arange(
        0,
        recording_duration + BIN_SIZE,
        BIN_SIZE
    )

    spike_counts, _ = np.histogram(
        cluster_spikes,
        bins=bin_edges
    )

    # Keep ONLY bins containing at least one spike
    occupied_bins = spike_counts[spike_counts > 0]

    if len(occupied_bins) == 0:
        continue

    # Average spikes per occupied second
    mean_hz = occupied_bins.mean()

    good_firing_rates[int(cluster_id)] = mean_hz


# Convert results to Series
good_hz = pd.Series(
    good_firing_rates,
    name='mean_hz'
)

print('\n---- Good cluster firing rates ----')
print(f'Good clusters with spikes: {len(good_hz)}')
print(f'Average firing rate: {good_hz.mean():.2f} Hz')
print(f'Median firing rate: {good_hz.median():.2f} Hz')


# ---- Histogram 1: Distribution of firing rates ----
if len(good_hz) > 0:

    plt.figure(figsize=(10, 6))

    plt.hist(
        good_hz.values,
        bins=np.logspace(
            np.log10(good_hz.min()),
            np.log10(good_hz.max()),
            30
        ),
        color=colors['good'],
        edgecolor='white'
    )

    plt.xscale('log')
    plt.xlabel('Average firing rate (Hz, log scale)')
    plt.ylabel('Number of good clusters')
    plt.title(
        f'Firing Rate Distribution of Good Clusters '
        f'(Occupied 1-second bins, n={len(good_hz)})'
    )

    plt.legend()
    plt.tight_layout()
    plt.show()




# ---- Channel helpers ----
def get_total_channels():
    """Total number of channels used in the sort."""
    if os.path.exists('channel_map.npy'):
        return len(np.load('channel_map.npy').flatten())
    if os.path.exists('templates.npy'):
        return np.load('templates.npy', mmap_mode='r').shape[2]
    return None


def get_cluster_channels():
    """Return a Series: cluster_id -> channel, taken from the 'peak' column."""
    # 1) 'peak' column in cluster_group.tsv
    if 'peak' in df.columns:
        return df.set_index('cluster_id')['peak']

    # 2) cluster_peak.tsv, then 3) cluster_info.tsv (Phy merges cluster_*.tsv files)
    for fname in ('cluster_peak.tsv', 'cluster_info.tsv'):
        if os.path.exists(fname):
            info = pd.read_csv(fname, sep='\t')
            if 'peak' in info.columns:
                return info.set_index('cluster_id')['peak']

    raise FileNotFoundError(
        "No 'peak' column found in cluster_group.tsv, cluster_peak.tsv or cluster_info.tsv"
    )


def get_channel_layout(n_shanks):
    """DataFrame with one row per channel: channel, x, y, shank (0-based)."""
    if not os.path.exists('channel_positions.npy'):
        return None
    pos = np.load('channel_positions.npy')
    if os.path.exists('channel_map.npy'):
        chan_map = np.load('channel_map.npy').flatten().astype(int)
    else:
        chan_map = np.arange(len(pos))
    layout = pd.DataFrame({'channel': chan_map, 'x': pos[:, 0], 'y': pos[:, 1]})

    # Assign shanks by splitting the sorted distinct x values at the largest gaps
    ux = np.sort(layout['x'].unique())
    if n_shanks > 1 and len(ux) > n_shanks:
        gaps = np.diff(ux)
        cut = np.sort(np.argsort(gaps)[-(n_shanks - 1):])
        edges = (ux[cut] + ux[cut + 1]) / 2
    elif n_shanks > 1:
        edges = (ux[:-1] + ux[1:]) / 2
    else:
        edges = np.array([])
    layout['shank'] = np.searchsorted(edges, layout['x'])
    return layout


# ---- Distinct channels with at least one 'unsorted' / 'split' cluster ----
total_channels = get_total_channels()
print(f'Total number of channels: {total_channels}')

cluster_channels = get_cluster_channels()

df_ch = df.copy()
df_ch['channel'] = df_ch['cluster_id'].map(cluster_channels)
df_ch.loc[df_ch['channel'] < 0, 'channel'] = np.nan  # peak = -1 means no usable spikes

n_missing = df_ch['channel'].isna().sum()
if n_missing:
    print(f'Warning: {n_missing} clusters have no channel info and were skipped.')

channels_per_group = {}
for grp in ['unsorted', 'split']:
    n_channels = df_ch.loc[df_ch['group'] == grp, 'channel'].dropna().nunique()
    channels_per_group[grp] = n_channels
    print(f'Distinct channels with at least one "{grp}" cluster: {n_channels}')

# ---- Merging progress: (1 - split channels / total channels) ----
n_split_channels = channels_per_group['split']
if total_channels:
    frac_merged = 1 - n_split_channels / total_channels
    print(f'Channels with at least one "split" cluster: {n_split_channels} / {total_channels}')
    print(f'Percentage done merging: {100 * frac_merged:.1f}%')
else:
    print('Percentage done merging: n/a (could not determine total channels)')

# ---- Stacked bar charts: units per channel, one panel per shank (noise excluded) ----
stack_colors = {**colors, 'unsorted': 'black', 'split': 'grey'}
stack_order = ['good', 'mua', 'nonsoma_mua', 'nonsoma_good', 'lowcount',
               'unsorted', 'split']

df_plot = df_ch.dropna(subset=['channel']).copy()
df_plot = df_plot[df_plot['group'] != 'noise']  # exclude noise from the graph
df_plot['channel'] = df_plot['channel'].astype(int)

layout = get_channel_layout(N_SHANKS)

if layout is None:
    print('channel_positions.npy not found; skipping per-shank units-per-channel chart.')
elif len(df_plot) == 0:
    print('No clusters with channel info; skipping units-per-channel chart.')
else:
    n_unplaced = int((~df_plot['channel'].isin(layout['channel'])).sum())
    if n_unplaced:
        print(f'Warning: {n_unplaced} clusters peak on channels missing from the '
              'channel map and were skipped in the shank chart.')
        df_plot = df_plot[df_plot['channel'].isin(layout['channel'])]

    counts = pd.crosstab(df_plot['channel'], df_plot['group'])

    # Fixed stacking order; groups not in the list go on top, in a default color
    present = [g for g in stack_order if g in counts.columns]
    extra = [g for g in counts.columns if g not in stack_order]
    counts = counts[present + extra]

    # One row per channel on the probe (zero where no units), with its position
    counts = counts.reindex(layout['channel'], fill_value=0)
    counts.index.name = 'channel'
    layout = layout.set_index('channel')

    # Geometry: row pitch along the shank, and bar height for staggered columns
    uy = np.sort(layout['y'].unique())
    row_pitch = np.diff(uy).min() if len(uy) > 1 else 1.0
    max_cols = max(layout.loc[layout['shank'] == s, 'x'].nunique()
                   for s in sorted(layout['shank'].unique()))
    bar_h = 0.9 * row_pitch / max_cols

    shanks = sorted(layout['shank'].unique())
    stacked_max = counts.sum(axis=1).max()

    fig, axes = plt.subplots(1, len(shanks), figsize=(3.8 * len(shanks), 10),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax, s in zip(axes, shanks):
        sub = layout[layout['shank'] == s]
        xs = np.sort(sub['x'].unique())
        n_cols = len(xs)
        col_idx = np.searchsorted(xs, sub['x'].values)
        # stagger columns vertically within a row so each channel has its own bar
        y_pos = sub['y'].values + (col_idx - (n_cols - 1) / 2) * (row_pitch / max_cols)

        left = np.zeros(len(sub))
        for grp in counts.columns:
            vals = counts.loc[sub.index, grp].values
            ax.barh(y_pos, vals, left=left, height=bar_h,
                    color=stack_colors.get(grp, '#cccccc'))
            left += vals

        ax.set_title(f'Shank {s + 1}\n({int(left.sum())} units)')
        ax.set_xlabel('Number of units')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, stacked_max * 1.05)

    axes[0].set_ylabel('Depth along shank (µm)')
    handles = [Patch(color=stack_colors.get(g, '#cccccc'), label=g) for g in counts.columns]
    fig.legend(handles=handles, title='Group', loc='upper right')
    fig.suptitle(f'Units per Channel by Shank '
                 f'(total units = {int(counts.values.sum())}, noise excluded)')
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    plt.show()