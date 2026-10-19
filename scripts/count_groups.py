import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('cluster_group.tsv', sep="\t")

# ---- Bar plot of group counts (fixed order + colors) ----
order = ['good', 'mua', 'nonsoma', 'noise', 'lowcount']
colors = {
    'good':     '#3a9d5d',
    'mua':      '#f28c28',
    'nonsoma':  '#2f6fbf',
    'noise':    '#7a3b3b',
    'lowcount': '#9e9e9e',
}

group_counts = df['group'].value_counts().reindex(order, fill_value=0)

# Only keep 'lowcount' if it has at least 1 cluster
if group_counts['lowcount'] < 1:
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

# Histogram of spike counts per "good" cluster
spike_times = np.load('spike_times.npy').flatten()
spike_clusters = np.load('spike_clusters.npy').flatten()
assert len(spike_times) == len(spike_clusters)

# Number of spikes per cluster
cluster_ids, n_spikes = np.unique(spike_clusters, return_counts=True)
spikes_per_cluster = pd.Series(n_spikes, index=cluster_ids)

good_ids = df.loc[df['group'] == 'good', 'cluster_id']
good_spike_counts = spikes_per_cluster.reindex(good_ids).dropna()

plt.figure(figsize=(10, 6))
plt.hist(good_spike_counts, bins=30, color=colors['good'], edgecolor='white')
plt.xlabel('Number of spikes in cluster')
plt.ylabel('Number of "good" clusters')
plt.title(f'Spike Counts per Good Cluster (n = {len(good_spike_counts)})')
plt.show()