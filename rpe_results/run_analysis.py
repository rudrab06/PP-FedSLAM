import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define file names
FILE_ERROR = 'error_array.npy'
FILE_DISTANCE = 'distances_from_start.npy'
FILE_SECONDS = 'seconds_from_start.npy'
FILE_STATS = 'stats.json'
FILE_INFO = 'info.json' 

# --- 1. Load Data ---
try:
    rpe_array = np.load(FILE_ERROR) 
    distances_from_start = np.load(FILE_DISTANCE)
    seconds_from_start = np.load(FILE_SECONDS)
    with open(FILE_STATS, 'r') as f:
        stats = json.load(f)
    with open(FILE_INFO, 'r') as f:
        info = json.load(f)

    # Align array lengths
    min_len = min(len(rpe_array), len(distances_from_start), len(seconds_from_start))
    rpe_array = rpe_array[:min_len]
    distances_from_start = distances_from_start[:min_len]
    seconds_from_start = seconds_from_start[:min_len]
        
    print("All RPE data files loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: One or more files not found. Missing file: {e}")
    exit()

# Create DataFrame for plotting/analysis
df = pd.DataFrame({
    'RPE_m': rpe_array,
    'Distance_m': distances_from_start,
    'Seconds_s': seconds_from_start
})

# --- 2. Print Statistics ---
print("\n" + "="*50)
print("     RELATIVE POSE ERROR (RPE) STATISTICS")
print("="*50)
print(json.dumps(stats, indent=4))
print("="*50 + "\n")

# =========================================================
# --- PLOT 1: RPE vs. Time (Line Plot) ---
# =========================================================
plt.figure(figsize=(12, 6))

plot_title = info.get("title", "Relative Pose Error (RPE) over Time")
label = info.get("label", "RPE (m)")

plt.plot(df['Seconds_s'], df['RPE_m'], label=label, linewidth=1.5, color='darkgreen')

stats_text = (
    f'RMSE: {stats["rmse"]:.3f} m\n'
    f'Mean: {stats["mean"]:.3f} m\n'
    f'Median: {stats["median"]:.3f} m\n'
    f'Std Dev: {stats["std"]:.3f} m'
)
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9),
         verticalalignment='top', fontsize=10)

plt.title(plot_title, fontsize=14)
plt.xlabel('Seconds from Start (s)', fontsize=12)
plt.ylabel('Translational Error (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("rpe_vs_time_plot.png")
print("Plot 1 saved: rpe_vs_time_plot.png")

# =========================================================
# --- PLOT 2: RPE Distribution (Histogram) ---
# =========================================================
plt.figure(figsize=(10, 6))
plt.hist(df['RPE_m'], bins=20, edgecolor='black', alpha=0.7)
plt.axvline(stats['rmse'], color='blue', linestyle='dashed', linewidth=1.5, label=f'RMSE: {stats["rmse"]:.3f} m')
plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {stats["mean"]:.3f} m')
plt.title('Relative Pose Error (RPE) Distribution', fontsize=14)
plt.xlabel('Translational Error (m)', fontsize=12)
plt.ylabel('Frequency (Number of Poses)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rpe_distribution_histogram.png")
print("Plot 2 saved: rpe_distribution_histogram.png")

# =========================================================
# --- PLOT 3 (ADVANCED): RPE Box Plot by Distance Segment ---
# =========================================================

# Determine segmentation parameters
max_dist = df['Distance_m'].max()
segment_size = 0.05  # Use 0.05m segments for high granularity
num_segments = int(np.ceil(max_dist / segment_size))

# Create bins and labels
bins = np.arange(0, max_dist + segment_size, segment_size)
if bins[-1] < max_dist:
    bins = np.append(bins, max_dist)

# Labeling segments
labels = [f'{bins[i]:.2f}m - {bins[i+1]:.2f}m' for i in range(len(bins)-1)] 

df['Distance_Segment'] = pd.cut(df['Distance_m'], bins=bins, labels=labels, include_lowest=True, right=True)

# Get sorted unique labels and segments data (using .cat.categories for guaranteed sort)
unique_labels = df['Distance_Segment'].cat.categories
df_filtered = df.dropna(subset=['Distance_Segment'])
segments_data = [df_filtered['RPE_m'][df_filtered['Distance_Segment'] == label].values for label in unique_labels]

plt.figure(figsize=(14, 7))
plt.boxplot(segments_data, 
            labels=[l.split('m - ')[0] + 'm' for l in unique_labels], # Use only the segment start point for label
            vert=True, patch_artist=True, medianprops=dict(color='red'))

plt.title(f'RPE Distribution (Box Plot) across Trajectory Segments ({segment_size:.2f}m Segments)', fontsize=14)
plt.xlabel('Distance Segment Start Point (m)', fontsize=12)
plt.ylabel('Translational Error (m)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("rpe_boxplot_by_segment.png")
print("Plot 3 saved: rpe_boxplot_by_segment.png")