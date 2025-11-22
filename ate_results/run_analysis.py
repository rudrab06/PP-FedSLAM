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
FILE_ALIGNMENT = 'alignment_transformation_sim3.npy'

# --- 1. Load Data ---
try:
    # Renaming variable to ATE for consistent terminology
    ate_array = np.load(FILE_ERROR) 
    distances_from_start = np.load(FILE_DISTANCE)
    with open(FILE_STATS, 'r') as f:
        stats = json.load(f)
        
    # Optional files
    try:
        seconds_from_start = np.load(FILE_SECONDS)
    except FileNotFoundError:
        # Use simple time placeholder if seconds_from_start is missing
        seconds_from_start = np.linspace(0, 1, len(ate_array))
        
    try:
        alignment_matrix = np.load(FILE_ALIGNMENT)
    except FileNotFoundError:
        alignment_matrix = None
        
    print("✅ All necessary data files loaded successfully.")

except FileNotFoundError as e:
    print(f"❌ Error: One or more files not found. Please verify files are present.")
    print(f"Missing file: {e}")
    exit()

# Validate data lengths match
if len(ate_array) != len(distances_from_start) or len(ate_array) != len(seconds_from_start):
    print(f"❌ Error: Array length mismatch!")
    print(f"  ate_array: {len(ate_array)}")
    print(f"  distances_from_start: {len(distances_from_start)}")
    print(f"  seconds_from_start: {len(seconds_from_start)}")
    exit(1)

# Validate data is not empty
if len(ate_array) == 0:
    print("❌ Error: No data to analyze!")
    exit(1)

# Create DataFrame for plotting/analysis
df = pd.DataFrame({
    'ATE_m': ate_array,
    'Distance_m': distances_from_start,
    'Seconds_s': seconds_from_start
})

# --- 2. Print Results ---
print("\n" + "="*50)
print("     ABSOLUTE TRAJECTORY ERROR (ATE) STATISTICS")
print("="*50)
print(json.dumps(stats, indent=4))
if alignment_matrix is not None:
    print("\n" + "="*50)
    print("     ALIGNMENT TRANSFORMATION (Sim3)")
    print("="*50)
    print(alignment_matrix)
print("="*50 + "\n")

# =========================================================
# --- PLOT 1: ATE vs. Distance (Line Plot) ---
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(df['Distance_m'], df['ATE_m'], label='Estimated Trajectory ATE', linewidth=1.5, color='darkorange')

# Safely extract stats with defaults if missing
rmse = stats.get("rmse", 0.0)
mean = stats.get("mean", 0.0)
median = stats.get("median", 0.0)
std = stats.get("std", 0.0)

stats_text = (
    f'RMSE: {rmse:.3f} m\n'
    f'Mean: {mean:.3f} m\n'
    f'Median: {median:.3f} m\n'
    f'Std Dev: {std:.3f} m'
)
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.9),
         verticalalignment='top', fontsize=10)

plt.title('Absolute Trajectory Error (ATE) over Trajectory Distance', fontsize=14)
plt.xlabel('Distance from Start (m)', fontsize=12)
plt.ylabel('Translational Error (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("ate_vs_distance_plot.png")
print("✅ Plot 1 saved: ate_vs_distance_plot.png")

# =========================================================
# --- PLOT 2: ATE Distribution (Histogram) ---
# =========================================================
plt.figure(figsize=(10, 6))
plt.hist(df['ATE_m'], bins=20, edgecolor='black', alpha=0.7)
if 'rmse' in stats:
    plt.axvline(stats['rmse'], color='blue', linestyle='dashed', linewidth=1.5, label=f'RMSE: {stats["rmse"]:.3f} m')
if 'mean' in stats:
    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {stats["mean"]:.3f} m')
plt.title('Absolute Trajectory Error (ATE) Distribution', fontsize=14)
plt.xlabel('Translational Error (m)', fontsize=12)
plt.ylabel('Frequency (Number of Poses)', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("ate_distribution_histogram.png")
print("✅ Plot 2 saved: ate_distribution_histogram.png")

# =========================================================
# --- PLOT 3 (ADVANCED): ATE Box Plot by Distance Segment ---
# =========================================================

# Determine segmentation parameters
max_dist = df['Distance_m'].max()
if max_dist <= 0:
    print("⚠️  Warning: Maximum distance is 0 or negative. Skipping Plot 3.")
    max_dist = 1.0  # fallback to avoid division by zero
segment_size = 0.05  # Using 0.05m segments for high granularity

# Create bins and labels
bins = np.arange(0, max_dist + segment_size, segment_size)
if len(bins) == 0:
    bins = np.array([0, max_dist])
elif bins[-1] < max_dist:
    bins = np.append(bins, max_dist)

# Ensure we have at least 2 bins
if len(bins) < 2:
    bins = np.array([0, max_dist])

# Labeling segments by their start point for cleaner x-axis
if len(bins) < 2:
    labels = []
else:
    labels = [f'{bins[i]:.2f}m - {bins[i+1]:.2f}m' for i in range(len(bins)-1)] 

df['Distance_Segment'] = pd.cut(df['Distance_m'], bins=bins, labels=labels, include_lowest=True, right=True)

# Get segments for boxplot (FIXED: using .cat.categories for guaranteed sorted labels)
unique_labels = df['Distance_Segment'].cat.categories

# Filter out empty segments and extract data
df_filtered = df.dropna(subset=['Distance_Segment'])
segments_data = []
segment_labels = []

for label in unique_labels:
    segment_values = df_filtered['ATE_m'][df_filtered['Distance_Segment'] == label].values
    # Only include segments with at least one data point
    if len(segment_values) > 0:
        segments_data.append(segment_values)
        # Extract segment start point for label (handle format safely)
        try:
            label_parts = label.split('m - ')
            if len(label_parts) > 0:
                segment_labels.append(label_parts[0] + 'm')
            else:
                segment_labels.append(str(label))
        except Exception:
            segment_labels.append(str(label))

# Only plot if we have data
if len(segments_data) == 0:
    print("⚠️  Warning: No data segments found for box plot. Skipping Plot 3.")
else:
    plt.figure(figsize=(14, 7))
    # Plot the box plot
    plt.boxplot(segments_data, 
                labels=segment_labels,
                vert=True, patch_artist=True, medianprops=dict(color='red'))

    plt.title(f'ATE Distribution (Box Plot) across Trajectory Segments ({segment_size:.2f}m Segments)', fontsize=14)
    plt.xlabel('Distance Segment Start Point (m)', fontsize=12)
    plt.ylabel('Translational Error (m)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("ate_boxplot_by_segment.png")
    print("✅ Plot 3 saved: ate_boxplot_by_segment.png")

# --- 4. Final CSV save (Combined Data) ---
df_save = df[['Distance_m', 'ATE_m', 'Seconds_s']].rename(columns={'Distance_m': 'Distance_from_Start_m', 'ATE_m': 'ATE_m', 'Seconds_s': 'Seconds_from_Start'})
csv_filename = 'ate_data_full.csv'
df_save.to_csv(csv_filename, index=False)
print(f"✅ Full ATE data saved to CSV: {csv_filename}")