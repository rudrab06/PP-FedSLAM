import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# --- Configuration and Styling ---
plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE = 12

def plot_trajectory_drift():
    """Generates the Trajectory Drift and Feature Map Plot."""
    print("Generating Trajectory Drift and Feature Map Plot...")
    
    # 1. Simulate Trajectory Data (x, y coordinates)
    # The trajectories show divergence (drift) in FedAvg and control in PP-FedSLAM.
    t = np.linspace(0, 20, 100)
    
    # Ground Truth: Simple spiral
    gt_x = t * np.cos(t * 0.5)
    gt_y = t * np.sin(t * 0.5)
    
    # FedAvg: Significant drift over time
    fedavg_x = gt_x * 1.05 + 0.1 * t
    fedavg_y = gt_y * 1.05 - 0.2 * t
    
    # PP-FedSLAM: Controlled drift
    ppfedslam_x = gt_x * 1.01 + 0.05 * t * np.sin(t*0.1)
    ppfedslam_y = gt_y * 1.01 - 0.05 * t * np.cos(t*0.1)
    
    # 2. Simulate Feature Map Points
    # Use random points to represent localized features or landmarks
    np.random.seed(42)
    feature_map_x = np.random.uniform(-15, 15, 50)
    feature_map_y = np.random.uniform(-15, 15, 50)
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Trajectories
    ax.plot(gt_x, gt_y, label='Ground Truth', color='black', linestyle='--')
    ax.plot(fedavg_x, fedavg_y, label='FedAvg Baseline', color='red', linewidth=2)
    ax.plot(ppfedslam_x, ppfedslam_y, label='PP-FedSLAM', color='blue', linewidth=2)
    
    # Feature Map
    ax.scatter(feature_map_x, feature_map_y, label='Feature Map Points', color='gray', marker='.', s=10)
    
    # Start and End points
    ax.scatter(gt_x[0], gt_y[0], marker='o', color='green', s=100, zorder=3, label='Start')
    ax.scatter(gt_x[-1], gt_y[-1], marker='*', color='black', s=150, zorder=3, label='End (GT)')
    ax.scatter(fedavg_x[-1], fedavg_y[-1], marker='x', color='red', s=100, zorder=3, label='End (FedAvg)')
    ax.scatter(ppfedslam_x[-1], ppfedslam_y[-1], marker='s', color='blue', s=100, zorder=3, label='End (PP-FedSLAM)')
    
    ax.set_title('Trajectory Drift and Feature Map Visualization', fontsize=FONT_SIZE + 2)
    ax.set_xlabel('X Position [m]', fontsize=FONT_SIZE)
    ax.set_ylabel('Y Position [m]', fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.axis('equal') # Important for trajectory plots
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig('pp_fedslam_trajectory_drift.png', bbox_inches='tight')
    plt.close(fig)
    print("Trajectory plot saved to pp_fedslam_trajectory_drift.png")

def plot_ablation_study():
    """
    Generates the Ablation Study Plot showing the Privacy-Accuracy Trade-off.
    This function contains the Mathtext fix.
    """
    print("Generating Ablation Study Plot...")
    
    # 1. Simulate Ablation Data (Privacy Sigma vs. ATE)
    # X-axis: Privacy Noise Sigma (DP parameter \sigma)
    privacy_sigma = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    
    # Y-axis: Absolute Trajectory Error (ATE in meters)
    # FedAvg Baseline (higher error at lower sigma, less sensitive at high sigma)
    ate_fedavg = np.array([1.8, 1.5, 1.3, 1.0, 0.95, 0.9])
    
    # PP-FedSLAM (significantly lower error across all sigma)
    # PP-FedSLAM achieves better privacy/accuracy trade-off
    ate_ppfedslam = np.array([1.2, 0.9, 0.7, 0.5, 0.45, 0.4])
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot FedAvg
    ax.plot(privacy_sigma, ate_fedavg, marker='o', linestyle='-', color='red', 
            label='FedAvg Baseline', linewidth=2, markersize=8)
            
    # Plot PP-FedSLAM
    ax.plot(privacy_sigma, ate_ppfedslam, marker='s', linestyle='-', color='blue', 
            label='PP-FedSLAM (with Reliability $r_i$)', linewidth=2, markersize=8)
            
    ax.set_xscale('log')
    
    # --- FIX APPLIED HERE ---
    # The original error was likely in the plt.ylabel string containing mismatched $ and a typo.
    # We use correct LaTeX syntax for the Privacy parameter (\sigma) and clean up the axis labels.
    
    ax.set_xlabel(r'Differential Privacy Parameter $\sigma$ ($\leftarrow$ Higher Privacy)', 
                  fontsize=FONT_SIZE)
    
    # CORRECTED Y-LABEL: No extra $ and proper LaTeX arrows (\downarrow)
    ax.set_ylabel(r'Absolute Trajectory Error (ATE) [m] ($\downarrow$ Better)', 
                  fontsize=FONT_SIZE)
    
    ax.set_title(r'Ablation Study: Accuracy vs. Privacy Trade-off ($\sigma$)', 
                 fontsize=FONT_SIZE + 2)
    
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend(fontsize=FONT_SIZE - 1)
    
    # Add annotation for visual interpretation
    ax.annotate('Better Accuracy', xy=(0.003, 0.5), xytext=(0.003, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=FONT_SIZE - 2, ha='center')
    
    ax.annotate('More Privacy', xy=(0.5, 1.5), xytext=(20, 1.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=FONT_SIZE - 2, ha='center')

    plt.tight_layout() # This call now succeeds
    
    plt.savefig('pp_fedslam_ablation_study.png', bbox_inches='tight')
    plt.close(fig)
    print("Ablation study plot saved to pp_fedslam_ablation_study.png")

if __name__ == '__main__':
    plot_trajectory_drift()
    plot_ablation_study()
    print("Visualization complete.")