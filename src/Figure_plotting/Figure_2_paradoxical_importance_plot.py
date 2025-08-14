import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def plot_paradoxical_importance(robust_csv_path, fragile_csv_path, output_path='paradoxical_importance.pdf'):
    """
    Visualize relationship between rotation parameter and importance of 
    Robust & Fragile circuits
    
    Parameters:
    robust_csv_path (str): path to Robust circuit CSV 
    fragile_csv_path (str): path to Fragile circuit CSV 
    output_path (str): path to output graph file 
    """
    
    # Read data
    df_robust = pd.read_csv(robust_csv_path)
    df_fragile = pd.read_csv(fragile_csv_path)
    
    # Figure setting
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Figure creation (1x2 subplot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    
    # colors
    color_robust = '#2E86AB'  # blue
    color_fragile = '#E63946'  # red
    point_alpha = 0.5
    
    # (a) Robust circuit
    x_robust = df_robust['rotation_angle']
    y_robust = df_robust['gate_importance']
    
    # scatter plot
    ax1.scatter(x_robust, y_robust, color=color_robust, alpha=point_alpha, s=30, label='Gates')
    
    # regression calculation and plotting
    slope_robust, intercept_robust, r_value_robust, p_value_robust, std_err_robust = stats.linregress(x_robust, y_robust)
    x_line = np.linspace(-0.1, np.pi/2 + 0.1, 100)
    y_line = slope_robust * x_line + intercept_robust
    ax1.plot(x_line, y_line, color='darkblue', linewidth=2.5, linestyle='--', 
             label=f'Fit (r={r_value_robust:.3f})')
    
    # axis
    ax1.set_xlabel('Rotation Angle (θ)', fontsize=13)
    ax1.set_ylabel('Gate Importance', fontsize=13)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.legend(loc='center right', frameon=True, fancybox=False, fontsize=12)
    
    # X range (-0.1 ~ π/2 + 0.1)
    ax1.set_xlim(-0.1, np.pi/2 + 0.1)
    
    ax1.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax1.set_xticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'], fontsize=13)
        
    # subplot caption (bottom)
    ax1.text(0.5, -0.25, '(a) Representative Robust Circuit', transform=ax1.transAxes, 
             ha='center', fontsize=14, fontweight='bold')
    
    # (b) Fragile circuit
    x_fragile = df_fragile['rotation_angle']
    y_fragile = df_fragile['gate_importance']
    
    # scatter plot
    ax2.scatter(x_fragile, y_fragile, color=color_fragile, alpha=point_alpha, s=30, label='Gates')
    
    # regression calculation and plot
    slope_fragile, intercept_fragile, r_value_fragile, p_value_fragile, std_err_fragile = stats.linregress(x_fragile, y_fragile)
    x_line = np.linspace(-0.1, np.pi/2 + 0.1, 100)
    y_line = slope_fragile * x_line + intercept_fragile
    ax2.plot(x_line, y_line, color='darkblue', linewidth=2.5, linestyle='--', 
             label=f'Fit (r={r_value_fragile:.3f})')
    
    # axis
    ax2.set_xlabel('Rotation Angle (θ)', fontsize=13)
    ax2.set_ylabel('Gate Importance', fontsize=13)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.legend(loc='center right', frameon=True, fancybox=False, fontsize=12)
    
    # X range (-0.1 ~ π/2 + 0.1)
    ax2.set_xlim(-0.1, np.pi/2 + 0.1)
    ax2.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
    ax2.set_xticklabels(['0', 'π/8', 'π/4', '3π/8', 'π/2'], fontsize=13)
    
    # subplot caption (bottom)
    ax2.text(0.5, -0.25, '(b) Representative Fragile Circuit', transform=ax2.transAxes, 
             ha='center', fontsize=14, fontweight='bold')
    
    # Y axis scale (for easy comparison)
    y_max = max(y_robust.max(), y_fragile.max()) * 1.1
    y_min = min(y_robust.min(), y_fragile.min()) * 0.9
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    # layout
    fig.subplots_adjust(hspace=0.4)
    #plt.tight_layout()
    
    # statistics 
    print("\n=== Paradoxical Importance Analysis ===")
    print(f"\nRobust Circuit:")
    print(f"  Number of rotation gates: {len(df_robust)}")
    print(f"  Correlation coefficient (r): {r_value_robust:.4f}")
    print(f"  Slope: {slope_robust:.4f}")
    print(f"  P-value: {p_value_robust:.4e}")
    
    print(f"\nFragile Circuit:")
    print(f"  Number of rotation gates: {len(df_fragile)}")
    print(f"  Correlation coefficient (r): {r_value_fragile:.4f}")
    print(f"  Slope: {slope_fragile:.4f}")
    print(f"  P-value: {p_value_fragile:.4e}")
    
    # save graphs
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'robust': {'r': r_value_robust, 'slope': slope_robust, 'p_value': p_value_robust},
        'fragile': {'r': r_value_fragile, 'slope': slope_fragile, 'p_value': p_value_fragile}
    }


# additional analysis: average importance for each angle range 
def analyze_angle_bins(robust_csv_path, fragile_csv_path, n_bins=8):
    """
    analyze average importance by dividing rotation angle by range
    """
    df_robust = pd.read_csv(robust_csv_path)
    df_fragile = pd.read_csv(fragile_csv_path)
    
    # angle range (0 ~ 2π)
    bins = np.linspace(0, 2*np.pi, n_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Robust circuit
    df_robust['angle_bin'] = pd.cut(df_robust['rotation_angle'], bins=bins)
    robust_mean = df_robust.groupby('angle_bin')['gate_importance'].mean()
    robust_std = df_robust.groupby('angle_bin')['gate_importance'].std()
    
    # Fragile circuit
    df_fragile['angle_bin'] = pd.cut(df_fragile['rotation_angle'], bins=bins)
    fragile_mean = df_fragile.groupby('angle_bin')['gate_importance'].mean()
    fragile_std = df_fragile.groupby('angle_bin')['gate_importance'].std()
    
    # Bar plot visualization 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    width = 0.35
    x = np.arange(len(bin_centers))
    
    bars1 = ax.bar(x - width/2, robust_mean, width, yerr=robust_std, 
                    label='Robust', color='#2E86AB', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, fragile_mean, width, yerr=fragile_std,
                    label='Fragile', color='#E63946', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Rotation Angle Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Gate Importance', fontsize=12, fontweight='bold')
    ax.set_title('Average Importance by Angle Range', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i*2*np.pi/n_bins:.2f}~{(i+1)*2*np.pi/n_bins:.2f}' 
                        for i in range(n_bins)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('angle_bin_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()


# example usage
if __name__ == "__main__":
    # main graph
    robust_csv = "circuit_098_robust_rotation_gates.csv"
    fragile_csv = "circuit_018_fragile_rotation_gates.csv"
    
    results = plot_paradoxical_importance(robust_csv, fragile_csv)
    
    # additional analysis (option)
    #analyze_angle_bins(robust_csv, fragile_csv)


