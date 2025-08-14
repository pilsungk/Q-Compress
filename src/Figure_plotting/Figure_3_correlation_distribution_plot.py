import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = '12q-circ_id-angle_r.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: Make sure '{file_path}' is in the same directory.")
    exit()


status_col = 'robustness'
correlation_col = 'angle_correlation.overall_pearson_r'


try:
    robust_correlations = df[df[status_col] == 'robust'][correlation_col]
    fragile_correlations = df[df[status_col] == 'fragile'][correlation_col]
except KeyError:
    print(f"Error: Could not find the required columns. Please ensure the CSV contains '{status_col}' and '{correlation_col}'.")
    exit()
    

plt.figure(figsize=(10, 8))

# KDE (Kernel Density Estimate) 
sns.kdeplot(robust_correlations, fill=True, label=f'Robust (N={len(robust_correlations)})', color='blue', alpha=0.6, linewidth=2.5)
sns.kdeplot(fragile_correlations, fill=True, label=f'Fragile (N={len(fragile_correlations)})', color='red', alpha=0.6, linewidth=2.5)


plt.axvline(robust_correlations.mean(), color='blue', linestyle='--', label=f'Robust Mean: {robust_correlations.mean():.3f}')
plt.axvline(fragile_correlations.mean(), color='red', linestyle='--', label=f'Fragile Mean: {fragile_correlations.mean():.3f}')


#plt.title('Distribution of Angle-Importance Correlation (12 Qubits)', fontsize=16)
plt.xlabel('Angle-Importance Correlation (r)', fontsize=28)
plt.ylabel('Density', fontsize=28)
plt.legend(fontsize=18)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()


plt.savefig('figure3_correlation_distribution.pdf')
plt.show()
