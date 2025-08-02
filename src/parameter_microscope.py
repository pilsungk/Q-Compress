#!/usr/bin/env python3

"""
parameter_microscope.py
Microscopic analysis of gate parameters
"The Parameter Hypothesis": different parameter values determine fragility for structually uniform circuits
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import glob
from scipy import stats
import pandas as pd


@dataclass
class ParameterAnalysisResult:
    """parameter analysis result"""
    circuit_id: str
    classification: str
    fidelity: float
    
    # All parameter values
    all_parameters: List[float]
    rotation_parameters: List[float]  # RX, RY, RZ parameters only
    
    # Parameter statistics
    param_mean: float
    param_std: float
    param_median: float
    param_range: float  # max - min
    param_iqr: float  # interquartile range
    
    # Parameter distribution characteristics
    param_skewness: float
    param_kurtosis: float
    param_entropy: float
    
    # Small vs Large parameter analysis
    small_params_count: int  # |param| < threshold
    large_params_count: int  # |param| > threshold
    small_param_ratio: float
    
    # Critical gates parameter analysis
    critical_parameters: List[float]
    critical_param_mean: float
    critical_param_std: float
    
    # Parameter clustering
    param_clusters: Dict[str, List[float]]  # 'small', 'medium', 'large'
    
    # Extreme parameter analysis
    min_param: float
    max_param: float
    extreme_param_ratio: float  # ratio of very small + very large params
    
    # Parameter uniqueness
    unique_param_count: int
    param_diversity: float  # unique/total ratio
    
    # Gate-type specific parameter analysis
    rx_params: List[float]
    ry_params: List[float]
    rz_params: List[float]
    gate_param_variance: Dict[str, float]  # variance for each gate type


class ParameterMicroscope:
    """parameter microscope"""
    
    def __init__(self, 
                 importance_threshold_percentile: float = 90,
                 small_param_threshold: float = 0.1,
                 large_param_threshold: float = 3.0):
        self.importance_threshold_percentile = importance_threshold_percentile
        self.small_threshold = small_param_threshold
        self.large_threshold = large_param_threshold
        self.results = []
        
    def analyze_circuit(self, json_file_path: str) -> ParameterAnalysisResult:
        """analyze parameters of a single circuit"""
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        circuit_id = Path(json_file_path).stem
        classification = data['robustness_class']
        fidelity = data['fidelity']
        
        # extract circuit structure 
        gates = data['circuit_structure']['gates']
        importance_scores = data['importance_scores']
        
        # Critical gates identification
        threshold = np.percentile(importance_scores, self.importance_threshold_percentile)
        critical_indices = [i for i, score in enumerate(importance_scores) if score > threshold]
        
        # extract all parameters
        all_parameters = []
        rotation_parameters = []
        gate_type_params = defaultdict(list)
        
        for gate in gates:
            if 'params' in gate and gate['params']:
                for param in gate['params']:
                    all_parameters.append(param)
                    
                    # separately store rotation gates
                    if gate['name'] in ['rx', 'ry', 'rz']:
                        rotation_parameters.append(param)
                        gate_type_params[gate['name']].append(param)
        
        # Critical gates parameter
        critical_parameters = []
        for idx in critical_indices:
            if idx < len(gates) and 'params' in gates[idx] and gates[idx]['params']:
                critical_parameters.extend(gates[idx]['params'])
        
        # parameter statistics 
        if all_parameters:
            param_array = np.array(all_parameters)
            param_mean = np.mean(param_array)
            param_std = np.std(param_array)
            param_median = np.median(param_array)
            param_range = np.max(param_array) - np.min(param_array)
            param_iqr = np.percentile(param_array, 75) - np.percentile(param_array, 25)
            param_skewness = stats.skew(param_array)
            param_kurtosis = stats.kurtosis(param_array)
            param_entropy = self._calculate_param_entropy(param_array)
            min_param = np.min(param_array)
            max_param = np.max(param_array)
        else:
            param_mean = param_std = param_median = param_range = param_iqr = 0
            param_skewness = param_kurtosis = param_entropy = 0
            min_param = max_param = 0
        
        # Small vs Large parameter analysis
        small_params = [p for p in all_parameters if abs(p) < self.small_threshold]
        large_params = [p for p in all_parameters if abs(p) > self.large_threshold]
        
        small_params_count = len(small_params)
        large_params_count = len(large_params)
        small_param_ratio = small_params_count / len(all_parameters) if all_parameters else 0
        
        # Extreme parameter ratio
        very_small = [p for p in all_parameters if abs(p) < 0.01]
        very_large = [p for p in all_parameters if abs(p) > 6.0]
        extreme_param_ratio = (len(very_small) + len(very_large)) / len(all_parameters) if all_parameters else 0
        
        # Critical parameters statistics
        if critical_parameters:
            critical_param_mean = np.mean(critical_parameters)
            critical_param_std = np.std(critical_parameters)
        else:
            critical_param_mean = critical_param_std = 0
        
        # Parameter clustering
        param_clusters = {
            'small': [p for p in all_parameters if abs(p) < 1.0],
            'medium': [p for p in all_parameters if 1.0 <= abs(p) < 3.0],
            'large': [p for p in all_parameters if abs(p) >= 3.0]
        }
        
        # Parameter uniqueness
        unique_params = len(set([round(p, 6) for p in all_parameters]))
        param_diversity = unique_params / len(all_parameters) if all_parameters else 0
        
        # Gate-type specific variance
        gate_param_variance = {}
        for gate_type, params in gate_type_params.items():
            if params:
                gate_param_variance[gate_type] = np.var(params)
            else:
                gate_param_variance[gate_type] = 0
        
        return ParameterAnalysisResult(
            circuit_id=circuit_id,
            classification=classification,
            fidelity=fidelity,
            all_parameters=all_parameters,
            rotation_parameters=rotation_parameters,
            param_mean=param_mean,
            param_std=param_std,
            param_median=param_median,
            param_range=param_range,
            param_iqr=param_iqr,
            param_skewness=param_skewness,
            param_kurtosis=param_kurtosis,
            param_entropy=param_entropy,
            small_params_count=small_params_count,
            large_params_count=large_params_count,
            small_param_ratio=small_param_ratio,
            critical_parameters=critical_parameters,
            critical_param_mean=critical_param_mean,
            critical_param_std=critical_param_std,
            param_clusters=param_clusters,
            min_param=min_param,
            max_param=max_param,
            extreme_param_ratio=extreme_param_ratio,
            unique_param_count=unique_params,
            param_diversity=param_diversity,
            rx_params=gate_type_params['rx'],
            ry_params=gate_type_params['ry'],
            rz_params=gate_type_params['rz'],
            gate_param_variance=gate_param_variance
        )
    
    def _calculate_param_entropy(self, params: np.ndarray, bins: int = 50) -> float:
        """calculate entropy of parameter distribution"""
        if len(params) == 0:
            return 0
        
        hist, _ = np.histogram(params, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0
        
        # Normalize to get probabilities
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob))
        
        return entropy
    
    def analyze_directory(self, directory_path: str, pattern: str = "circuit_*.json") -> List[ParameterAnalysisResult]:
        """analyze all circuits in a directory"""
        json_files = glob.glob(f"{directory_path}/{pattern}")
        results = []
        
        print(f"Analyzing parameter patterns in {len(json_files)} circuits...")
        
        for file_path in sorted(json_files):
            try:
                result = self.analyze_circuit(file_path)
                results.append(result)
                print(f"Analyzed {result.circuit_id}: param_mean={result.param_mean:.4f}, "
                      f"param_std={result.param_std:.4f}, fidelity={result.fidelity:.4f}")
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        self.results = results
        return results
    
    def compare_parameter_patterns(self) -> Dict:
        """compare Robust vs Fragile circuits parameter pattern"""
        robust = [r for r in self.results if r.classification == 'robust']
        fragile = [r for r in self.results if r.classification == 'fragile']
        
        if not robust or not fragile:
            print("Not enough data for comparison")
            return {}
        
        def get_stats(circuits, attr):
            values = [getattr(c, attr) for c in circuits]
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        def perform_ttest(robust_circuits, fragile_circuits, attr):
            robust_vals = [getattr(c, attr) for c in robust_circuits]
            fragile_vals = [getattr(c, attr) for c in fragile_circuits]
            return stats.ttest_ind(robust_vals, fragile_vals)
        
        comparison = {
            'robust_count': len(robust),
            'fragile_count': len(fragile),
            
            # Basic parameter statistics
            'param_mean': {
                'robust': get_stats(robust, 'param_mean'),
                'fragile': get_stats(fragile, 'param_mean'),
                'ttest': perform_ttest(robust, fragile, 'param_mean')
            },
            
            'param_std': {
                'robust': get_stats(robust, 'param_std'),
                'fragile': get_stats(fragile, 'param_std'),
                'ttest': perform_ttest(robust, fragile, 'param_std')
            },
            
            'param_range': {
                'robust': get_stats(robust, 'param_range'),
                'fragile': get_stats(fragile, 'param_range'),
                'ttest': perform_ttest(robust, fragile, 'param_range')
            },
            
            # Distribution characteristics
            'param_skewness': {
                'robust': get_stats(robust, 'param_skewness'),
                'fragile': get_stats(fragile, 'param_skewness'),
                'ttest': perform_ttest(robust, fragile, 'param_skewness')
            },
            
            'param_entropy': {
                'robust': get_stats(robust, 'param_entropy'),
                'fragile': get_stats(fragile, 'param_entropy'),
                'ttest': perform_ttest(robust, fragile, 'param_entropy')
            },
            
            # Small parameter analysis
            'small_param_ratio': {
                'robust': get_stats(robust, 'small_param_ratio'),
                'fragile': get_stats(fragile, 'small_param_ratio'),
                'ttest': perform_ttest(robust, fragile, 'small_param_ratio')
            },
            
            # Extreme parameter analysis
            'extreme_param_ratio': {
                'robust': get_stats(robust, 'extreme_param_ratio'),
                'fragile': get_stats(fragile, 'extreme_param_ratio'),
                'ttest': perform_ttest(robust, fragile, 'extreme_param_ratio')
            },
            
            # Critical parameter analysis
            'critical_param_mean': {
                'robust': get_stats(robust, 'critical_param_mean'),
                'fragile': get_stats(fragile, 'critical_param_mean'),
                'ttest': perform_ttest(robust, fragile, 'critical_param_mean')
            },
            
            'critical_param_std': {
                'robust': get_stats(robust, 'critical_param_std'),
                'fragile': get_stats(fragile, 'critical_param_std'),
                'ttest': perform_ttest(robust, fragile, 'critical_param_std')
            },
            
            # Parameter diversity
            'param_diversity': {
                'robust': get_stats(robust, 'param_diversity'),
                'fragile': get_stats(fragile, 'param_diversity'),
                'ttest': perform_ttest(robust, fragile, 'param_diversity')
            }
        }
        
        return comparison
    
    def find_parameter_extremes(self, top_n: int = 5) -> Dict:
        return {
            'highest_param_std': sorted(self.results, key=lambda x: x.param_std, reverse=True)[:top_n],
            'lowest_param_std': sorted(self.results, key=lambda x: x.param_std)[:top_n],
            'highest_small_ratio': sorted(self.results, key=lambda x: x.small_param_ratio, reverse=True)[:top_n],
            'highest_extreme_ratio': sorted(self.results, key=lambda x: x.extreme_param_ratio, reverse=True)[:top_n],
            'highest_diversity': sorted(self.results, key=lambda x: x.param_diversity, reverse=True)[:top_n]
        }
    
    def analyze_gate_type_differences(self) -> Dict:
        robust = [r for r in self.results if r.classification == 'robust']
        fragile = [r for r in self.results if r.classification == 'fragile']
        
        gate_analysis = {}
        
        for gate_type in ['rx', 'ry', 'rz']:
            # Collect all parameters for this gate type
            robust_params = []
            fragile_params = []
            
            for result in robust:
                if gate_type in result.gate_param_variance:
                    gate_params = getattr(result, f'{gate_type}_params')
                    robust_params.extend(gate_params)
            
            for result in fragile:
                if gate_type in result.gate_param_variance:
                    gate_params = getattr(result, f'{gate_type}_params')
                    fragile_params.extend(gate_params)
            
            if robust_params and fragile_params:
                # Statistical comparison
                ttest = stats.ttest_ind(robust_params, fragile_params)
                ks_test = stats.ks_2samp(robust_params, fragile_params)
                
                gate_analysis[gate_type] = {
                    'robust_mean': np.mean(robust_params),
                    'robust_std': np.std(robust_params),
                    'fragile_mean': np.mean(fragile_params),
                    'fragile_std': np.std(fragile_params),
                    'ttest': ttest,
                    'ks_test': ks_test,
                    'robust_count': len(robust_params),
                    'fragile_count': len(fragile_params)
                }
        
        return gate_analysis
    
    def visualize_parameter_analysis(self, save_path: Optional[str] = None):
        if not self.results:
            print("No results to visualize")
            return
        
        robust = [r for r in self.results if r.classification == 'robust']
        fragile = [r for r in self.results if r.classification == 'fragile']
        
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Parameter mean comparison
        ax1 = plt.subplot(4, 4, 1)
        param_mean_robust = [r.param_mean for r in robust]
        param_mean_fragile = [r.param_mean for r in fragile]
        
        ax1.boxplot([param_mean_robust, param_mean_fragile], labels=['Robust', 'Fragile'])
        ax1.set_ylabel('Parameter Mean')
        ax1.set_title('Parameter Mean Comparison')
        ax1.grid(True, alpha=0.3)
        
        # 2. Parameter std comparison
        ax2 = plt.subplot(4, 4, 2)
        param_std_robust = [r.param_std for r in robust]
        param_std_fragile = [r.param_std for r in fragile]
        
        ax2.boxplot([param_std_robust, param_std_fragile], labels=['Robust', 'Fragile'])
        ax2.set_ylabel('Parameter Standard Deviation')
        ax2.set_title('Parameter Variability')
        ax2.grid(True, alpha=0.3)
        
        # 3. Small parameter ratio
        ax3 = plt.subplot(4, 4, 3)
        small_ratio_robust = [r.small_param_ratio for r in robust]
        small_ratio_fragile = [r.small_param_ratio for r in fragile]
        
        ax3.boxplot([small_ratio_robust, small_ratio_fragile], labels=['Robust', 'Fragile'])
        ax3.set_ylabel('Small Parameter Ratio')
        ax3.set_title('Small Parameters (|p| < 0.1)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Parameter entropy
        ax4 = plt.subplot(4, 4, 4)
        entropy_robust = [r.param_entropy for r in robust]
        entropy_fragile = [r.param_entropy for r in fragile]
        
        ax4.boxplot([entropy_robust, entropy_fragile], labels=['Robust', 'Fragile'])
        ax4.set_ylabel('Parameter Entropy')
        ax4.set_title('Parameter Distribution Entropy')
        ax4.grid(True, alpha=0.3)
        
        # 5. Fidelity vs Parameter Mean
        ax5 = plt.subplot(4, 4, 5)
        ax5.scatter(param_mean_robust, [r.fidelity for r in robust], 
                   alpha=0.6, label='Robust', s=50, color='blue')
        ax5.scatter(param_mean_fragile, [r.fidelity for r in fragile], 
                   alpha=0.6, label='Fragile', s=50, color='red')
        ax5.set_xlabel('Parameter Mean')
        ax5.set_ylabel('Fidelity')
        ax5.set_title('Fidelity vs Parameter Mean')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Fidelity vs Parameter Std
        ax6 = plt.subplot(4, 4, 6)
        ax6.scatter(param_std_robust, [r.fidelity for r in robust], 
                   alpha=0.6, label='Robust', s=50, color='blue')
        ax6.scatter(param_std_fragile, [r.fidelity for r in fragile], 
                   alpha=0.6, label='Fragile', s=50, color='red')
        ax6.set_xlabel('Parameter Std')
        ax6.set_ylabel('Fidelity')
        ax6.set_title('Fidelity vs Parameter Variability')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Parameter range comparison
        ax7 = plt.subplot(4, 4, 7)
        range_robust = [r.param_range for r in robust]
        range_fragile = [r.param_range for r in fragile]
        
        ax7.boxplot([range_robust, range_fragile], labels=['Robust', 'Fragile'])
        ax7.set_ylabel('Parameter Range')
        ax7.set_title('Parameter Range (Max - Min)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Extreme parameter ratio
        ax8 = plt.subplot(4, 4, 8)
        extreme_robust = [r.extreme_param_ratio for r in robust]
        extreme_fragile = [r.extreme_param_ratio for r in fragile]
        
        ax8.boxplot([extreme_robust, extreme_fragile], labels=['Robust', 'Fragile'])
        ax8.set_ylabel('Extreme Parameter Ratio')
        ax8.set_title('Very Small + Very Large Parameters')
        ax8.grid(True, alpha=0.3)
        
        # 9-11. Gate-type specific analysis
        gate_analysis = self.analyze_gate_type_differences()
        gate_types = ['rx', 'ry', 'rz']
        colors = ['red', 'green', 'blue']
        
        for i, gate_type in enumerate(gate_types):
            ax = plt.subplot(4, 4, 9 + i)
            
            if gate_type in gate_analysis:
                data = gate_analysis[gate_type]
                
                # Create violin plot for parameter distributions
                robust_params = []
                fragile_params = []
                
                for result in robust:
                    gate_params = getattr(result, f'{gate_type}_params')
                    robust_params.extend(gate_params)
                
                for result in fragile:
                    gate_params = getattr(result, f'{gate_type}_params')
                    fragile_params.extend(gate_params)
                
                if robust_params and fragile_params:
                    ax.hist(robust_params, bins=30, alpha=0.7, label='Robust', density=True, color='blue')
                    ax.hist(fragile_params, bins=30, alpha=0.7, label='Fragile', density=True, color='red')
                    ax.set_xlabel(f'{gate_type.upper()} Parameter Value')
                    ax.set_ylabel('Density')
                    ax.set_title(f'{gate_type.upper()} Parameter Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        # 12. Critical vs Non-critical parameter comparison
        ax12 = plt.subplot(4, 4, 12)
        critical_mean_robust = [r.critical_param_mean for r in robust]
        critical_mean_fragile = [r.critical_param_mean for r in fragile]
        
        ax12.boxplot([critical_mean_robust, critical_mean_fragile], labels=['Robust', 'Fragile'])
        ax12.set_ylabel('Critical Parameter Mean')
        ax12.set_title('Critical Gates Parameter Mean')
        ax12.grid(True, alpha=0.3)
        
        # 13. Parameter diversity
        ax13 = plt.subplot(4, 4, 13)
        diversity_robust = [r.param_diversity for r in robust]
        diversity_fragile = [r.param_diversity for r in fragile]
        
        ax13.boxplot([diversity_robust, diversity_fragile], labels=['Robust', 'Fragile'])
        ax13.set_ylabel('Parameter Diversity')
        ax13.set_title('Parameter Uniqueness Ratio')
        ax13.grid(True, alpha=0.3)
        
        # 14. Parameter histogram comparison
        ax14 = plt.subplot(4, 4, 14)
        
        # Collect all parameters
        all_robust_params = []
        all_fragile_params = []
        
        for result in robust:
            all_robust_params.extend(result.all_parameters)
        
        for result in fragile:
            all_fragile_params.extend(result.all_parameters)
        
        ax14.hist(all_robust_params, bins=50, alpha=0.7, label='Robust', density=True, color='blue')
        ax14.hist(all_fragile_params, bins=50, alpha=0.7, label='Fragile', density=True, color='red')
        ax14.set_xlabel('Parameter Value')
        ax14.set_ylabel('Density')
        ax14.set_title('All Parameters Distribution')
        ax14.legend()
        ax14.grid(True, alpha=0.3)
        ax14.set_xlim(-1, 7)  # Focus on main range
        
        # 15. Parameter skewness
        ax15 = plt.subplot(4, 4, 15)
        skew_robust = [r.param_skewness for r in robust]
        skew_fragile = [r.param_skewness for r in fragile]
        
        ax15.boxplot([skew_robust, skew_fragile], labels=['Robust', 'Fragile'])
        ax15.set_ylabel('Parameter Skewness')
        ax15.set_title('Parameter Distribution Asymmetry')
        ax15.grid(True, alpha=0.3)
        
        # 16. 2D parameter space
        ax16 = plt.subplot(4, 4, 16)
        ax16.scatter(param_mean_robust, param_std_robust, 
                    alpha=0.6, label='Robust', s=50, color='blue')
        ax16.scatter(param_mean_fragile, param_std_fragile, 
                    alpha=0.6, label='Fragile', s=50, color='red')
        ax16.set_xlabel('Parameter Mean')
        ax16.set_ylabel('Parameter Std')
        ax16.set_title('Parameter Mean vs Std Space')
        ax16.legend()
        ax16.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_parameter_report(self) -> str:
        if not self.results:
            return "No analysis results available"
        
        comparison = self.compare_parameter_patterns()
        extremes = self.find_parameter_extremes(3)
        gate_analysis = self.analyze_gate_type_differences()
        
        report = []
        report.append("=== PARAMETER MICROSCOPE ANALYSIS REPORT ===")
        report.append("HYPOTHESIS: 'Parameter values determine circuit fragility'\n")
        
        report.append(f"Total circuits analyzed: {len(self.results)}")
        report.append(f"Robust circuits: {comparison['robust_count']}")
        report.append(f"Fragile circuits: {comparison['fragile_count']}\n")
        
        # Main hypothesis tests
        report.append("=== MAIN PARAMETER HYPOTHESES ===")
        
        # 1. Parameter mean
        param_mean = comparison['param_mean']
        report.append(f"1. Parameter Mean:")
        report.append(f"   Robust:  {param_mean['robust']['mean']:.4f} ± {param_mean['robust']['std']:.4f}")
        report.append(f"   Fragile: {param_mean['fragile']['mean']:.4f} ± {param_mean['fragile']['std']:.4f}")
        report.append(f"   Difference: {param_mean['fragile']['mean'] - param_mean['robust']['mean']:+.4f}")
        report.append(f"   P-value: {param_mean['ttest'].pvalue:.6f} " + 
                     f"({'SIGNIFICANT' if param_mean['ttest'].pvalue < 0.05 else 'not significant'})")
        
        # 2. Parameter std
        param_std = comparison['param_std']
        report.append(f"\n2. Parameter Variability:")
        report.append(f"   Robust:  {param_std['robust']['mean']:.4f} ± {param_std['robust']['std']:.4f}")
        report.append(f"   Fragile: {param_std['fragile']['mean']:.4f} ± {param_std['fragile']['std']:.4f}")
        report.append(f"   Difference: {param_std['fragile']['mean'] - param_std['robust']['mean']:+.4f}")
        report.append(f"   P-value: {param_std['ttest'].pvalue:.6f} " + 
                     f"({'SIGNIFICANT' if param_std['ttest'].pvalue < 0.05 else 'not significant'})")
        
        # 3. Small parameter ratio
        small_ratio = comparison['small_param_ratio']
        report.append(f"\n3. Small Parameter Ratio:")
        report.append(f"   Robust:  {small_ratio['robust']['mean']:.4f} ± {small_ratio['robust']['std']:.4f}")
        report.append(f"   Fragile: {small_ratio['fragile']['mean']:.4f} ± {small_ratio['fragile']['std']:.4f}")
        report.append(f"   Difference: {small_ratio['fragile']['mean'] - small_ratio['robust']['mean']:+.4f}")
        report.append(f"   P-value: {small_ratio['ttest'].pvalue:.6f} " + 
                     f"({'SIGNIFICANT' if small_ratio['ttest'].pvalue < 0.05 else 'not significant'})")
        
        # Gate-type specific analysis
        report.append("\n=== GATE-TYPE SPECIFIC ANALYSIS ===")
        
        for gate_type, data in gate_analysis.items():
            report.append(f"\n{gate_type.upper()} Gates:")
            report.append(f"   Robust:  {data['robust_mean']:.4f} ± {data['robust_std']:.4f} ({data['robust_count']} params)")
            report.append(f"   Fragile: {data['fragile_mean']:.4f} ± {data['fragile_std']:.4f} ({data['fragile_count']} params)")
            report.append(f"   T-test p-value: {data['ttest'].pvalue:.6f}")
            report.append(f"   KS-test p-value: {data['ks_test'].pvalue:.6f}")
        
        # Extreme cases
        report.append("\n=== EXTREME PARAMETER CASES ===")
        
        report.append("Highest Parameter Variability:")
        for i, result in enumerate(extremes['highest_param_std'], 1):
            report.append(f"   {i}. {result.circuit_id} ({result.classification}): "
                         f"std={result.param_std:.4f}, fidelity={result.fidelity:.4f}")
        
        report.append("Lowest Parameter Variability:")
        for i, result in enumerate(extremes['lowest_param_std'], 1):
            report.append(f"   {i}. {result.circuit_id} ({result.classification}): "
                         f"std={result.param_std:.4f}, fidelity={result.fidelity:.4f}")
        
        # Find most significant differences
        report.append("\n=== MOST SIGNIFICANT FINDINGS ===")
        
        significant_tests = []
        for test_name, test_data in comparison.items():
            if isinstance(test_data, dict) and 'ttest' in test_data:
                p_value = test_data['ttest'].pvalue
                if p_value < 0.05:
                    diff = test_data['fragile']['mean'] - test_data['robust']['mean']
                    significant_tests.append((test_name, p_value, diff))
        
        significant_tests.sort(key=lambda x: x[1])  # Sort by p-value
        
        if significant_tests:
            report.append("Statistically significant differences found:")
            for test_name, p_value, diff in significant_tests:
                report.append(f"   {test_name}: p={p_value:.6f}, diff={diff:+.4f}")
        else:
            report.append("No statistically significant differences found in parameter patterns.")
        
        # Conclusion
        report.append("\n=== CONCLUSION ===")
        
        if significant_tests:
            report.append("PARAMETER HYPOTHESIS PARTIALLY CONFIRMED:")
            report.append("Some parameter characteristics show significant differences.")
            report.append("The fragility mechanism may involve specific parameter patterns.")
        else:
            report.append("PARAMETER HYPOTHESIS INCONCLUSIVE:")
            report.append("No clear parameter-based differences between robust and fragile circuits.")
            report.append("The fragility mechanism remains elusive at the parameter level.")
        
        return "\n".join(report)


def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python parameter_microscope.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    print("PARAMETER MICROSCOPE ANALYSIS")
    print("Examining gate parameter values for fragility clues...")
    print("This is our deepest dive yet into the quantum circuit structure!\n")
    
    analyzer = ParameterMicroscope()
    results = analyzer.analyze_directory(directory_path)
    
    if not results:
        print("No results to analyze")
        return
    
    # compare pattern
    comparison = analyzer.compare_parameter_patterns()
    
    # generate report
    report = analyzer.generate_parameter_report()
    print(report)
    
    # visualization
    print("\nGenerating parameter microscope visualization...")
    fig = analyzer.visualize_parameter_analysis()
    plt.savefig("parameter_microscope_analysis.pdf", dpi=300, bbox_inches='tight')
    print("Visualization saved to: parameter_microscope_analysis.pdf")
    
    # save report
    with open("parameter_microscope_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
    print("Report saved to: parameter_microscope_report.txt")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'circuit_id': result.circuit_id,
            'classification': result.classification,
            'fidelity': result.fidelity,
            'param_mean': result.param_mean,
            'param_std': result.param_std,
            'param_range': result.param_range,
            'param_skewness': result.param_skewness,
            'param_entropy': result.param_entropy,
            'small_param_ratio': result.small_param_ratio,
            'extreme_param_ratio': result.extreme_param_ratio,
            'param_diversity': result.param_diversity,
            'critical_param_mean': result.critical_param_mean,
            'critical_param_std': result.critical_param_std
        })
    
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv("parameter_microscope_data.csv", index=False)
    print("Detailed data saved to: parameter_microscope_data.csv")
    
    plt.show()


if __name__ == "__main__":
    main()
