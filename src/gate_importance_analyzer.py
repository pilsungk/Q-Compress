#!/usr/bin/env python3

"""
Gate Importance Definition and Analysis
Compatible with Qiskit 1.4+
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt

class GateImportanceAnalyzer:
    """
    Gate Importance Definition and Analysis
    
    Gate Importance I_i for gate i is defined as the causal impact of removing that gate
    on the overall circuit fidelity.
    """
    
    def __init__(self):
        self.importance_threshold = 1e-10
    
    def compute_gate_importance(self, circuit: QuantumCircuit) -> list:
        """
        Compute importance score for each gate in the circuit.
        
        Definition:
        I_i = Causal impact of gate i on circuit fidelity
        
        Mathematical Definition:
        For gate i in circuit C:
        I_i = 1 - F(|ψ_original⟩, |ψ_without_i⟩)
        
        Where:
        - |ψ_original⟩ = Statevector(C) [original circuit]
        - |ψ_without_i⟩ = Statevector(C \ {gate_i}) [circuit without gate i]
        - F(|ψ₁⟩, |ψ₂⟩) = |⟨ψ₁|ψ₂⟩|² [quantum state fidelity]
        
        Special Cases:
        - If F > 0.9999: I_i = -log₁₀(max(1-F, 1e-16)) [log scale for high fidelity]
        - If F ≤ 0.9999: I_i = 1 - F [linear scale for significant impact]
        
        Range: I_i ∈ [0, 1]
        - I_i ≈ 0: Gate has minimal impact (redundant/small rotation)
        - I_i ≈ 1: Gate has critical impact (essential for computation)
        """
        
        print(f"    Computing gate importance for {len(circuit.data)} gates...")
        
        # Compute original state
        try:
            original_state = Statevector(circuit)
        except Exception as e:
            print(f"    Error computing original state: {e}")
            return [0.0] * len(circuit.data)
        
        importance_scores = []
        
        for i in range(len(circuit.data)):
            # Create modified circuit without gate i
            modified_circuit = QuantumCircuit(circuit.num_qubits)
            
            for j, instruction in enumerate(circuit.data):
                if j != i:  # Skip gate i
                    try:
                        modified_circuit.append(
                            instruction.operation, 
                            instruction.qubits, 
                            instruction.clbits
                        )
                    except Exception as e:
                        print(f"    Warning: Could not add gate {j}: {e}")
                        continue
            
            # Compute modified state
            try:
                modified_state = Statevector(modified_circuit)
                fidelity = state_fidelity(original_state, modified_state)
                
                # Apply importance transformation
                if fidelity > 0.9999:
                    # High fidelity: use logarithmic scale
                    importance = -np.log10(max(1 - fidelity, 1e-16))
                else:
                    # Significant impact: use linear scale  
                    importance = 1 - fidelity
                
                importance_scores.append(importance)
                
            except Exception as e:
                print(f"    Error computing importance for gate {i}: {e}")
                importance_scores.append(0.0)
        
        return importance_scores
    
    def analyze_importance_distribution(self, importance_scores: list) -> dict:
        """
        Analyze the distribution characteristics of gate importance scores.
        
        This analysis tests the Information Distribution Hypothesis:
        - Robust circuits: High entropy, uniform distribution
        - Fragile circuits: Low entropy, concentrated distribution
        """
        
        scores = np.array(importance_scores)
        
        # Basic statistics
        mean_importance = np.mean(scores)
        std_importance = np.std(scores)
        
        # Normalize for entropy calculation (ensure sum = 1)
        normalized_scores = scores / (np.sum(scores) + 1e-10)
        
        # Shannon Entropy: H = -Σ(p_i * log(p_i))
        # High H: uniform distribution (robust hypothesis)
        # Low H: concentrated distribution (fragile hypothesis)
        entropy = -np.sum(normalized_scores * np.log(normalized_scores + 1e-10))
        
        # Concentration Ratio: importance of top 10% gates
        sorted_scores = np.sort(scores)[::-1]
        top10_count = max(1, len(scores) // 10)
        concentration_ratio = np.sum(sorted_scores[:top10_count]) / (np.sum(scores) + 1e-10)
        
        # Gini Coefficient: measure of inequality
        # 0 = perfect equality, 1 = perfect inequality
        sorted_scores_gini = np.sort(scores)
        n = len(scores)
        if n > 0 and np.sum(scores) > 0:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_scores_gini)) / (n * np.sum(scores)) - (n + 1) / n
        else:
            gini = 0.0
        
        # Critical gates (above 90th percentile)
        if len(scores) > 0:
            threshold_90 = np.percentile(scores, 90)
            critical_gates = np.where(scores > threshold_90)[0]
        else:
            critical_gates = np.array([])
        
        # Uniformity index (inverse of coefficient of variation)
        cv = std_importance / (mean_importance + 1e-10)
        uniformity_index = 1 / (1 + cv)
        
        return {
            'mean': float(mean_importance),
            'std': float(std_importance),
            'entropy': float(entropy),
            'concentration_ratio': float(concentration_ratio),
            'gini_coefficient': float(gini),
            'uniformity_index': float(uniformity_index),
            'critical_gates_count': len(critical_gates),
            'critical_gates_indices': critical_gates.tolist(),
            'percentiles': {
                '50': float(np.percentile(scores, 50)) if len(scores) > 0 else 0.0,
                '75': float(np.percentile(scores, 75)) if len(scores) > 0 else 0.0,
                '90': float(np.percentile(scores, 90)) if len(scores) > 0 else 0.0,
                '95': float(np.percentile(scores, 95)) if len(scores) > 0 else 0.0,
                '99': float(np.percentile(scores, 99)) if len(scores) > 0 else 0.0
            }
        }
    
    def compare_robust_vs_fragile_importance(self, robust_scores_list: list, 
                                           fragile_scores_list: list):
        """
        Compare importance distributions between robust and fragile circuits
        """
        
        print("=== IMPORTANCE DISTRIBUTION COMPARISON ===")
        
        # Analyze each group
        robust_analyses = [self.analyze_importance_distribution(scores) 
                          for scores in robust_scores_list]
        fragile_analyses = [self.analyze_importance_distribution(scores) 
                           for scores in fragile_scores_list]
        
        # Extract key metrics
        metrics = ['entropy', 'concentration_ratio', 'gini_coefficient', 'uniformity_index']
        
        comparison_results = {}
        
        for metric in metrics:
            robust_values = [analysis[metric] for analysis in robust_analyses]
            fragile_values = [analysis[metric] for analysis in fragile_analyses]
            
            # Statistical test (try scipy, fallback to basic stats)
            try:
                from scipy import stats
                statistic, p_value = stats.mannwhitneyu(robust_values, fragile_values, alternative='two-sided')
            except ImportError:
                # Basic comparison without statistical test
                statistic = 0.0
                p_value = 1.0
                print(f"  Warning: scipy not available, no statistical test for {metric}")
            
            # Effect size (Cohen's d)
            def cohens_d(group1, group2):
                if len(group1) == 0 or len(group2) == 0:
                    return 0.0
                n1, n2 = len(group1), len(group2)
                if n1 == 1 and n2 == 1:
                    return 0.0
                var1 = np.var(group1, ddof=1) if n1 > 1 else 0.0
                var2 = np.var(group2, ddof=1) if n2 > 1 else 0.0
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
                if pooled_std == 0:
                    return 0.0
                return (np.mean(group1) - np.mean(group2)) / pooled_std
            
            effect_size = cohens_d(robust_values, fragile_values)
            
            comparison_results[metric] = {
                'robust_mean': float(np.mean(robust_values)) if robust_values else 0.0,
                'robust_std': float(np.std(robust_values)) if robust_values else 0.0,
                'fragile_mean': float(np.mean(fragile_values)) if fragile_values else 0.0,
                'fragile_std': float(np.std(fragile_values)) if fragile_values else 0.0,
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'difference': float(np.mean(robust_values) - np.mean(fragile_values)) if robust_values and fragile_values else 0.0
            }
            
            print(f"\n{metric.upper()}:")
            print(f"  Robust: {comparison_results[metric]['robust_mean']:.4f} ± {comparison_results[metric]['robust_std']:.4f}")
            print(f"  Fragile: {comparison_results[metric]['fragile_mean']:.4f} ± {comparison_results[metric]['fragile_std']:.4f}")
            print(f"  Difference: {comparison_results[metric]['difference']:.4f}")
            print(f"  P-value: {comparison_results[metric]['p_value']:.6f}")
            print(f"  Effect size: {comparison_results[metric]['effect_size']:.3f}")
        
        return comparison_results

    def visualize_importance_distribution(self, importance_scores: list, 
                                        title: str = "Gate Importance Distribution"):
        """Visualize the importance distribution"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            scores = np.array(importance_scores)
            
            # 1. Histogram
            ax1.hist(scores, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Gate Importance Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{title}\nHistogram')
            ax1.grid(True, alpha=0.3)
            
            # 2. Sorted importance (rank plot)
            sorted_scores = np.sort(scores)[::-1]
            ax2.plot(range(len(sorted_scores)), sorted_scores, 'o-', markersize=3)
            ax2.set_xlabel('Gate Rank')
            ax2.set_ylabel('Importance Score')
            ax2.set_title('Importance by Rank')
            ax2.grid(True, alpha=0.3)
            if np.any(sorted_scores > 0):
                ax2.set_yscale('log')
            
            # 3. Cumulative distribution
            sorted_scores_cum = np.sort(scores)
            cumulative = np.arange(1, len(sorted_scores_cum) + 1) / len(sorted_scores_cum)
            ax3.plot(sorted_scores_cum, cumulative, 'b-', linewidth=2)
            ax3.set_xlabel('Importance Score')
            ax3.set_ylabel('Cumulative Probability')
            ax3.set_title('Cumulative Distribution')
            ax3.grid(True, alpha=0.3)
            
            # 4. Box plot with statistics
            ax4.boxplot(scores, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax4.set_ylabel('Importance Score')
            ax4.set_title('Statistical Summary')
            ax4.grid(True, alpha=0.3)
            
            # Add statistics text
            analysis = self.analyze_importance_distribution(importance_scores)
            stats_text = f"""Statistics:
Mean: {analysis['mean']:.4f}
Std: {analysis['std']:.4f}
Entropy: {analysis['entropy']:.4f}
Concentration: {analysis['concentration_ratio']:.4f}
Gini: {analysis['gini_coefficient']:.4f}
Critical gates: {analysis['critical_gates_count']}"""
            
            ax4.text(1.1, 0.5, stats_text, transform=ax4.transAxes, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None

# Example usage and test
if __name__ == "__main__":
    
    # Create test circuit
    qc = QuantumCircuit(4)
    
    # Add gates with different expected importance
    qc.ry(0.001, 0)  # Low importance (small rotation)
    qc.cx(0, 1)      # High importance (entangling)
    qc.ry(np.pi/2, 2)  # Medium importance (significant rotation)
    qc.rz(0.01, 3)   # Low importance (small rotation)
    qc.cx(1, 2)      # High importance (entangling)
    qc.ry(0.002, 1)  # Low importance (small rotation)
    
    print("Test Circuit:")
    print(qc)
    
    # Analyze importance
    analyzer = GateImportanceAnalyzer()
    importance_scores = analyzer.compute_gate_importance(qc)
    
    print(f"\nGate Importance Scores:")
    for i, score in enumerate(importance_scores):
        gate_name = qc.data[i].operation.name
        print(f"  Gate {i} ({gate_name}): {score:.6f}")
    
    # Analyze distribution
    analysis = analyzer.analyze_importance_distribution(importance_scores)
    print(f"\nDistribution Analysis:")
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}")
    
    # Visualize (optional, requires matplotlib)
    try:
        fig = analyzer.visualize_importance_distribution(importance_scores, "Test Circuit")
        plt.show()
    except Exception as e:
        print(f"Could not create visualization: {e}")
