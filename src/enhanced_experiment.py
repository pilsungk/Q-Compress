#!/usr/bin/env python3

"""
Enhanced Quantum Circuit Analysis with Complete Circuit Storage
Compatible with Qiskit 1.4+
FINAL VERSION - All qasm() calls removed
"""

import os
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

# Import your existing modules
from unified_quantum_causal_poc import (
    ExperimentConfig, CircuitConfig, UnifiedQuantumCausalAnalyzer
)

# Import the new analysis functions we created
from gate_importance_analyzer import GateImportanceAnalyzer

# Qiskit 1.4 compatibility
try:
    from qiskit.qasm2 import dumps as qasm_dumps
    QASM_AVAILABLE = True
except ImportError:
    QASM_AVAILABLE = False
    print("Warning: qasm2 not available, will save circuit structure only")

class EnhancedQuantumExperiment:
    """
    Enhanced experiment class that saves all circuits and performs detailed analysis
    """
    
    def __init__(self, n_qubits=12, n_circuits=100, output_dir="enhanced_experiment", compression_ratio=0.09):
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression_ratio = compression_ratio  
        
        # Create subdirectories
        (self.output_dir / "circuits").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.exp_config = ExperimentConfig(
            seed=42,
            use_multiprocessing=False,  # For stability
            use_light_cone=False,
            use_sampling=False
        )
        
        self.quantum_analyzer = UnifiedQuantumCausalAnalyzer(self.exp_config)
        self.importance_analyzer = GateImportanceAnalyzer()
        
        print(f"Enhanced Quantum Experiment initialized")
        print(f"Target: {n_qubits} qubits, {n_circuits} circuits")
        print(f"Output directory: {self.output_dir}")
    
    def run_complete_experiment(self):
        """
        Run the complete experiment:
        1. Generate 100 circuits
        2. Analyze each at 9% compression
        3. Save detailed information for all circuits
        4. Perform hypothesis testing
        """
        
        print("\n=== STARTING COMPLETE EXPERIMENT ===")
        start_time = time.time()
        
        # Auto-tune parameters
        estimated_gates, depth_factor, redundancy_rate = self.auto_tune_params()
        circuit_config = CircuitConfig(
            n_qubits=self.n_qubits,
            depth_factor=depth_factor,
            redundancy_rate=redundancy_rate
        )
        
        print(f"Circuit parameters: gates~{estimated_gates}, depth_factor={depth_factor}, redundancy_rate={redundancy_rate}, compression_ratio={self.compression_ratio}")
        
        all_results = []
        robust_count = 0
        fragile_count = 0
        
        # Generate and analyze all circuits
        for circuit_idx in range(self.n_circuits):
            print(f"\nProcessing circuit {circuit_idx + 1}/{self.n_circuits}")
            
            # Generate unique seed
            circuit_seed = 42 + circuit_idx * 1000
            
            # Generate circuit
            circuit = self.quantum_analyzer.create_test_circuit(circuit_config, trial_seed=circuit_idx)
            
            # Analyze circuit at 9% compression
            detailed_result = self.analyze_single_circuit(circuit, circuit_seed, circuit_idx)
            all_results.append(detailed_result)
            
            # Count robustness
            if detailed_result['robustness_class'] == 'robust':
                robust_count += 1
            else:
                fragile_count += 1
            
            # Save individual circuit
            self.save_single_circuit(detailed_result, circuit_idx)
            
            # Progress update
            if (circuit_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {circuit_idx + 1}/{self.n_circuits}, "
                      f"Robust: {robust_count}, Fragile: {fragile_count}, "
                      f"Time: {elapsed:.1f}s")
        
        # Save complete results
        self.save_complete_results(all_results)
        
        # Perform analysis
        analysis_results = self.perform_hypothesis_testing(all_results)
        
        # Generate report
        self.generate_final_report(all_results, analysis_results)
        
        total_time = time.time() - start_time
        print(f"\n=== EXPERIMENT COMPLETE ===")
        print(f"Total time: {total_time:.1f}s")
        print(f"Final count: {robust_count} robust, {fragile_count} fragile")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results, analysis_results
    
    def auto_tune_params(self):
        """Auto-tune parameters based on qubit count"""
        if self.n_qubits <= 12:
            gate_per_qubit = 65
            depth_factor = 2.5
            redundancy_rate = 0.25
        elif self.n_qubits <= 14:
            gate_per_qubit = 65
            depth_factor = 2.5 
            redundancy_rate = 0.15
            #depth_factor = 2.5 
            #redundancy_rate = 0.25
            #depth_factor = 3.0   : 100% robust when depth_factor=0.3, redundancy_rate=0.3
            #redundancy_rate = 0.3
        elif self.n_qubits <= 16:  
            gate_per_qubit = 65
            depth_factor = 2.8
            redundancy_rate = 0.2
        else:
            gate_per_qubit = 65
            depth_factor = 3.0
            redundancy_rate = 0.12
        
        estimated_gates = int(self.n_qubits * gate_per_qubit)
        return estimated_gates, depth_factor, redundancy_rate
    
    def circuit_to_dict(self, circuit):
        """Convert circuit to serializable dictionary"""
        gates_data = []
        for i, instruction in enumerate(circuit.data):
            gate_info = {
                'index': i,
                'name': instruction.operation.name,
                'qubits': [circuit.qubits.index(q) for q in instruction.qubits],
                'params': list(instruction.operation.params) if hasattr(instruction.operation, 'params') else []
            }
            gates_data.append(gate_info)
        
        return {
            'n_qubits': circuit.num_qubits,
            'gates': gates_data
        }
    
    def analyze_single_circuit(self, circuit, seed, circuit_idx):
        """Perform detailed analysis on a single circuit"""
        
        # Basic circuit info
        gate_counts = {}
        two_qubit_count = 0
        for gate in circuit.data:
            gate_name = gate.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            if len(gate.qubits) == 2:
                two_qubit_count += 1
        
        circuit_info = {
            'circuit_index': circuit_idx,
            'seed': seed,
            'n_qubits': circuit.num_qubits,
            'total_gates': len(circuit.data),
            'depth': circuit.depth(),
            'gate_distribution': gate_counts,
            'entanglement_ratio': two_qubit_count / len(circuit.data) if len(circuit.data) > 0 else 0
        }
        
        # Compute gate importance
        print(f"  Computing gate importance...")
        importance_scores = self.importance_analyzer.compute_gate_importance(circuit)
        importance_analysis = self.importance_analyzer.analyze_importance_distribution(importance_scores)
        
        # Perform compression at 9%
        print(f"  Testing compression at 9%...")
        compression_ratio = self.compression_ratio
        print(f"  Testing compression at {compression_ratio*100:.1f}%...")
        pruned_circuit, fidelity = self.quantum_analyzer.prune_circuit(circuit, compression_ratio, 'causal')
        
        # Classify robustness
        robustness_class = 'robust' if fidelity >= 0.9 else 'fragile'
        
        print(f"  Result: {robustness_class} (F = {fidelity:.4f})")
        
        # Save circuit structure (no QASM to avoid compatibility issues)
        circuit_structure = self.circuit_to_dict(circuit)
        
        # Optionally try to save QASM if available
        circuit_qasm = None
        if QASM_AVAILABLE:
            try:
                circuit_qasm = qasm_dumps(circuit)
            except Exception as e:
                print(f"  Note: Could not save QASM: {e}")
        
        return {
            'circuit_info': circuit_info,
            'importance_scores': importance_scores,
            'importance_analysis': importance_analysis,
            'compression_ratio': compression_ratio,
            'fidelity': fidelity,
            'robustness_class': robustness_class,
            'circuit_structure': circuit_structure,
            'circuit_qasm': circuit_qasm
        }
    
    def save_single_circuit(self, result, circuit_idx):
        """Save individual circuit result"""
        filename = f"circuit_{circuit_idx:03d}_{result['robustness_class']}.json"
        filepath = self.output_dir / "circuits" / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def save_complete_results(self, all_results):
        """Save complete results dataset"""
        
        # Save full detailed results
        detailed_file = self.output_dir / "complete_detailed_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for result in all_results:
            summary_data.append({
                'circuit_index': result['circuit_info']['circuit_index'],
                'seed': result['circuit_info']['seed'],
                'compression_ratio': result['compression_ratio'],
                'fidelity': result['fidelity'],
                'robustness_class': result['robustness_class'],
                'total_gates': result['circuit_info']['total_gates'],
                'depth': result['circuit_info']['depth'],
                'entanglement_ratio': result['circuit_info']['entanglement_ratio'],
                'importance_entropy': result['importance_analysis']['entropy'],
                'importance_concentration': result['importance_analysis']['concentration_ratio']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "experiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\nComplete results saved to {detailed_file}")
        print(f"Summary CSV saved to {summary_file}")
    
    def perform_hypothesis_testing(self, all_results):
        """Perform statistical hypothesis testing"""
        
        print("\n=== HYPOTHESIS TESTING ===")
        
        # Separate robust and fragile circuits
        robust_circuits = [r for r in all_results if r['robustness_class'] == 'robust']
        fragile_circuits = [r for r in all_results if r['robustness_class'] == 'fragile']
        
        print(f"Robust circuits: {len(robust_circuits)}")
        print(f"Fragile circuits: {len(fragile_circuits)}")
        
        if len(robust_circuits) == 0 or len(fragile_circuits) == 0:
            print("Warning: No clear bimodal separation found!")
            return None
        
        # Extract importance scores for comparison
        robust_importance_scores = [r['importance_scores'] for r in robust_circuits]
        fragile_importance_scores = [r['importance_scores'] for r in fragile_circuits]
        
        # Perform comparison
        comparison_results = self.importance_analyzer.compare_robust_vs_fragile_importance(
            robust_importance_scores, fragile_importance_scores
        )
        
        # Save hypothesis test results
        hypothesis_file = self.output_dir / "analysis" / "hypothesis_test_results.json"
        with open(hypothesis_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        print(f"Hypothesis test results saved to {hypothesis_file}")
        
        return comparison_results
    
    def generate_final_report(self, all_results, analysis_results):
        """Generate final experiment report"""
        
        robust_count = len([r for r in all_results if r['robustness_class'] == 'robust'])
        fragile_count = len([r for r in all_results if r['robustness_class'] == 'fragile'])
        
        # Calculate key statistics
        all_fidelities = [r['fidelity'] for r in all_results]
        robust_fidelities = [r['fidelity'] for r in all_results if r['robustness_class'] == 'robust']
        fragile_fidelities = [r['fidelity'] for r in all_results if r['robustness_class'] == 'fragile']
        
        report = {
            'experiment_summary': {
                'n_qubits': self.n_qubits,
                'n_circuits': self.n_circuits,
                'compression_ratio': self.compression_ratio,
                'threshold': 0.9,
                'timestamp': datetime.now().isoformat()
            },
            'bimodal_classification': {
                'robust_count': robust_count,
                'robust_percentage': robust_count / self.n_circuits * 100,
                'fragile_count': fragile_count,
                'fragile_percentage': fragile_count / self.n_circuits * 100
            },
            'fidelity_statistics': {
                'overall_mean': float(np.mean(all_fidelities)),
                'overall_std': float(np.std(all_fidelities)),
                'robust_mean': float(np.mean(robust_fidelities)) if robust_fidelities else 0,
                'robust_std': float(np.std(robust_fidelities)) if robust_fidelities else 0,
                'fragile_mean': float(np.mean(fragile_fidelities)) if fragile_fidelities else 0,
                'fragile_std': float(np.std(fragile_fidelities)) if fragile_fidelities else 0
            }
        }
        
        if analysis_results:
            report['hypothesis_test'] = analysis_results
        
        # Calculate gap if both groups exist
        if robust_fidelities and fragile_fidelities:
            gap = min(robust_fidelities) - max(fragile_fidelities)
            report['bimodal_separation'] = {
                'gap': float(gap),
                'min_robust': float(min(robust_fidelities)),
                'max_fragile': float(max(fragile_fidelities))
            }
        
        # Save report
        report_file = self.output_dir / "final_experiment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n=== FINAL REPORT ===")
        print(f"Classification: {robust_count} robust ({robust_count/self.n_circuits*100:.1f}%), "
              f"{fragile_count} fragile ({fragile_count/self.n_circuits*100:.1f}%)")
        print(f"Overall fidelity: {np.mean(all_fidelities):.4f} ± {np.std(all_fidelities):.4f}")
        if robust_fidelities and fragile_fidelities:
            gap = min(robust_fidelities) - max(fragile_fidelities)
            print(f"Gap between groups: {gap:.4f}")
            print(f"Min robust: {min(robust_fidelities):.4f}, Max fragile: {max(fragile_fidelities):.4f}")
        
        print(f"Report saved to {report_file}")


# Main execution function
def run_enhanced_experiment(compression_ratio=0.09):  # parameter added
    """
    Main function to run the enhanced experiment
    """
    
    # Configuration
    N_QUBITS = 14
    N_CIRCUITS = 100
    OUTPUT_DIR = f"enhanced_{N_QUBITS}qubit_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create and run experiment
    experiment = EnhancedQuantumExperiment(
        n_qubits=N_QUBITS,
        n_circuits=N_CIRCUITS,
        output_dir=OUTPUT_DIR
    )
    
    # Run complete experiment
    all_results, analysis_results = experiment.run_complete_experiment()
    
    return all_results, analysis_results


if __name__ == "__main__":
    print("Enhanced Quantum Circuit Analysis - Final Version")
    print("=" * 50)
    
    # Check dependencies
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, state_fidelity
        try:
            from scipy import stats
            scipy_available = True
        except ImportError:
            scipy_available = False
            print("Warning: scipy not available, statistical tests will be limited")
        print("✓ Core dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install: pip install qiskit scipy")
        exit(1)
    
    # Run experiment
    print("Starting enhanced quantum circuit analysis...")
    results, analysis = run_enhanced_experiment()
    
    print("\n" + "=" * 50)
    print("Experiment completed successfully!")
    print("Check the output directory for detailed results.")
