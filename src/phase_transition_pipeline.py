#!/usr/bin/env python3

"""
Two-Stage Phase Transition Pipeline
Stage 1: Find phase transition point
Stage 2: Run enhanced_experiment.py at that point
"""

import numpy as np
import time
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

# Import existing modules
from unified_quantum_causal_poc import (
    ExperimentConfig, CircuitConfig, UnifiedQuantumCausalAnalyzer
)
from enhanced_experiment import EnhancedQuantumExperiment

class PhaseTransitionPipeline:
    """
    Two-stage pipeline for phase transition experiments
    """
    
    def __init__(self, n_qubits, search_ranges=None, log_to_file=True):
        self.n_qubits = n_qubits
        
        # Default search ranges
        self.search_ranges = search_ranges or {
            'depth_factor': {'min': 1.5, 'max': 5.0, 'step': 0.5},
            'redundancy_rate': {'min': 0.05, 'max': 0.45, 'step': 0.05},
            'compression_ratio': {'min': 0.05, 'max': 0.60, 'step': 0.02}
        }
        
        # Setup logging
        self.log_to_file = log_to_file
        self.setup_logging()
        
        # Base configuration
        self.base_config = ExperimentConfig(
            seed=42,
            use_multiprocessing=False,
            use_light_cone=False,
            use_sampling=False
        )
        
        self.logger.info(f"Phase Transition Pipeline initialized for {n_qubits} qubits")
        self.logger.info(f"Will automatically detect natural phase transition")
        self.logger.info(f"Search ranges: {self.search_ranges}")
    
    def setup_logging(self):
        """Setup logging to file and console"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.log_to_file:
            # Create logs directory
            log_dir = Path("pipeline_logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"phase_transition_{self.n_qubits}qubit_{timestamp}.log"
        
        # Setup logger
        self.logger = logging.getLogger(f'PhaseTransition_{self.n_qubits}q')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_to_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging to file: {log_file}")
    
    def stage1_find_transition_point(self, n_probe_circuits=30):
        """
        Stage 1: Find the natural phase transition point
        Returns optimal parameters showing clearest bimodal separation
        """
        
        self.logger.info("="*60)
        self.logger.info("STAGE 1: FINDING NATURAL PHASE TRANSITION POINT")
        self.logger.info("="*60)
        
        # Get initial parameter estimates
        initial_params = self._get_initial_params()
        self.logger.info(f"Initial parameter estimates: {initial_params}")
        
        # Find natural transition point
        optimal_params = self._find_natural_transition(
            initial_params, n_probe_circuits
        )
        
        # Validate the found parameters
        #validation_result = self._validate_parameters(
        #    optimal_params, n_circuits=50
        #)
        
        self.logger.info("=== STAGE 1 RESULTS ===")
        self.logger.info(f"Optimal depth_factor: {optimal_params['depth_factor']:.2f}")
        self.logger.info(f"Optimal redundancy_rate: {optimal_params['redundancy_rate']:.3f}")
        self.logger.info(f"Optimal compression_ratio: {optimal_params['compression_ratio']:.3f}")
        #self.logger.info(f"Natural fragile rate: {validation_result['fragile_rate']:.3f}")
        #self.logger.info(f"Bimodal gap: {validation_result['gap']:.3f}")
        #self.logger.info(f"Separation quality: {validation_result['quality']:.3f}")
        
        # Save Stage 1 results
        #self._save_stage1_results(optimal_params, validation_result)
        self._save_stage1_results(optimal_params, None)
        
        return optimal_params
    
    def stage2_run_full_experiment(self, optimal_params):
        """
        Stage 2: Run enhanced_experiment.py with optimal parameters
        """
        
        self.logger.info("="*60)
        self.logger.info("STAGE 2: RUNNING FULL EXPERIMENT (100 CIRCUITS)")
        self.logger.info("="*60)
        
        # Create modified enhanced experiment
        experiment = self._create_enhanced_experiment(optimal_params)
        
        # Run the complete experiment
        self.logger.info("Starting enhanced experiment with optimized parameters...")
        start_time = time.time()
        
        all_results, analysis_results = experiment.run_complete_experiment()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Stage 2 completed in {elapsed_time:.1f} seconds")
        
        # Save Stage 2 results
        self._save_stage2_results(all_results, analysis_results, optimal_params)
        
        return all_results, analysis_results
    
    def run_complete_pipeline(self):
        """
        Run the complete two-stage pipeline
        """
        
        self.logger.info("="*60)
        self.logger.info("COMPLETE PHASE TRANSITION PIPELINE")
        self.logger.info(f"Target: {self.n_qubits} qubits, natural phase transition detection")
        self.logger.info("="*60)
        
        # Stage 1: Find natural transition point
        optimal_params = self.stage1_find_transition_point()
        
        # Stage 2: Run full experiment
        all_results, analysis_results = self.stage2_run_full_experiment(optimal_params)
        
        # Summary
        robust_count = len([r for r in all_results if r['robustness_class'] == 'robust'])
        fragile_count = len([r for r in all_results if r['robustness_class'] == 'fragile'])
        
        self.logger.info("="*60)
        self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*60)
        self.logger.info(f"Final results: {robust_count} robust, {fragile_count} fragile")
        self.logger.info(f"Natural fragile rate: {fragile_count/100:.3f}")
        self.logger.info(f"Parameters used: {optimal_params}")
        
        return {
            'optimal_parameters': optimal_params,
            'experiment_results': all_results,
            'analysis_results': analysis_results,
            'final_stats': {
                'robust_count': robust_count,
                'fragile_count': fragile_count,
                'fragile_rate': fragile_count / 100
            }
        }
    
    def _get_initial_params(self):
        """Get initial parameter estimates based on qubit count"""
        
        # Flexible scaling for any qubit count
        if self.n_qubits <= 6:
            base_params = {
                'depth_factor': 1.8,
                'redundancy_rate': 0.35,
                'compression_ratio': 0.12
            }
        elif self.n_qubits <= 8:
            base_params = {
                'depth_factor': 2.0,
                'redundancy_rate': 0.30,
                'compression_ratio': 0.15
            }
        elif self.n_qubits <= 10:
            base_params = {
                'depth_factor': 2.3,
                'redundancy_rate': 0.28,
                'compression_ratio': 0.18
            }
        elif self.n_qubits <= 12:
            base_params = {
                'depth_factor': 2.5,
                'redundancy_rate': 0.25,
                'compression_ratio': 0.20
            }
        elif self.n_qubits <= 14:
            base_params = {
                'depth_factor': 3.0,
                'redundancy_rate': 0.20,
                'compression_ratio': 0.25
            }
        elif self.n_qubits <= 16:
            base_params = {
                'depth_factor': 3.5,
                'redundancy_rate': 0.15,
                'compression_ratio': 0.30
            }
        elif self.n_qubits <= 18:
            base_params = {
                'depth_factor': 4.0,
                'redundancy_rate': 0.12,
                'compression_ratio': 0.35
            }
        elif self.n_qubits <= 20:
            base_params = {
                'depth_factor': 4.5,
                'redundancy_rate': 0.10,
                'compression_ratio': 0.40
            }
        else:
            # For very large qubit counts (>20)
            base_params = {
                'depth_factor': 5.0,
                'redundancy_rate': 0.08,
                'compression_ratio': 0.45
            }
        
        self.logger.info(f"Initial parameter estimates for {self.n_qubits} qubits: {base_params}")
        return base_params
    
    def _find_natural_transition(self, initial_params, n_probe_circuits):
        """
        Find natural phase transition point by detecting clearest bimodal separation
        around 9% compression level (similar to 12-qubit findings)
        """
        
        self.logger.info("Searching for natural phase transition around 9% compression level...")
        
        depth_factor = initial_params['depth_factor']
        redundancy_rate = initial_params['redundancy_rate']
        
        # Focus search around 9% compression (like 12-qubit results)
        # Test range: 5% to 20% compression
        compression_ratios = np.arange(0.05, 0.40, 0.03)  # 5%-25%, 1% steps
        
        best_separation = None
        best_quality = 0
        
        self.logger.info(f"Testing {len(compression_ratios)} compression ratios around 9% level...")
        
        for i, ratio in enumerate(compression_ratios):
            test_params = {
                'depth_factor': depth_factor,
                'redundancy_rate': redundancy_rate,
                'compression_ratio': ratio
            }
            
            # Test with probe circuits
            separation_result = self._evaluate_separation_quality_9pct_style(test_params, n_probe_circuits)
            
            if separation_result['quality'] > best_quality:
                best_quality = separation_result['quality']
                best_separation = {
                    **test_params,
                    'separation_info': separation_result
                }
            
            if (i + 1) % 5 == 0:
                self.logger.info(f"  Tested {i + 1}/{len(compression_ratios)}, "
                               f"best quality so far: {best_quality:.3f}")
        
        if best_separation is None:
            raise ValueError("No clear bimodal separation found in search range")
        
        self.logger.info(f"Best separation found at compression_ratio={best_separation['compression_ratio']:.3f}")
        self.logger.info(f"Separation quality: {best_quality:.3f}")
        
        # Return just the parameters (without separation_info)
        return {
            'depth_factor': best_separation['depth_factor'],
            'redundancy_rate': best_separation['redundancy_rate'],
            'compression_ratio': best_separation['compression_ratio']
        }
    
    def _evaluate_separation_quality_9pct_style(self, params, n_circuits):
        """
        Evaluate separation quality optimized for 9%-style results
        (looking for clear separation with minority fragile group)
        """
        
        fidelities = []
        
        # Create circuit config
        circuit_config = CircuitConfig(
            n_qubits=self.n_qubits,
            depth_factor=params['depth_factor'],
            redundancy_rate=params['redundancy_rate']
        )
        
        # Create analyzer
        analyzer = UnifiedQuantumCausalAnalyzer(self.base_config)
        
        for i in range(n_circuits):
            try:
                # Generate circuit
                circuit = analyzer.create_test_circuit(circuit_config, trial_seed=i)
                
                # Test compression
                _, fidelity = analyzer.prune_circuit(
                    circuit, params['compression_ratio'], 'causal'
                )
                
                fidelities.append(fidelity)
                
            except Exception as e:
                self.logger.warning(f"Error in circuit {i}: {e}")
                continue
        
        if len(fidelities) < 10:
            return {'quality': 0, 'gap': 0, 'robust_count': 0, 'fragile_count': 0, 'fragile_rate': 0}
        
        # Analyze separation
        robust_fidelities = [f for f in fidelities if f >= 0.9]
        fragile_fidelities = [f for f in fidelities if f < 0.9]
        
        robust_count = len(robust_fidelities)
        fragile_count = len(fragile_fidelities)
        
        ## # Calculate separation quality optimized for 9%-style (minority fragile)
        ## if robust_count >= 5 and fragile_count >= 2:  # Need clear majority robust, some fragile
        ##     gap = min(robust_fidelities) - max(fragile_fidelities)
        ##     
        ##     # Quality metric optimized for 9%-style separation
        ##     # Prefer clear gap + reasonable fragile minority (10-30%)
        ##     fragile_rate = fragile_count / len(fidelities)
        ##     
        ##     # Penalty function: prefer fragile rate around 10-30%
        ##     if fragile_rate < 0.05:
        ##         rate_factor = fragile_rate / 0.05  # Too few fragile
        ##     elif fragile_rate <= 0.30:
        ##         rate_factor = 1.0  # Good range
        ##     else:
        ##         rate_factor = max(0.3, 1.0 - (fragile_rate - 0.30) * 2)  # Too many fragile
        ##     
        ##     quality = gap * rate_factor * (len(fidelities) / n_circuits)
        ##     
        ## else:
        ##     # No clear separation or insufficient samples
        ##     gap = 0
        ##     quality = 0
        ##     fragile_rate = fragile_count / len(fidelities) if fidelities else 0
        ## 

        # Simplified separation quality: just need any fragile circuits
        if fragile_count >= 1 and robust_count >= 1:  
            gap = min(robust_fidelities) - max(fragile_fidelities)
            quality = gap  
        else:
            # No fragile circuits found
            gap = 0
            quality = 0
        
        fragile_rate = fragile_count / len(fidelities) if fidelities else 0


        return {
            'quality': quality,
            'gap': gap,
            'robust_count': robust_count,
            'fragile_count': fragile_count,
            'fragile_rate': fragile_rate
        }
    
    def _test_parameter_set(self, params, n_circuits):
        """Test a parameter set and return fragile rate"""
        
        fragile_count = 0
        
        # Create circuit config
        circuit_config = CircuitConfig(
            n_qubits=self.n_qubits,
            depth_factor=params['depth_factor'],
            redundancy_rate=params['redundancy_rate']
        )
        
        # Create analyzer
        analyzer = UnifiedQuantumCausalAnalyzer(self.base_config)
        
        for i in range(n_circuits):
            try:
                # Generate circuit
                circuit = analyzer.create_test_circuit(circuit_config, trial_seed=i)
                
                # Test compression
                _, fidelity = analyzer.prune_circuit(
                    circuit, params['compression_ratio'], 'causal'
                )
                
                if fidelity < 0.9:
                    fragile_count += 1
                    
            except Exception as e:
                print(f"    Warning: Error in circuit {i}: {e}")
                continue
        
        return fragile_count / n_circuits
    
    def _validate_parameters(self, params, n_circuits):
        """Validate the found parameters with more circuits"""
        
        self.logger.info(f"Validating parameters with {n_circuits} circuits...")
        validation_result = self._evaluate_separation_quality_9pct_style(params, n_circuits)
        
        self.logger.info(f"Validation results:")
        self.logger.info(f"  Fragile rate: {validation_result['fragile_rate']:.3f}")
        self.logger.info(f"  Gap: {validation_result['gap']:.3f}")
        self.logger.info(f"  Quality: {validation_result['quality']:.3f}")
        
        return validation_result
    
    def _save_stage1_results(self, optimal_params, validation_result):
        """Save Stage 1 results to file"""
        if not self.log_to_file:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path("pipeline_results")
        results_dir.mkdir(exist_ok=True)
        
        stage1_results = {
            'timestamp': timestamp,
            'n_qubits': self.n_qubits,
            'search_ranges': self.search_ranges,
            'optimal_parameters': optimal_params
        }
        
        # update only when validation_result is present
        if validation_result is not None:
            stage1_results.update({
                'validation_results': validation_result,
                'natural_fragile_rate': validation_result['fragile_rate'],
                'bimodal_gap': validation_result['gap'],
                'separation_quality': validation_result['quality']
            })
        else:
            # mark that validation is omitted
            stage1_results.update({
                'validation_results': None,
                'note': 'Validation stage skipped for efficiency'
            })
    

        filename = results_dir / f"stage1_results_{self.n_qubits}qubit_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(stage1_results, f, indent=2)
        
        self.logger.info(f"Stage 1 results saved to: {filename}")
    
    def _save_stage2_results(self, all_results, analysis_results, optimal_params):
        """Save Stage 2 results summary"""
        if not self.log_to_file:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path("pipeline_results")
        results_dir.mkdir(exist_ok=True)
        
        # Calculate final statistics
        robust_count = len([r for r in all_results if r['robustness_class'] == 'robust'])
        fragile_count = len([r for r in all_results if r['robustness_class'] == 'fragile'])
        
        stage2_summary = {
            'timestamp': timestamp,
            'n_qubits': self.n_qubits,
            'optimal_parameters_used': optimal_params,
            'final_statistics': {
                'robust_count': robust_count,
                'fragile_count': fragile_count,
                'final_fragile_rate': fragile_count / 100
            },
            'experiment_completed': True
        }
        
        filename = results_dir / f"stage2_summary_{self.n_qubits}qubit_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(stage2_summary, f, indent=2)
        
        self.logger.info(f"Stage 2 summary saved to: {filename}")
    
    def _create_enhanced_experiment(self, optimal_params):
        """Create enhanced experiment with modified parameters"""
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"phase_transition_{self.n_qubits}qubit_{timestamp}"
        
        # Create modified enhanced experiment
        experiment = ModifiedEnhancedExperiment(
            n_qubits=self.n_qubits,
            n_circuits=100,
            output_dir=output_dir,
            optimal_params=optimal_params
        )
        
        return experiment


class ModifiedEnhancedExperiment(EnhancedQuantumExperiment):
    """
    Enhanced experiment modified to use optimal parameters
    """
    
    def __init__(self, n_qubits, n_circuits, output_dir, optimal_params):
        # Initialize parent class
        super().__init__(n_qubits, n_circuits, output_dir)
        
        # Store optimal parameters
        self.optimal_params = optimal_params
        self.compression_ratio = optimal_params['compression_ratio']
        
        print(f"Modified Enhanced Experiment initialized")
        print(f"Using optimal parameters: {optimal_params}")
    
    def auto_tune_params(self):
        """Override to use pre-found optimal parameters"""
        
        # Use the optimal parameters from Stage 1
        depth_factor = self.optimal_params['depth_factor']
        redundancy_rate = self.optimal_params['redundancy_rate']
        
        # Calculate estimated gates
        gate_per_qubit = 65  # Keep this consistent
        estimated_gates = int(self.n_qubits * gate_per_qubit)
        
        return estimated_gates, depth_factor, redundancy_rate
    
    def analyze_single_circuit(self, circuit, seed, circuit_idx):
        """Override to use optimal compression ratio"""
        
        # Get basic circuit info (same as parent)
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
        
        # Use optimal compression ratio from Stage 1
        compression_ratio = self.optimal_params['compression_ratio']
        print(f"  Testing compression at {compression_ratio*100:.1f}%...")
        
        pruned_circuit, fidelity = self.quantum_analyzer.prune_circuit(circuit, compression_ratio, 'causal')
        
        # Classify robustness
        robustness_class = 'robust' if fidelity >= 0.9 else 'fragile'
        
        print(f"  Result: {robustness_class} (F = {fidelity:.4f})")
        
        # Save circuit structure
        circuit_structure = self.circuit_to_dict(circuit)
        
        return {
            'circuit_info': circuit_info,
            'importance_scores': importance_scores,
            'importance_analysis': importance_analysis,
            'compression_ratio': compression_ratio,
            'fidelity': fidelity,
            'robustness_class': robustness_class,
            'circuit_structure': circuit_structure,
            'optimal_params_used': self.optimal_params
        }


# Main execution functions
def run_pipeline_for_qubits(n_qubits, search_ranges=None, log_to_file=True):
    """
    Run complete pipeline for specified number of qubits
    
    Args:
        n_qubits: Number of qubits (any positive integer)
        search_ranges: Custom search ranges dict (optional)
        log_to_file: Whether to log to file (default: True)
    
    Example search_ranges:
    {
        'depth_factor': {'min': 1.0, 'max': 5.0, 'step': 0.5},
        'redundancy_rate': {'min': 0.05, 'max': 0.50, 'step': 0.05},
        'compression_ratio': {'min': 0.05, 'max': 0.70, 'step': 0.02}
    }
    """
    
    pipeline = PhaseTransitionPipeline(
        n_qubits=n_qubits,
        search_ranges=search_ranges,
        log_to_file=log_to_file
    )
    results = pipeline.run_complete_pipeline()
    
    return results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Phase Transition Pipeline for Quantum Circuits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase_transition_pipeline.py --n_qubit 14
  python phase_transition_pipeline.py --n_qubit 16 --no_log_file
  python phase_transition_pipeline.py --n_qubit 12 --compression_max 0.50
        """
    )
    
    parser.add_argument(
        '--n_qubit', type=int, default=14,
        help='Number of qubits (default: 14)'
    )
    
    parser.add_argument(
        '--no_log_file', action='store_true',
        help='Disable file logging (only console output)'
    )
    
    parser.add_argument(
        '--depth_min', type=float, default=1.5,
        help='Minimum depth factor for search (default: 1.5)'
    )
    
    parser.add_argument(
        '--depth_max', type=float, default=5.0,
        help='Maximum depth factor for search (default: 5.0)'
    )
    
    parser.add_argument(
        '--redundancy_min', type=float, default=0.05,
        help='Minimum redundancy rate for search (default: 0.05)'
    )
    
    parser.add_argument(
        '--redundancy_max', type=float, default=0.45,
        help='Maximum redundancy rate for search (default: 0.45)'
    )
    
    parser.add_argument(
        '--compression_min', type=float, default=0.05,
        help='Minimum compression ratio for search (default: 0.05)'
    )
    
    parser.add_argument(
        '--compression_max', type=float, default=0.60,
        help='Maximum compression ratio for search (default: 0.60)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    print("Phase Transition Pipeline")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  Number of qubits: {args.n_qubit}")
    print(f"  Natural phase transition detection: Enabled")
    print(f"  File logging: {'Disabled' if args.no_log_file else 'Enabled'}")
    
    # Create custom search ranges from arguments
    search_ranges = {
        'depth_factor': {
            'min': args.depth_min, 
            'max': args.depth_max, 
            'step': 0.5
        },
        'redundancy_rate': {
            'min': args.redundancy_min, 
            'max': args.redundancy_max, 
            'step': 0.05
        },
        'compression_ratio': {
            'min': args.compression_min, 
            'max': args.compression_max, 
            'step': 0.02
        }
    }
    
    print(f"\nSearch ranges:")
    print(f"  Depth factor: {args.depth_min:.1f} - {args.depth_max:.1f}")
    print(f"  Redundancy rate: {args.redundancy_min:.2f} - {args.redundancy_max:.2f}")
    print(f"  Compression ratio: {args.compression_min:.2f} - {args.compression_max:.2f}")
    
    # Run the pipeline
    try:
        results = run_pipeline_for_qubits(
            n_qubits=args.n_qubit,
            search_ranges=search_ranges,
            log_to_file=not args.no_log_file
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Optimal parameters found:")
        for key, value in results['optimal_parameters'].items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nFinal results:")
        print(f"  Robust circuits: {results['final_stats']['robust_count']}")
        print(f"  Fragile circuits: {results['final_stats']['fragile_count']}")
        print(f"  Natural fragile rate: {results['final_stats']['fragile_rate']:.3f}")
        
        if not args.no_log_file:
            print(f"\nLog files and results saved to:")
            print(f"  - pipeline_logs/ (detailed logs)")
            print(f"  - pipeline_results/ (JSON results)")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
