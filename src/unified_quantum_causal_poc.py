"""
Unified Quantum Causal Pruning PoC
Final version
- Reproducibility
- Parallelization
- Scalability up to 20+ qubits
- Depth analysis
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Set
import time
import multiprocessing as mp
from functools import lru_cache
import psutil
import gc
import json
import random
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# globals
MAX_QUBITS_FULL_STATEVECTOR = 20
IMPORTANCE_THRESHOLD = 1e-10

# dataclass
@dataclass
class ExperimentConfig:
    """Experiment config"""
    seed: int = 42
    n_jobs: Optional[int] = None
    use_multiprocessing: bool = True
    use_light_cone: bool = False
    use_sampling: bool = True
    sampling_rate: float = 0.3
    

@dataclass
class CircuitConfig:
    """Circuit config"""
    n_qubits: int
    n_layers: Optional[int] = None
    depth_factor: float = 3.0
    redundancy_rate: float = 0.3
    connectivity: str = 'linear'  # 'linear', 'all-to-all', 'grid'


# Worker functions (multiprocessing)
def compute_gate_importance_worker(args):
    """worker function for multiprocessing"""
    n_qubits, circuit_data, gate_idx = args
    
    # circuit construction
    circuit = QuantumCircuit(n_qubits)
    for op, qubits, clbits in circuit_data:
        circuit.append(op, qubits, clbits)
    
    # original state
    original_state = Statevector(circuit)
    
    # modified circuit
    modified_circuit = QuantumCircuit(n_qubits)
    for i, (op, qubits, clbits) in enumerate(circuit_data):
        if i != gate_idx:
            modified_circuit.append(op, qubits, clbits)
    
    # modified state
    modified_state = Statevector(modified_circuit)
    
    # Fidelity calculation
    fidelity = state_fidelity(original_state, modified_state)
    
    if fidelity > 0.9999:
        importance = -np.log10(max(1 - fidelity, 1e-16))
    else:
        importance = 1 - fidelity
    
    return gate_idx, importance


def run_single_trial_worker(args):
    """single trial execution worker"""
    trial_idx, experiment_config, circuit_config, compression_ratio = args
    
    # random seed
    np.random.seed(experiment_config.seed + trial_idx)
    random.seed(experiment_config.seed + trial_idx)
    
    # Analyzer construction (multiprocessing disabled)
    analyzer = UnifiedQuantumCausalAnalyzer(
        ExperimentConfig(
            seed=experiment_config.seed,
            n_jobs=1,
            use_multiprocessing=False
        )
    )
    
    # circuit construction
    circuit = analyzer.create_test_circuit(circuit_config, trial_seed=trial_idx)
    
    # pruning with different methods
    results = {}
    for method in ['random', 'magnitude', 'causal']:
        pruned, fidelity = analyzer.prune_circuit(circuit, compression_ratio, method)
        results[method] = fidelity
    
    return {
        'trial': trial_idx,
        'random': results['random'],
        'magnitude': results['magnitude'],
        'causal': results['causal']
    }


class UnifiedQuantumCausalAnalyzer:
    """Unified quantum causal analyzer"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_jobs = config.n_jobs or max(1, mp.cpu_count() - 1)
        self._set_random_seeds()
        self._check_system_resources()
        
    def _set_random_seeds(self):
        """random seed"""
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        
    def _check_system_resources(self):
        """system resource check"""
        if psutil:
            memory_gb = psutil.virtual_memory().available / (1024**3)
            self.max_safe_qubits = int(np.log2(memory_gb * 1024**3 / 16))
            print(f"System: {mp.cpu_count()} CPUs, {memory_gb:.1f} GB available")
            print(f"Max safe qubits: {self.max_safe_qubits}")
        else:
            self.max_safe_qubits = 16
    
    def create_test_circuit(self, config: CircuitConfig, 
                           trial_seed: Optional[int] = None) -> QuantumCircuit:
        """test circuit construction"""
        if trial_seed is not None:
            rng = np.random.RandomState(self.config.seed + trial_seed)
        else:
            rng = np.random
        
        qc = QuantumCircuit(config.n_qubits)
        
        # set number of layers
        if config.n_layers is None:
            n_layers = int(config.n_qubits * config.depth_factor)
        else:
            n_layers = config.n_layers
        
        for layer in range(n_layers):
            # 1. Rotation layer
            for i in range(config.n_qubits):
                if rng.random() > config.redundancy_rate:
                    # important rotation
                    angle = rng.uniform(np.pi/6, np.pi/2)
                else:
                    # small angle
                    angle = rng.uniform(0.001, 0.05)
                
                gate_type = rng.choice(['ry', 'rz', 'rx'])
                getattr(qc, gate_type)(angle, i)
            
            # 2. Entangling layer
            if layer < n_layers - 1:
                self._add_entangling_layer(qc, config, layer, rng)
            
            # # 3. Barrier
            # if layer < n_layers - 1:
            #     qc.barrier()
        
        # 4. additional redundant gates
        n_redundant = int(config.n_qubits * config.redundancy_rate)
        for _ in range(n_redundant):
            i = rng.randint(0, config.n_qubits)
            qc.rz(rng.uniform(0.001, 0.01), i)
        
        return qc
    
    def _add_entangling_layer(self, qc: QuantumCircuit, config: CircuitConfig, 
                             layer: int, rng):
        """Entangling layer insertion"""
        if config.connectivity == 'linear':
            # Linear connectivity
            if layer % 2 == 0:
                for i in range(0, config.n_qubits - 1, 2):
                    qc.cx(i, i + 1)
            else:
                for i in range(1, config.n_qubits - 1, 2):
                    qc.cx(i, i + 1)
                if config.n_qubits > 2:
                    qc.cx(config.n_qubits - 1, 0)
                    
        elif config.connectivity == 'all-to-all':
            # Sparse all-to-all
            for i in range(config.n_qubits // 2):
                j = (i + config.n_qubits // 2) % config.n_qubits
                if j != i:
                    qc.cx(i, j)
                    
        elif config.connectivity == 'grid':
            # 2D grid-like connectivity
            side = int(np.sqrt(config.n_qubits))
            for i in range(side):
                for j in range(side - 1):
                    idx = i * side + j
                    if idx + 1 < config.n_qubits:
                        qc.cx(idx, idx + 1)
    
    def compute_gate_importance(self, circuit: QuantumCircuit) -> List[float]:
        """Calculate gate importance"""
        n_qubits = circuit.num_qubits
        n_gates = len(circuit.data)
        
        print(f"Computing importance for {n_gates} gates on {n_qubits} qubits...")
        
        # choose strategy
        if n_qubits > 16 and self.config.use_sampling:
            return self._compute_importance_sampling(circuit)
        elif n_qubits > 12 and self.config.use_light_cone:
            return self._compute_importance_light_cone(circuit)
        else:
            return self._compute_importance_full(circuit)
    
    def _compute_importance_full(self, circuit: QuantumCircuit) -> List[float]:
        """Calculate importance using the whole statevector"""
        if not self.config.use_multiprocessing or len(circuit.data) < 10:
            # sequential processing
            return self._compute_importance_sequential(circuit)
        
        # multiprocessing
        n_gates = len(circuit.data)
        importance_scores = [0.0] * n_gates
        
        # serialize circuit
        circuit_data = [(inst.operation, inst.qubits, inst.clbits) 
                       for inst in circuit.data]
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Future creation
            future_to_idx = {}
            for i in range(n_gates):
                args = (circuit.num_qubits, circuit_data, i)
                future = executor.submit(compute_gate_importance_worker, args)
                future_to_idx[future] = i
            
            # collect results
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    importance_scores[idx] = future.result()[1]
                    completed += 1
                    if completed % 10 == 0:
                        print(f"  Processed {completed}/{n_gates} gates...")
                except Exception as e:
                    print(f"Error processing gate {idx}: {e}")
                    importance_scores[idx] = 0.0
        
        return importance_scores
    
    def _compute_importance_sequential(self, circuit: QuantumCircuit) -> List[float]:
        """calculate importance sequentially"""
        importance_scores = []
        original_state = Statevector(circuit)
        
        for i in range(len(circuit.data)):
            # remove gate i 
            modified = QuantumCircuit(circuit.num_qubits)
            for j, inst in enumerate(circuit.data):
                if i != j:
                    modified.append(inst.operation, inst.qubits, inst.clbits)
            
            modified_state = Statevector(modified)
            fidelity = state_fidelity(original_state, modified_state)
            
            if fidelity > 0.9999:
                importance = -np.log10(max(1 - fidelity, 1e-16))
            else:
                importance = 1 - fidelity
            
            importance_scores.append(importance)
        
        return importance_scores
    
    def _compute_importance_light_cone(self, circuit: QuantumCircuit) -> List[float]:
        """calculate importance using light cone """
        print("  Using light cone reduction...")
        importance_scores = []
        
        for i in range(len(circuit.data)):
            # Light cone calculation
            affected_qubits = self._get_light_cone(circuit, i)
            
            if len(affected_qubits) > 10:
                # sample if light cone is too big
                importance = self._estimate_single_gate_importance(circuit, i)
            else:
                # calculate using reduced circuit 
                importance = self._compute_reduced_importance(circuit, i, affected_qubits)
            
            importance_scores.append(importance)
            
            if (i + 1) % 20 == 0:
                print(f"    Processed {i + 1}/{len(circuit.data)} gates...")
        
        return importance_scores
    
    def _compute_importance_sampling(self, circuit: QuantumCircuit) -> List[float]:
        """estimate importance based on sampling"""
        print(f"  Using sampling (rate={self.config.sampling_rate})...")
        n_gates = len(circuit.data)
        sample_size = max(10, int(n_gates * self.config.sampling_rate))
        
        # random sampling
        sampled_indices = np.random.choice(n_gates, sample_size, replace=False)
        importance_dict = {}
        
        for idx in sampled_indices:
            importance = self._compute_single_gate_importance(circuit, idx)
            importance_dict[idx] = importance
        
        # estimate remainders using the average
        avg_importance = np.mean(list(importance_dict.values()))
        
        importance_scores = []
        for i in range(n_gates):
            if i in importance_dict:
                importance_scores.append(importance_dict[i])
            else:
                importance_scores.append(avg_importance * 0.8)  # conservative estimation
        
        return importance_scores
    
    def _get_light_cone(self, circuit: QuantumCircuit, gate_idx: int) -> Set[int]:
        """calculate light cone of a gate"""
        affected_qubits = set()
        
        # start point
        gate_data = circuit.data[gate_idx]
        for qubit in gate_data.qubits:
            affected_qubits.add(circuit.qubits.index(qubit))
        
        # Forward propagation
        for i in range(gate_idx + 1, len(circuit.data)):
            gate = circuit.data[i]
            gate_qubits = {circuit.qubits.index(q) for q in gate.qubits}
            
            if gate_qubits & affected_qubits:
                affected_qubits.update(gate_qubits)
        
        return affected_qubits
    
    def _compute_single_gate_importance(self, circuit: QuantumCircuit, gate_idx: int) -> float:
        """calculate importance of a single gate"""
        try:
            original_state = Statevector(circuit)
            
            # remove gate
            modified = QuantumCircuit(circuit.num_qubits)
            for i, inst in enumerate(circuit.data):
                if i != gate_idx:
                    modified.append(inst.operation, inst.qubits, inst.clbits)
            
            modified_state = Statevector(modified)
            fidelity = state_fidelity(original_state, modified_state)
            
            if fidelity > 0.9999:
                importance = -np.log10(max(1 - fidelity, 1e-16))
            else:
                importance = 1 - fidelity
            
            gc.collect()
            return importance
            
        except Exception as e:
            print(f"Error computing importance for gate {gate_idx}: {e}")
            return 0.0
    
    def _compute_reduced_importance(self, circuit: QuantumCircuit, gate_idx: int, 
                                  affected_qubits: Set[int]) -> float:
        """calculate importance in a reduced circuit"""
        # TODO: approximate since actual implementation is complex 
        return self._compute_single_gate_importance(circuit, gate_idx)
    
    def _estimate_single_gate_importance(self, circuit: QuantumCircuit, gate_idx: int) -> float:
        """Estimate statistically"""
        # heuristic based on gate type
        gate = circuit.data[gate_idx]
        if gate.operation.name in ['cx', 'cz']:
            return 0.1  # Entangling gates are usually important
        elif gate.operation.name in ['rx', 'ry', 'rz']:
            if hasattr(gate.operation, 'params'):
                angle = abs(gate.operation.params[0])
                return angle / np.pi  # proportionate to the angle 
        return 0.05
    
    def prune_circuit(self, circuit: QuantumCircuit, compression_ratio: float, 
                     method: str = 'causal') -> Tuple[QuantumCircuit, float]:
        """circuit pruning"""
        if method == 'random':
            return self._prune_random(circuit, compression_ratio)
        elif method == 'magnitude':
            return self._prune_magnitude(circuit, compression_ratio)
        elif method == 'causal':
            return self._prune_causal(circuit, compression_ratio)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
    
    def _prune_random(self, circuit: QuantumCircuit, compression_ratio: float) -> Tuple[QuantumCircuit, float]:
        """random pruning"""
        n_gates = len(circuit.data)
        n_remove = int(n_gates * compression_ratio)
        
        indices = list(range(n_gates))
        np.random.shuffle(indices)
        gates_to_remove = set(indices[:n_remove])
        
        pruned = QuantumCircuit(circuit.num_qubits)
        for i, inst in enumerate(circuit.data):
            if i not in gates_to_remove:
                pruned.append(inst.operation, inst.qubits, inst.clbits)
        
        # Fidelity calculation
        original_state = Statevector(circuit)
        pruned_state = Statevector(pruned)
        fidelity = state_fidelity(original_state, pruned_state)
        
        return pruned, fidelity
    
    def _prune_magnitude(self, circuit: QuantumCircuit, compression_ratio: float) -> Tuple[QuantumCircuit, float]:
        """magnitude-based pruning"""
        gate_magnitudes = []
        
        for i, inst in enumerate(circuit.data):
            op = inst.operation
            if hasattr(op, 'params') and op.params:
                magnitude = abs(op.params[0])
            else:
                magnitude = 1.0  # Non-parameterized gates
            gate_magnitudes.append((i, magnitude))
        
        # remove small ones first
        gate_magnitudes.sort(key=lambda x: (x[1], x[0]))
        n_remove = int(len(circuit.data) * compression_ratio)
        gates_to_remove = set(idx for idx, _ in gate_magnitudes[:n_remove])
        
        pruned = QuantumCircuit(circuit.num_qubits)
        for i, inst in enumerate(circuit.data):
            if i not in gates_to_remove:
                pruned.append(inst.operation, inst.qubits, inst.clbits)
        
        original_state = Statevector(circuit)
        pruned_state = Statevector(pruned)
        fidelity = state_fidelity(original_state, pruned_state)
        
        return pruned, fidelity
    
    def _prune_causal(self, circuit: QuantumCircuit, compression_ratio: float) -> Tuple[QuantumCircuit, float]:
        """causal pruning"""
        # calculate importance
        importance_scores = self.compute_gate_importance(circuit)
        
        # sort
        indexed_scores = [(i, score) for i, score in enumerate(importance_scores)]
        indexed_scores.sort(key=lambda x: (x[1], x[0]))
        
        # remove
        n_remove = int(len(circuit.data) * compression_ratio)
        gates_to_remove = set(idx for idx, _ in indexed_scores[:n_remove])
        
        pruned = QuantumCircuit(circuit.num_qubits)
        for i, inst in enumerate(circuit.data):
            if i not in gates_to_remove:
                pruned.append(inst.operation, inst.qubits, inst.clbits)
        
        original_state = Statevector(circuit)
        pruned_state = Statevector(pruned)
        fidelity = state_fidelity(original_state, pruned_state)
        
        return pruned, fidelity


class UnifiedExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.analyzer = UnifiedQuantumCausalAnalyzer(config)
        self.results = []
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'system_info': self._get_system_info()
        }
    
    def _get_system_info(self):
        info = {
            'cpu_count': mp.cpu_count(),
            'numpy_version': np.__version__
        }
        if psutil:
            info['total_memory_gb'] = psutil.virtual_memory().total / (1024**3)
        return info
    
    def run_standard_experiment(self, 
                               qubit_range: List[int] = [4, 6, 8],
                               compression_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
                               n_trials: int = 10):
        print("=== Standard Causal Pruning Experiment ===\n")
        
        for n_qubits in qubit_range:
            print(f"\nTesting {n_qubits}-qubit circuits...")
            circuit_config = CircuitConfig(n_qubits=n_qubits)
            
            qubit_results = {
                'n_qubits': n_qubits,
                'compression_results': []
            }
            
            for ratio in compression_ratios:
                print(f"  Compression ratio: {ratio:.0%}")
                
                if self.config.use_multiprocessing and n_trials >= 5:
                    # multiprocessing
                    results = self._run_parallel_trials(
                        circuit_config, ratio, n_trials
                    )
                else:
                    # sequential processing
                    results = self._run_sequential_trials(
                        circuit_config, ratio, n_trials
                    )
                
                avg_results = {
                    'compression_ratio': ratio,
                    'random': {
                        'mean': np.mean([r['random'] for r in results]),
                        'std': np.std([r['random'] for r in results])
                    },
                    'magnitude': {
                        'mean': np.mean([r['magnitude'] for r in results]),
                        'std': np.std([r['magnitude'] for r in results])
                    },
                    'causal': {
                        'mean': np.mean([r['causal'] for r in results]),
                        'std': np.std([r['causal'] for r in results])
                    }
                }
                
                qubit_results['compression_results'].append(avg_results)
                
                print(f"    Random: {avg_results['random']['mean']:.4f} ± {avg_results['random']['std']:.4f}")
                print(f"    Magnitude: {avg_results['magnitude']['mean']:.4f} ± {avg_results['magnitude']['std']:.4f}")
                print(f"    Causal: {avg_results['causal']['mean']:.4f} ± {avg_results['causal']['std']:.4f}")
            
            self.results.append(qubit_results)
    
    def _run_parallel_trials(self, circuit_config: CircuitConfig, 
                           compression_ratio: float, n_trials: int) -> List[Dict]:
        args_list = [
            (trial, self.config, circuit_config, compression_ratio)
            for trial in range(n_trials)
        ]
        
        with ProcessPoolExecutor(max_workers=self.analyzer.n_jobs) as executor:
            results = list(executor.map(run_single_trial_worker, args_list))
        
        return results
    
    def _run_sequential_trials(self, circuit_config: CircuitConfig,
                             compression_ratio: float, n_trials: int) -> List[Dict]:
        results = []
        
        for trial in range(n_trials):
            circuit = self.analyzer.create_test_circuit(circuit_config, trial_seed=trial)
            
            trial_result = {'trial': trial}
            for method in ['random', 'magnitude', 'causal']:
                _, fidelity = self.analyzer.prune_circuit(circuit, compression_ratio, method)
                trial_result[method] = fidelity
            
            results.append(trial_result)
        
        return results
    
    def run_depth_analysis(self, n_qubits: int = 6,
                          layer_range: List[int] = [2, 4, 6, 8, 10]):
        print(f"\n=== Depth Analysis for {n_qubits}-qubit circuits ===\n")
        
        depth_results = []
        
        for n_layers in layer_range:
            print(f"Testing {n_layers} layers...")
            
            circuit_config = CircuitConfig(
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            
            circuit = self.analyzer.create_test_circuit(circuit_config)
            
            # circuit analysis
            properties = {
                'n_layers': n_layers,
                'total_gates': len(circuit.data),
                'depth': circuit.depth()
            }
            
            # 30% compression test
            _, fidelity = self.analyzer.prune_circuit(circuit, 0.3, 'causal')
            properties['fidelity_30'] = fidelity
            
            depth_results.append(properties)
            
            print(f"  Gates: {properties['total_gates']}, "
                  f"Depth: {properties['depth']}, "
                  f"Fidelity@30%: {properties['fidelity_30']:.4f}")
        
        return depth_results
    
    def run_scalability_test(self, max_qubits: int = 16):
        print("\n=== Scalability Test ===\n")
        
        scalability_results = []
        
        for n_qubits in range(4, max_qubits + 1, 2):
            print(f"Testing {n_qubits} qubits...")
            
            circuit_config = CircuitConfig(n_qubits=n_qubits)
            circuit = self.analyzer.create_test_circuit(circuit_config)
            
            start_time = time.time()
            
            try:
                # test only importance calculation 
                importance_scores = self.analyzer.compute_gate_importance(circuit)
                
                elapsed_time = time.time() - start_time
                
                result = {
                    'n_qubits': n_qubits,
                    'n_gates': len(circuit.data),
                    'computation_time': elapsed_time,
                    'success': True,
                    'method_used': self._get_method_used(n_qubits)
                }
                
                print(f"  Success! Time: {elapsed_time:.2f}s")
                
            except Exception as e:
                print(f"  Failed: {e}")
                result = {
                    'n_qubits': n_qubits,
                    'n_gates': len(circuit.data),
                    'computation_time': -1,
                    'success': False,
                    'error': str(e)
                }
            
            scalability_results.append(result)
        
        return scalability_results
    
    def _get_method_used(self, n_qubits: int) -> str:
        if n_qubits > 16 and self.config.use_sampling:
            return "sampling"
        elif n_qubits > 12 and self.config.use_light_cone:
            return "light_cone"
        else:
            return "full_statevector"
    
    def save_results(self, filename: str):
        save_data = {
            'metadata': self.metadata,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")
    
    def plot_results(self):
        if not self.results:
            print("No results to plot!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Fidelity vs Compression for different qubit numbers
        for result in self.results:
            n_qubits = result['n_qubits']
            comp_data = result['compression_results']
            
            ratios = [d['compression_ratio'] for d in comp_data]
            causal_means = [d['causal']['mean'] for d in comp_data]
            causal_stds = [d['causal']['std'] for d in comp_data]
            
            ax1.errorbar(ratios, causal_means, yerr=causal_stds,
                        label=f'{n_qubits} qubits', marker='o', capsize=5)
        
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Fidelity')
        ax1.set_title('Causal Pruning Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.8, 1.02)
        
        # 2. Method comparison
        if self.results:
            last_result = self.results[-1]
            comp_data = last_result['compression_results']
            
            ratios = [d['compression_ratio'] for d in comp_data]
            random_means = [d['random']['mean'] for d in comp_data]
            magnitude_means = [d['magnitude']['mean'] for d in comp_data]
            causal_means = [d['causal']['mean'] for d in comp_data]
            
            ax2.plot(ratios, random_means, 'o-', label='Random')
            ax2.plot(ratios, magnitude_means, 's-', label='Magnitude')
            ax2.plot(ratios, causal_means, '^-', label='Causal')
            
            ax2.set_xlabel('Compression Ratio')
            ax2.set_ylabel('Fidelity')
            ax2.set_title(f'Method Comparison ({last_result["n_qubits"]} qubits)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Improvement over random
        for result in self.results:
            n_qubits = result['n_qubits']
            improvements = []
            ratios = []
            
            for comp in result['compression_results']:
                ratio = comp['compression_ratio']
                random_mean = comp['random']['mean']
                causal_mean = comp['causal']['mean']
                
                if random_mean < 0.999:
                    improvement = (causal_mean - random_mean) / random_mean * 100
                else:
                    improvement = 0
                
                ratios.append(ratio)
                improvements.append(improvement)
            
            ax3.plot(ratios, improvements, 'o-', label=f'{n_qubits} qubits')
        
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('Improvement over Random (%)')
        ax3.set_title('Causal Method Advantage')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        
        for result in self.results:
            n_qubits = result['n_qubits']
            
            # Find best compression at 95% fidelity
            best_compression = 0
            for comp in result['compression_results']:
                if comp['causal']['mean'] >= 0.95:
                    best_compression = comp['compression_ratio']
            
            summary_text += f"{n_qubits} qubits:\n"
            summary_text += f"  Max compression @ 95% fidelity: {best_compression:.1%}\n"
            summary_text += f"  Final causal fidelity: {result['compression_results'][-1]['causal']['mean']:.4f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        return fig


# Main execution
if __name__ == "__main__":
    print("Unified Quantum Causal Pruning PoC\n")
    print("="*50 + "\n")
    
    # configuration
    config = ExperimentConfig(
        seed=42,
        n_jobs=None,  # Auto-detect
        use_multiprocessing=True,
        use_light_cone=True,
        use_sampling=True,
        sampling_rate=0.3
    )
    
    # execute the experiment
    experiment = UnifiedExperiment(config)
    
    # 1. standard experiment (4-10 qubits)
    experiment.run_standard_experiment(
        qubit_range=[4, 6, 8, 10],
        compression_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
        n_trials=10
    )
    
    # 2. depth analysis
    depth_results = experiment.run_depth_analysis(
        n_qubits=6,
        layer_range=[2, 4, 6, 8, 10]
    )
    
    # 3. scalability test 
    scalability_results = experiment.run_scalability_test(max_qubits=16)
    
    # 4. save results
    experiment.save_results("unified_quantum_causal_results.json")
    
    # 5. visualization
    fig = experiment.plot_results()
    plt.savefig("unified_quantum_causal_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 14q test (optional)
    if input("\nTest 14-qubit circuit? (y/n): ").lower() == 'y':
        print("\n=== 14-Qubit Test ===")
        config_14 = CircuitConfig(n_qubits=14)
        circuit_14 = experiment.analyzer.create_test_circuit(config_14)
        
        print(f"Circuit: {circuit_14.num_qubits} qubits, {len(circuit_14.data)} gates")
        
        start_time = time.time()
        _, fidelity = experiment.analyzer.prune_circuit(circuit_14, 0.3, 'causal')
        elapsed_time = time.time() - start_time
        
        print(f"30% compression fidelity: {fidelity:.4f}")
        print(f"Computation time: {elapsed_time:.2f}s")
