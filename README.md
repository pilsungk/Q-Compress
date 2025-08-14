# Emergent Bifurcations in Quantum Circuit Stability from Hidden Parameter Statistics

This repository contains the official source code and data for the paper, "Emergent Bifurcations in Quantum Circuit Stability from Hidden Parameter Statistics".

## Abstract

The compression of quantum circuits is a foundational challenge for near-term quantum computing, yet the principles governing circuit stability remain poorly understood. We investigate this problem through a large-scale numerical analysis of 300 structurally-uniform circuits across 10, 12, and 14 qubits. Despite their macroscopic uniformity, we find that each ensemble universally bifurcates into distinct robust and fragile classes. We solve the puzzle of this emergent bifurcation, demonstrating that its origin is not structural, but is instead encoded in the statistical properties of the gate rotation parameters. Fragile circuits consistently exhibit a universal signature of ``statistical brittleness,'' characterized by low parameter variability and a scarcity of small-angle gates. We uncover the underlying physical mechanism for this phenomenon: Paradoxical Importance where smaller-angle gates are counter-intuitively more critical to the circuit's function, an effect most pronounced in fragile circuits. This reliance on fine-tuning explains why statistically brittle circuits are uniquely vulnerable to failure under compression. These findings establish a new framework for engineering resilient quantum algorithms, shifting the focus from macroscopic structure to the microscopic statistical properties of a circuit's parameters.

## Repository Structure

-   `/src`: Contains all Python source code for circuit generation, analysis, and plotting.
-   `/data`: Contains the generated circuit ensembles and data for figures.

## Reproducing the Results
The main findings of the paper can be reproduced with the following steps.

### Step 1: Generate Circuit Ensembles
The phase_transition_pipeline.py script first identifies the critical compression ratio and then generates the 100-circuit ensemble for a given qubit size.

To generate the 10-qubit circuit ensemble, run:

```bash
python src/phase_transition_pipeline.py --n_qubit 10
```
This will create an output directory containing the 100 generated circuits. Repeat for n_qubit 12 and 14.

### Step 2: Run Statistical Analyses
Once the circuits are generated, run the analysis scripts on the output directory. For example, to run the Parameter Microscope analysis:

```bash
python src/parameter_microscope.py --input_dir <path_to_generated_circuits>
```
This will analyze the rotation parameter distributions (RX, RY, RZ) and generate the summary data used in the paper.

### Citation
If you find this work useful, please cite our paper:

@misc{kang2025emergentbifurcation,
      title="{Emergent Bifurcations in Quantum Circuit Stability from Hidden Parameter Statistics}", 
      author={Pilsung Kang},
      year={2025},
      eprint={2508.00484},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2508.00484 }, 
}
