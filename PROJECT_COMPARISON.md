# 🚀 Project Comparison: Classical vs. Quantum-SwarmVLA

This document outlines the performance enhancements and architectural shifts between your previous classical implementation and the new **Quantum-SwarmVLA** system.

## 📊 Performance & Architecture Tabulation

| Feature / Metric | Previous Research (Classical) | Quantum-SwarmVLA (New Proposed) | ⚛️ Quantum Enhancement / Innovation |
| :--- | :--- | :--- | :--- |
| **Vision Backbone** | **Florence-2** (Vision Foundation Model) | **Hybrid ResNet + Quantum Neural Kernel (NQK)** | Maps classical features into a high-dimensional **Hilbert Space** using quantum entanglement for better class separability. |
| **Multimodal Processing** | **VL-Mamba** (State Space Model) | **Quantum Feature Map (ZZ / RY Gates)** | Uses qubit rotation and entanglement to capture non-linear relationships between data points that classical kernels might miss. |
| **Swarm Logic** | Basic majority voting | **Swarm Confidence Aggregation**| Leverages weighted averaging across the drone swarm to improve resilience against obstructed sensors and eliminate false positives. |

| **Decision Confidence** | ~90-92% Accuracy | **High Fidelity with Quantum Entropy** | Quantum measurement entropy provides a more robust "uncertainty" metric than standard Softmax probability. |
| **Architecture Type** | Deep Learning (Transformer/SSM) | **Hybrid Quantum-Classical Edge** | Designed for **Edge Deployment** where a classical CPU handles pre-processing and a QPU (or simulator) handles the complex kernel. |

---

## 🏗️ Proposed Quantum Architecture Flow

The new architecture introduces two critical Quantum Modules into the pipeline:

1.  **NQK (Neural Quantum Kernel)**: Replaces the standard classification head. It takes features from the CNN (ResNet), encodes them into qubit rotation angles, executes a quantum circuit, and measures the output to determine the disaster type.
2.  **Swarm Confidence Aggregation**: A robust mechanism where data from multiple swarm agents (drones) surveying the same area is averaged to validate the disaster data before an alert is triggered, preventing false positives from single faulty sensors.


### Operational Flow:

1.  **Input**: Real-time Video/Image Stream or Upload.
2.  **Preprocessing**: Classical ResNet extracts 4-dimensional feature vectors.
3.  **Quantum Core**:
    *   **Encoding**: Data is mapped to effective Hamiltonian of the quantum system.
    *   **Processing**: Entangled qubits evolve state.
    *   **Measurement**: Collapse wave function to get classification class (Flood, Fire, etc.).
4.  **Consensus**: Swarm agents vote on the result confidence.
5.  **Action**:
    *   **Notification**: Twilio SMS to rescue teams.

