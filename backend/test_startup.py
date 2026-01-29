
import torch
import torchvision.models as models
from config import Config
import sys

print("Starting import test...")
try:
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading NQK...")
    from quantum_swarmvla_backend import QuantumNeuralKernel
    nqk = QuantumNeuralKernel(n_qubits=4)
    print("NQK initialized successfully.")
    
    # Test classification
    print("Testing classification (mock)...")
    # tensor = torch.randn(1, 3, 224, 224).to(device)
    # nqk.classify_classical(tensor)
    
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
