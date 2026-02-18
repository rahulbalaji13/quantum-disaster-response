import sys
print("Starting imports...", flush=True)
try:
    import flask
    print("Flask ok", flush=True)
except: print("Flask failed")

try:
    import torch
    print("Torch ok", flush=True)
except: print("Torch failed")

try:
    import torchvision
    print("Torchvision ok", flush=True)
except: print("Torchvision failed")

try:
    import transformers
    print("Transformers ok", flush=True)
except: print("Transformers failed")

try:
    import datasets
    print("Datasets ok", flush=True)
except: print("Datasets failed")

try:
    import qiskit
    print("Qiskit ok", flush=True)
except: print("Qiskit failed")
print("All done", flush=True)
