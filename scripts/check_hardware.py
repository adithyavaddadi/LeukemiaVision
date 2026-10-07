import sys
import platform
import os
import shutil
import psutil

try:
    import torch
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "None"
except ImportError:
    torch_version = "Not installed"
    cuda_available = False
    cuda_device = "None"

try:
    import torchvision
    vision_version = torchvision.__version__
except ImportError:
    vision_version = "Not installed"

mem = psutil.virtual_memory()
disk = shutil.disk_usage('.')

print("=" * 50)
print("HARDWARE & ENVIRONMENT DIAGNOSTIC")
print("=" * 50)
print(f"OS: {platform.system()} {platform.release()} (Build {platform.version()})")
print(f"Machine / Arch: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print(f"CPU Physical Cores: {psutil.cpu_count(logical=False)}")
print(f"CPU Logical Cores: {psutil.cpu_count(logical=True)}")
print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
print(f"Disk Total: {disk.total / (1024**3):.2f} GB")
print(f"Disk Free: {disk.free / (1024**3):.2f} GB")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch_version}")
print(f"TorchVision Version: {vision_version}")
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Device: {cuda_device}")
print("=" * 50)
