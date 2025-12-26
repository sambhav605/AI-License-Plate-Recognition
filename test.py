import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is still using the CPU. Re-check the installation.")