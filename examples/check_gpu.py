import torch
print("=" * 60)
print("REMOTE GOOGLE COLAB GPU VERIFICATION")
print("=" * 60)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM Capacity:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("=" * 60)
