import torch, sys
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"CUDA version    : {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  |  {p.total_memory // 1024**2} MB VRAM")
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f"\nGPU tensor test: OK  (shape={y.shape})")
else:
    print("No CUDA detected — installation may have an issue")
