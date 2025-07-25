#!/usr/bin/env python3

def main():
    """Check GPU availability and properties."""
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Please install PyTorch to check GPU availability.")
        return
    
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA devices detected. CUDA is not available on this system.")

if __name__ == "__main__":
    main()