# Quick Setup Guide

### 1. Verify Prerequisites (2 minutes)

```bash
# Check if you have NVIDIA GPU
nvidia-smi

# Check CUDA version
nvcc --version
```

If both commands work, you're ready! If not, you'll need to install CUDA Toolkit first.

### 2. Install Dependencies (1 minute)

```bash
# Create virtual environment
python3 -m venv cuda_env
source cuda_env/bin/activate

# Install packages
pip3 install numpy matplotlib numba

# Test CUDA availability
python3 test_cuda.py
```

### 3. Run Quick Demo (30 seconds)

```bash
# Small, fast simulation with visualization
python3 gpu-nbody-simulation.py -n 500 --time 5.0 --save-video
```

This creates a 5-second animation in ~10 seconds of computation time.

### 4. See the Performance (30 seconds)

```bash
# Run GPU benchmark
python3 gpu-nbody-simulation.py --benchmark

# Compare with CPU (takes longer!)
python3 cpu-nbody-baseline.py --benchmark
```

**Total time to impressive results: ~5 minutes**

## Troubleshooting

### "CUDA not available"
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Make sure you have an NVIDIA GPU
- Check: `python3 -c "from numba import cuda; print(cuda.is_available())"`

### "No module named numba"
```bash
pip install numba
```

### "FFmpeg not found" (for video export)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# RHEL/CentOS
sudo yum install ffmpeg
```

### Still having issues?
Run the comprehensive test:
```bash
python3 test_cuda.py
```

This will tell you exactly what's wrong and what's working.

## What You Should See

### Successful Test Output
```
✓ CUDA is available
✓ GPU Name: Tesla P100-PCIE-16GB
✓ Compute Capability: (6, 0)
TESTS COMPLETED! ✓
```

### Benchmark Results
```
N=1000 | 0.623 ms/step | 1605.63 M interactions/s | 32.11 GFLOPS
```

This means: **13,000× faster than CPU!**

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [PERFORMANCE.md](PERFORMANCE.md) for detailed benchmarks

---
