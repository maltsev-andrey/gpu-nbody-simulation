# CUDA N-Body Gravitational Simulation

**High-Performance GPU-Accelerated Physics Simulation**

[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A GPU-accelerated N-body gravitational simulation demonstrating **13,000× speedup** over CPU baseline through CUDA parallel computing. This project showcases GPU programming techniques using Python with Numba CUDA.

![N-Body Simulation Demo](assets/nbody_demo.gif)

## Performance Highlights

| Metric               | Value                                |
|----------------------|--------------------------------------|
| **GPU Speedup**      | 13,000× faster than CPU              |
| **Computation Time** | 2+ hours → 0.6 seconds (1000 bodies) |
| **Throughput**       | 1.6 billion interactions/second      |
| **Performance**      | 32 GFLOPS sustained (Tesla P100)     |

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Benchmarks](#performance-benchmarks)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Examples](#examples)
- [Documentation](#documentation)
- [License](#license)

## Features

- **Massive Parallelization**: Leverages thousands of CUDA cores for simultaneous force calculations
- **Real-time 3D Visualization**: Interactive matplotlib-based animation of particle dynamics
- **Video Export**: Save simulations as MP4 videos with optimized compression
- **Performance Profiling**: Built-in benchmarking tools with detailed metrics
- **Energy Conservation**: Tracks total system energy to verify numerical accuracy
- **Scalable Design**: Handles from 100 to 10,000+ bodies efficiently

## Quick Start

```bash
# Clone the repository
git clone https://github.com/maltsev-andrey/gpu-nbody-simulation.git
cd gpu-nbody-simulation

# Install dependencies
pip3 install -r requirements.txt

# Verify CUDA setup
python3 test_cuda.py

# Run simulation with visualization
python3 gpu-nbody-simulation.py --save-video

# Run performance benchmark
python3 gpu-nbody-simulation.py --benchmark
```

## Installation

### Prerequisites

- **Hardware**: CUDA-capable NVIDIA GPU (Compute Capability 3.0+)
- **Software**: 
  - CUDA Toolkit 10.2 or higher
  - Python 3.7+
  - FFmpeg (for video export)

### Setup

```bash
# Create virtual environment
python3 -m venv cuda_env
source cuda_env/bin/activate  

# Install Python dependencies
pip3 install numpy matplotlib numba

# Install CUDA support for Numba
pip3 install cudatoolkit

# Verify installation
python3 -c "from numba import cuda; print(f'CUDA Available: {cuda.is_available()}')"
```

### System Test

Run the comprehensive test suite:

```bash
python3 test_cuda.py
```

Expected output:
```
 CUDA is available
 GPU Name: Tesla P100-PCIE-16GB
 Compute Capability: (6, 0)
 Multiprocessors: 56
...
TESTS COMPLETED! 
```

## Usage

### Basic Simulation

```bash
# Default: 1000 bodies, 10 seconds simulation time
python3 gpu-nbody-simulation.py

# Specify number of bodies
python3 gpu-nbody-simulation.py -n 5000

# Longer simulation time
python3 gpu-nbody-simulation.py --time 30.0

# Save animation as video
python3 gpu-nbody-simulation.py --save-video

# Run without visualization (faster)
python3 gpu-nbody-simulation.py --no-viz
```

### Performance Benchmarking

```bash
# GPU benchmark (tests 100, 500, 1000, 2000, 5000 bodies)
python3 gpu-nbody-simulation.py --benchmark

# CPU baseline benchmark
python3 cpu-nbody-baseline.py --benchmark
```

### Advanced Configuration

```python
from gpu_nbody_simulation import CUDANBodySimulation, SimulationConfig

# Custom configuration
config = SimulationConfig(
    n_bodies=2000,              # Number of particles
    time_step=0.001,            # Integration timestep (smaller = more accurate)
    total_time=20.0,            # Total simulation time
    G=1.0,                      # Gravitational constant
    initial_radius=100.0,       # Initial distribution radius
    initial_velocity_scale=0.2, # Initial velocity magnitude
    visualize=True,             # Enable 3D visualization
    save_video=True,            # Save as MP4
    video_filename="custom.mp4" # Output filename
)

sim = CUDANBodySimulation(config)
sim.run_simulation()
```

## Performance Benchmarks

### GPU Performance (NVIDIA Tesla P100)

| Bodies | Time/Step | Interactions/sec | GFLOPS | Speedup vs CPU |
|--------|-----------|------------------|--------|----------------|
| 100    | 0.18 ms   | 55 M             | 1.1    | 442×           |
| 500    | 0.36 ms   | 691 M            | 13.8   | 5,601×         |
| 1000   | 0.62 ms   | 1,606 M          | 32.1   | **13,050×**    |
| 2000   | 1.12 ms   | 3,588 M          | 71.8   | 28,000×+       |
| 5000   | 2.82 ms   | 8,871 M          | 177.4  | 50,000×+       |

### CPU Baseline (Single-threaded Python)

| Bodies | Time/Step | Interactions/sec |
|--------|-----------|------------------|
| 100    | 80 ms     | 0.13 M           |
| 500    | 2,027 ms  | 0.12 M           |
| 1000   | 8,132 ms  | 0.12 M           |

### Real-World Impact

**For 1000 bodies over 1000 timesteps:**
- **CPU**: 8,132 seconds ≈ **2 hours 15 minutes**
- **GPU**: 0.62 seconds
- **Time saved**: Over 2 hours reduced to under 1 second! ⚡

## Technical Details

### Algorithm

**O(N²) Direct Summation**
- Each particle calculates gravitational forces from all other particles
- Highly parallelizable: N threads compute N² interactions simultaneously
- Softening parameter (ε = 10⁻⁵) prevents numerical singularities

### Mathematical Foundation

**Gravitational Force:**
```
F = G × m₁ × m₂ / r²
```

**Leapfrog Integration:**
```
v(t + Δt) = v(t) + a(t) × Δt
x(t + Δt) = x(t) + v(t + Δt) × Δt
```

Superior energy conservation compared to Euler integration for orbital mechanics.

### CUDA Implementation

**Kernel 1: Force Computation**
```cuda
compute_forces_kernel<<<blocks, threads>>>(positions, masses, accelerations, n)
```
- One thread per body
- Each thread: O(N) work
- Total: O(N²) interactions computed in parallel

**Kernel 2: Integration**
```cuda
integrate_kernel<<<blocks, threads>>>(positions, velocities, accelerations, dt)
```
- Updates positions and velocities
- O(N) complexity, trivially parallel

### Optimization Techniques

1. **Adaptive Thread Configuration**: Dynamically adjusts threads per block (32-256) based on problem size for optimal GPU occupancy
2. **Coalesced Memory Access**: Structure of Arrays (SoA) layout ensures efficient memory bandwidth utilization
3. **Minimized Data Transfer**: Computations remain on GPU; only visualization snapshots transferred to CPU
4. **Double Buffering**: Separate position/velocity/acceleration buffers prevent read-write conflicts

## Project Structure

```
gpu-nbody-simulation/
.
├── README.md                  # This file
├── TECHNICAL.md               # Detailed technical documentation
├── PERFORMANCE.md             # Benchmark results and analysis
├── requirements.txt           # Python dependencies
├── assets
│   └── nbody_cuda.gif
├── demo
│   └── nbody_cuda.mp4
├── docs
├── example
├── scripts
└── src
    ├── cpu-nbody-baseline.py
    ├── gpu-nbody-simulation.py
    └── test_cuda.py
```

## Requirements

### Python Packages

```txt
numpy>=1.19.0
matplotlib>=3.3.0
numba>=0.53.0
```

Install all at once:
```bash
pip3 install -r requirements.txt
```

### System Requirements

**Minimum:**
- CUDA-capable GPU (Compute Capability 3.0+)
- 2GB GPU memory
- 4GB system RAM

**Recommended:**
- Modern GPU (Tesla P100, RTX 2060+, or equivalent)
- 4GB+ GPU memory
- 8GB+ system RAM

**Tested On:**
- NVIDIA Tesla P100-PCIE-16GB
- CUDA 12.4
- RHEL 9 / Ubuntu 20.04+

## Examples

### Example 1: Small Cluster Simulation
```bash
python3 gpu-nbody-simulation.py -n 500 --time 20.0 --save-video
```
Output: 500-body system evolving over 20 seconds, saved as `nbody_cuda.mp4`

### Example 2: Performance Comparison
```bash
# Run both benchmarks
python3 cpu-nbody-baseline.py --benchmark > cpu_results.txt
python3 gpu-nbody-simulation.py --benchmark > gpu_results.txt

## Documentation

- **[Technical Documentation](TECHNICAL.md)**: In-depth algorithm explanation, CUDA kernel details, and optimization strategies
- **[Performance Analysis](PERFORMANCE.md)**: Comprehensive benchmark results, scaling behavior, and comparison with CPU
- **[API Reference](docs/API.md)**: Complete API documentation for classes and methods
- **[Tutorial](docs/TUTORIAL.md)**: Step-by-step guide to understanding and modifying the code

## Learning Resources

This project demonstrates:
- **GPU Parallel Programming**: CUDA kernel development using Numba
- **Performance Engineering**: Achieving 10,000× speedups through parallelization
- **Scientific Computing**: Accurate numerical simulation of physical systems
- **Data Visualization**: Real-time 3D rendering and animation export

## Contributing

Contributions are welcome! Areas for improvement:
- Barnes-Hut tree algorithm for O(N log N) complexity
- Multi-GPU support for larger simulations
- Additional force models (electromagnetic, molecular)
- Interactive 3D controls
- Web-based visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Andrey Maltsev**

This project showcases GPU computing expertise through a practical, high-performance scientific simulation with measurable results.

## Acknowledgments

- NVIDIA CUDA documentation and examples
- Numba project for excellent Python GPU support
- Scientific computing community for N-body algorithm references

## Contact

For questions, collaboration opportunities, or technical discussions about GPU computing and high-performance computing projects, feel free to reach out!

---

**If you find this project useful, please consider giving it a star!**
