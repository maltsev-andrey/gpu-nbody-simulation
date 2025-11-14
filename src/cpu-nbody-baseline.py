#!/usr/bin/env python3
"""
CPU Baseline N-Body Gravitation Simulation
Reference implementation for GPU performance comparison
Author: Andrey Maltsev
"""

import numpy as np
import time
from dataclasses import dataclass

# Configuration matching GPU version
SOFTENING = 1e-5

@dataclass
class SimulationConfig:
    """Configuration parameters for N-body simulation"""
    n_bodies: int = 50
    time_step: float = 0.01
    total_time: float = 10.0
    G: float = 1.0
    initial_radius: float = 50.0
    initial_velocity_scale: float = 0.5

class CPUNBodySimulation:
    """CPU-only N-body gravitational simulation for baseline comparison"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.n_bodies = config.n_bodies
        self.G = config.G
        self.dt = config.time_step
        
        # Initialize particle data
        self._initialize_bodies()
        
        # Allocate arrays
        self.accelerations = np.zeros((self.n_bodies, 3), dtype=np.float32)
        
        # Performance tracking
        self.computation_times = []
    
    def _initialize_bodies(self):
        """Initialize particle positions, velocities, and masses (same as GPU version)"""
        np.random.seed(42)  # Same seed as GPU version for fair comparison
        
        # Generate initial positions in a sphere
        phi = np.random.uniform(0, 2*np.pi, self.n_bodies)
        costheta = np.random.uniform(-1, 1, self.n_bodies)
        u = np.random.uniform(0, 1, self.n_bodies)
        r = self.config.initial_radius * (u ** (1/3))
        theta = np.arccos(costheta)
        
        self.positions = np.zeros((self.n_bodies, 3), dtype=np.float32)
        self.positions[:, 0] = r * np.sin(theta) * np.cos(phi)
        self.positions[:, 1] = r * np.sin(theta) * np.sin(phi)
        self.positions[:, 2] = r * np.cos(theta)
        
        # Generate initial velocities
        self.velocities = np.random.randn(self.n_bodies, 3).astype(np.float32)
        self.velocities *= self.config.initial_velocity_scale
        
        # Generate masses
        self.masses = np.random.lognormal(0, 0.5, self.n_bodies).astype(np.float32)
        self.masses = self.masses / np.mean(self.masses)
        
        # Add a central massive body if desired
        if self.n_bodies > 100:
            self.positions[0] = [0, 0, 0]
            self.velocities[0] = [0, 0, 0]
            self.masses[0] = 10.0
    
    def compute_forces(self):
        """Compute gravitational forces between all body pairs (O(N²) algorithm)"""
        self.accelerations.fill(0.0)
        
        # Double loop over all particle pairs
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i != j:
                    # Vector from body i to body j
                    dx = self.positions[j, 0] - self.positions[i, 0]
                    dy = self.positions[j, 1] - self.positions[i, 1]
                    dz = self.positions[j, 2] - self.positions[i, 2]
                    
                    # Distance with softening
                    dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING*SOFTENING
                    dist = np.sqrt(dist_sq)
                    dist_cubed = dist_sq * dist
                    
                    # Gravitational acceleration
                    force_mag = self.G * self.masses[j] / dist_cubed
                    
                    # Accumulate acceleration
                    self.accelerations[i, 0] += force_mag * dx
                    self.accelerations[i, 1] += force_mag * dy
                    self.accelerations[i, 2] += force_mag * dz
    
    def integrate(self):
        """Integrate positions and velocities using Leapfrog method"""
        # Update velocities
        self.velocities += self.accelerations * self.dt
        
        # Update positions
        self.positions += self.velocities * self.dt
    
    def compute_step(self):
        """Execute one simulation timestep"""
        start_time = time.perf_counter()
        
        self.compute_forces()
        self.integrate()
        
        computation_time = time.perf_counter() - start_time
        self.computation_times.append(computation_time)
    
    def run_simulation(self):
        """Execute the complete simulation"""
        n_steps = int(self.config.total_time / self.dt)
        
        print(f"\n{'='*60}")
        print(f"CPU Baseline N-Body Simulation")
        print(f"{'='*60}")
        print(f"Bodies: {self.n_bodies}")
        print(f"Time steps: {n_steps}")
        print(f"Algorithm: O(N²) direct summation")
        print("-" * 60)
        
        # Warm-up
        print("Warming up...")
        for _ in range(5):
            self.compute_step()
        
        # Reset timing
        self.computation_times = []
        
        # Main simulation loop
        print(f"Running simulation...")
        start_total = time.perf_counter()
        
        for step in range(n_steps):
            self.compute_step()
            
            if step % 100 == 0:
                avg_time = np.mean(self.computation_times[-100:]) if len(self.computation_times) >= 100 else np.mean(self.computation_times)
                print(f"Step {step:5d}/{n_steps} | Avg time/step: {avg_time*1000:.3f} ms")
        
        total_time = time.perf_counter() - start_total
        
        # Performance summary
        avg_time_per_step = np.mean(self.computation_times)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Total computation time: {total_time:.3f} seconds")
        print(f"Average time per step: {avg_time_per_step*1000:.3f} ms")
        print(f"Performance: {self.n_bodies * self.n_bodies / avg_time_per_step:.0f} interactions/second")
        print(f"Performance: {(self.n_bodies * self.n_bodies / avg_time_per_step)/1e6:.2f} M interactions/second")
        print("="*60)
        
        return {
            'total_time': total_time,
            'avg_time_per_step': avg_time_per_step,
            'interactions_per_sec': self.n_bodies * self.n_bodies / avg_time_per_step
        }

def benchmark_cpu(body_counts=[100, 500, 1000]):
    """Run CPU benchmark for different body counts"""
    results = []
    
    print("\n" + "="*60)
    print("CPU N-BODY BENCHMARK")
    print("="*60)
    
    for n in body_counts:
        config = SimulationConfig(
            n_bodies=n,
            total_time=1.0  # Just 1 second for benchmark
        )
        
        sim = CPUNBodySimulation(config)
        
        # Warm-up
        for _ in range(10):
            sim.compute_step()
        
        # Benchmark
        sim.computation_times = []
        start = time.perf_counter()
        for _ in range(100):
            sim.compute_step()
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / 100
        interactions_per_sec = n * n / avg_time
        
        results.append({
            'bodies': n,
            'time_per_step': avg_time * 1000,
            'interactions_per_sec': interactions_per_sec
        })
        
        print(f"N={n:5d} | {avg_time*1000:8.3f} ms/step | {interactions_per_sec/1e6:.2f} M interactions/s")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CPU baseline N-body simulation')
    parser.add_argument('-n', '--bodies', type=int, default=1000, help='Number of bodies')
    parser.add_argument('-t', '--time', type=float, default=10.0, help='Total simulation time')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_cpu()
    else:
        config = SimulationConfig(
            n_bodies=args.bodies,
            total_time=args.time
        )
        
        sim = CPUNBodySimulation(config)
        sim.run_simulation()

if __name__ == "__main__":
    main()