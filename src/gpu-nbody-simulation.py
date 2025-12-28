#!/usr/bin/env python3
"""
CUDA Accelerated N-Body Gravitation Simulation
High-Performance parallel implementattion using GPU computing
Author: Andrey Maltsev
"""

import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
import numba
from numba import cuda
import math
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import sys
import os

# Check if user root or not
if os.getuid() == 0:
    print("\n"+"-"*60 )
    print("ERROR: Don't run GPU as root! Use: su - ansible")
    print("-"*60 + "\n")
    sys.exit(1)

# CUDA kernel configuration
def get_threads_per_block(n_bodies):
    """
    Dynamically determine optimal threads per block based on problem size.
    Smaller problems need fewer threads per block to create more blocks.
    """
    if n_bodies <= 256:
        return 32   # Very small: 32 threads → more blocks
    elif n_bodies <= 1024:
        return 64   # Small: 64 threads
    elif n_bodies <= 2048:
        return 128  # Medium: 128 threads
    else:
        return 256  # Large: 256 threads
SOFTENING = 1e-5 # Prevents singularities in force calculation

@dataclass
class SimulationConfig:
    """Configuration parameters for N-body simulation"""
    n_bodies: int = 1000
    time_step: float = 0.01
    total_time: float = 10.0
    G: float = 1.0 # Gravitational constant (normalized units)
    initial_radius: float = 50.0
    initial_velocity_scale: float = 0.5
    visualize: bool = True
    save_video: bool = True
    video_filename: str = "nbody_cuda.mp4"
    update_interval: int = 10
    benchmark_mode: bool = False

@cuda.jit
def compute_forces_kernel(positions, masses, accelerations, n_bodies, G, softening):
    """
    CUDA kernel for computing gravitational forces between all body pairs.
    Each thread computes the total force on one body from all other bodies.
    
    Uses shared memory optimization for coalesced memory access.
    """
    tid = cuda.grid(1)

    if tid >= n_bodies:
        return

    # Load position of current body
    px = positions[tid, 0]
    py = positions[tid, 1]
    pz = positions[tid, 2]

    # Initial acceleartion
    ax = 0.0
    ay = 0.0
    az = 0.0

    # compute forces from all other bodies
    for j in range(n_bodies):
        if tid != j:
            # Vector from body i to body j
            dx = positions[j, 0] - px
            dy = positions[j, 1] - py
            dz = positions[j, 2] - pz

            # Distance with softening to prevent singularities
            dist_sq = dx * dx + dy * dy + dz * dz + softening*softening
            dist = math.sqrt(dist_sq)
            dist_cubed = dist_sq * dist

            # Gravitational force (F = G*m1*m2/r^2)
            # Acceleration (a = F/m1 = G*m2/r^2)
            force_mag = G * masses[j] / dist_cubed

            # Accumulate acceleration components
            ax += force_mag * dx
            ay += force_mag * dy
            az += force_mag * dz

    # Store computed acceleration
    accelerations[tid, 0] = ax
    accelerations[tid, 1] = ay
    accelerations[tid, 2] = az

@cuda.jit
def integrate_kernel(positions, velocities, accelerations, n_bodies, dt):
        """
        CUDA kernel for integrating positions and velocities using Leapfrog method.
        More stable than simple Euler integration for orbital mechanics.
        """
        tid = cuda.grid(1)
        
        if tid >= n_bodies:
            return
        
        # Update velocities (v = v + a*dt)
        velocities[tid, 0] += accelerations[tid, 0] * dt
        velocities[tid, 1] += accelerations[tid, 1] * dt
        velocities[tid, 2] += accelerations[tid, 2] * dt
        
        # Update positions (x = x + v*dt)
        positions[tid, 0] += velocities[tid, 0] * dt
        positions[tid, 1] += velocities[tid, 1] * dt
        positions[tid, 2] += velocities[tid, 2] * dt        

class CUDANBodySimulation:
        """
        GPU-accelerated N-body gravitational simulation using CUDA.
        Achieves massive speedup over CPU implementation through parallelization.
        """
        
        def __init__(self, config: SimulationConfig):
            self.config = config
            self.n_bodies = config.n_bodies
            self.G = config.G
            self.dt = config.time_step
         
            # Initialize particel data
            self._initialize_bodies()
            
            #Allocate device memory
            self.d_positions = cuda.to_device(self.positions)
            self.d_velocities = cuda.to_device(self.velocities)
            self.d_masses= cuda.to_device(self.masses)
            self.d_accelerations = cuda.device_array_like(self.velocities)
            
            # Calculate kernel launch configuration
            # self.blocks_per_grid = (self.n_bodies + THREADS_PER_BLOCK -1) // THREADS_PER_BLOCK
            self.THREADS_PER_BLOCK = get_threads_per_block(self.n_bodies)
            self.blocks_per_grid = (self.n_bodies + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
            
            # Performance tracking
            self.computation_times = []
            self.total_energy = []
 
        def _initialize_bodies(self):
            """Initialize particle positions, velocities, and masses"""
            np.random.seed(42) # Reproducible results
            
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
            
            #Generate initial velocities (small random perturbations)
            self.velocities = np.random.randn(self.n_bodies, 3).astype(np.float32)
            self.velocities *= self.config.initial_velocity_scale
            
            # Generate masses (log-normal distribution for realistic stellar masses)
            self.masses = np.random.lognormal(0, 0.5, self.n_bodies).astype(np.float32)
            self.masses = self.masses / np.mean (self.masses) # Normalize
            
            # Add a central massive body if desired
            if self .n_bodies > 100:
                # Place massive body at center
                self.positions[0] = [0, 0, 0]
                self.velocities[0] = [0, 0, 0]
                self.masses[0] = 10.0 # 10x average mass
       
        def compute_step(self):
            """Execute one simulation timestep on GPU""" 
            start_time = time.perf_counter()
  
            # Compute gravitation forces
            compute_forces_kernel[self.blocks_per_grid, self.THREADS_PER_BLOCK](
                self.d_positions, self.d_masses, self.d_accelerations,
                self.n_bodies, self.G, SOFTENING
            )
  
            # Integrate positions and velocities
            integrate_kernel[self.blocks_per_grid, self.THREADS_PER_BLOCK](
                 self.d_positions, self.d_velocities, self.d_accelerations,
                 self.n_bodies, self.dt
            )
  
            # Synchronize GPU
            cuda.synchronize()
  
            computation_time = time.perf_counter() - start_time
            self.computation_times.append(computation_time)
  
        def get_positions(self):
            """Copy positionsfrom GPU to CPU for visualisation"""
            return self.d_positions.copy_to_host()

        def get_velocities(self):
            """Copy velocities from GPU to CPU"""
            return self.d_velocities.copy_to_host()
 
        def calculate_total_energy(self):
            """Calculate total system energy (kinetic + potential)"""
            positions = self.get_positions()
            velocities = self.get_velocities()
            masses = self.masses
 
            # kinetic energy: 0.5 * m * v^2
            kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
 
            # potential energy: -G * m1 * m2 / r
            potential = 0.0
            for i in range(self.n_bodies):
                for j in range(i+1, self.n_bodies):
                    r = np.linalg.norm(positions[i] - positions[j])
                    if r > SOFTENING:
                        potential -= self.G * masses[i] * masses[j] / r
 
            return kinetic + potential    
 
        def run_simulation(self):
            """Execute the complete simulation"""
            n_steps = int(self.config.total_time / self.dt)
 
            print(f"Starting CUDA N-Body Simulation")
            print(f"Bodies: {self.n_bodies}")
            print(f"Time steps: {n_steps}")
            print(f"GPU: {cuda.get_current_device().name.decode()}")
            print("-" * 50)
 
            # Store positions for visualization
            if self.config.visualize:
                 position_history = []
             
            # Main simulation loop
            for step in range(n_steps):
                 self.compute_step()
                 
                 # Store positions for animation
                 if self.config.visualize and step % self.config.update_interval == 0:
                     position_history.append(self.get_positions().copy())
                
                 # Energy conservation check
                 if step % 100 == 0:
                     energy = self.calculate_total_energy()
                     self.total_energy.append(energy)
                     
                     if not self.config.benchmark_mode:
                         avg_time = np.mean(self.computation_times[-100:]) if len(self.computation_times) >= 100 else np.mean(self.computation_times)
                         print(f"Step {step:5d}/{n_steps} | Avg time/step: {avg_time*1000:.3f} ms | Energy: {energy:.6f}")
         
            # Performance summary
            total_computation_time = sum(self.computation_times)
            avg_time_per_step = np.mean(self.computation_times)
            
            print("\n" + "="*50)
            print("SIMULATION COMPLETE")
            print(f"Total computation time: {total_computation_time:.3f} seconds")
            print(f"Average time per step: {avg_time_per_step*1000:.3f} ms")
            print(f"Performance: {self.n_bodies * self.n_bodies / avg_time_per_step:.0f} interactions/second")
            print(f"Energy drift: {(self.total_energy[-1] - self.total_energy[0])/self.total_energy[0]*100:.2f}%")
            
            if self.config.visualize:
               self.create_visualization(position_history)

        def create_visualization(self, position_history) :
            """Create animated visualization of simulation results"""
            print("\nCreating visualization...")
 
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
 
            # Set up plot
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            # ax.set_title(f'CUDA N-Body Simulation ( {self.n_bodies} bodies )') 
            ax.set_title(f'Численное моделирование системы из N тел ( {self.n_bodies} bodies )')
 
            # Determine plot limits
            all_positions = np.array(position_history) 
            limit = np.max(np.abs(all_positions)) * 1.1
            ax.set_xlim([-limit, limit]) 
            ax.set_ylim([-limit, limit]) 
            ax.set_zlim([-limit, limit])  
 
            # Color by mass
            colors = plt.cm.viridis(self.masses / np.max(self.masses)) 
            sizes = 20 + 30 * (self.masses / np.max(self.masses)) 
 
            # Create scatter plot with first frame
            first_positions = position_history[0]
            scatter = ax.scatter(first_positions[:, 0], first_positions[:, 1], first_positions[:, 2], c=colors, s=sizes, alpha=0.6)

            def update(frame):
                positions = position_history[frame]
                scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                ax.set_title(f'Численное моделирование системы из N тел | Фрэйм {frame}/{len(position_history)}')
                return scatter,
        
            anim = FuncAnimation(fig, update, frames=len(position_history),
                           interval=50, blit=False)
        
            if self.config.save_video:
                writer = FFMpegWriter(fps=20, bitrate=2000)
                anim.save(self.config.video_filename, writer=writer)
                print(f"Animation saved to {self.config.video_filename}")
            
            plt.show()

def benchmark_comparison():
    """Run benchmark comparing different body counts"""
    body_counts = [100, 500, 1000, 2000, 5000]
    results = []

    print("\n" + "="*60)
    print("CUDA N_BODY BENCHMARK")
    print("="*60)

    for n in body_counts:
        config = SimulationConfig(
            n_bodies = n,
            total_time = 1.0,
            visualize = False,
            benchmark_mode = True         
        )

        sim = CUDANBodySimulation(config)    

        # Warm-up GPU
        for _ in range(10):
            sim.compute_step()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            sim.compute_step()
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / 100
        interactions_per_sec = n * n / avg_time
        
        results.append({
            'bodies': n,
            'time_per_step': avg_time * 1000,
            'interactions_per_sec': interactions_per_sec,
            'gflops': interactions_per_sec * 20 / 1e9  # ~20 FLOPs per interaction
        })
        
        print(f"N={n:5d} | {avg_time*1000:8.3f} ms/step | {interactions_per_sec/1e6:.2f} M interactions/s | {results[-1]['gflops']:.2f} GFLOPS")

    return results

def main():
    parser = argparse.ArgumentParser(description='CUDA-accelerated N-body gravitational simulation')
    parser.add_argument('-n', '--bodies', type=int, default=1000, help='Number of bodies')
    parser.add_argument('-t', '--time', type=float, default=10.0, help='Total simulation time')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--save-video', action='store_true', help='Save animation as video')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_comparison()
    else:
        config = SimulationConfig(
            n_bodies=args.bodies,
            total_time=args.time,
            time_step=args.dt,
            visualize=not args.no_viz,
            save_video=args.save_video
        )
        
        sim = CUDANBodySimulation(config)
        sim.run_simulation()

if __name__ == "__main__":
    main()  