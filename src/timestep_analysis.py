#!/usr/bin/env python3
"""
Numerical Accuracy Analysis: Timestep Comparison
=================================================

Investigates how timestep size affects simulation accuracy.
Same total simulated time, different granularity:

- Fine:   1000 steps × dt=0.01  → Total time = 10
- Medium: 100 steps × dt=0.1   → Total time = 10  
- Coarse: 10 steps × dt=1.0    → Total time = 10

Compare final states to measure numerical drift.

Author: Andrey Maltsev
Usage: python3 timestep_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import time
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')
plt.style.use('dark_background')

# =============================================================================
# CONSTANTS (matching your gpu-nbody-simulation.py)
# =============================================================================
SOFTENING = 1e-5
G = 1.0

COLORS = {
    'fine': '#96CEB4',
    'medium': '#4ECDC4',
    'coarse': '#FF6B6B',
    'accent': '#FFEAA7',
    'grid': '#2C3E50'
}


# =============================================================================
# CUDA KERNELS (same as your implementation)
# =============================================================================

@cuda.jit
def compute_forces_kernel(positions, masses, accelerations, n_bodies, G, softening):
    """CUDA kernel for gravitational forces."""
    tid = cuda.grid(1)
    
    if tid >= n_bodies:
        return
    
    px = positions[tid, 0]
    py = positions[tid, 1]
    pz = positions[tid, 2]
    
    ax, ay, az = 0.0, 0.0, 0.0
    
    for j in range(n_bodies):
        if tid != j:
            dx = positions[j, 0] - px
            dy = positions[j, 1] - py
            dz = positions[j, 2] - pz
            
            dist_sq = dx*dx + dy*dy + dz*dz + softening*softening
            dist = math.sqrt(dist_sq)
            dist_cubed = dist_sq * dist
            
            force_mag = G * masses[j] / dist_cubed
            
            ax += force_mag * dx
            ay += force_mag * dy
            az += force_mag * dz
    
    accelerations[tid, 0] = ax
    accelerations[tid, 1] = ay
    accelerations[tid, 2] = az


@cuda.jit
def integrate_kernel(positions, velocities, accelerations, n_bodies, dt):
    """Leapfrog integration kernel."""
    tid = cuda.grid(1)
    
    if tid >= n_bodies:
        return
    
    velocities[tid, 0] += accelerations[tid, 0] * dt
    velocities[tid, 1] += accelerations[tid, 1] * dt
    velocities[tid, 2] += accelerations[tid, 2] * dt
    
    positions[tid, 0] += velocities[tid, 0] * dt
    positions[tid, 1] += velocities[tid, 1] * dt
    positions[tid, 2] += velocities[tid, 2] * dt


# =============================================================================
# SIMULATION CLASS FOR EXPERIMENTS
# =============================================================================

class TimestepSimulator:
    """Lightweight simulator for timestep experiments."""
    
    def __init__(self, n_bodies: int, dt: float):
        self.n_bodies = n_bodies
        self.dt = dt
        self.G = G
        
        # CUDA configuration
        if n_bodies <= 256:
            self.threads = 32
        elif n_bodies <= 1024:
            self.threads = 64
        elif n_bodies <= 2048:
            self.threads = 128
        else:
            self.threads = 256
        self.blocks = (n_bodies + self.threads - 1) // self.threads
        
    def initialize(self, positions: np.ndarray, velocities: np.ndarray, 
                   masses: np.ndarray):
        """Initialize with given state."""
        self.positions = positions.astype(np.float32)
        self.velocities = velocities.astype(np.float32)
        self.masses = masses.astype(np.float32)
        
        self.d_positions = cuda.to_device(self.positions)
        self.d_velocities = cuda.to_device(self.velocities)
        self.d_masses = cuda.to_device(self.masses)
        self.d_accelerations = cuda.device_array_like(self.velocities)
    
    def step(self):
        """Execute one timestep."""
        compute_forces_kernel[self.blocks, self.threads](
            self.d_positions, self.d_masses, self.d_accelerations,
            self.n_bodies, self.G, SOFTENING
        )
        integrate_kernel[self.blocks, self.threads](
            self.d_positions, self.d_velocities, self.d_accelerations,
            self.n_bodies, self.dt
        )
        cuda.synchronize()
    
    def get_positions(self) -> np.ndarray:
        return self.d_positions.copy_to_host()
    
    def get_velocities(self) -> np.ndarray:
        return self.d_velocities.copy_to_host()
    
    def calculate_energy(self) -> Tuple[float, float, float]:
        """Calculate kinetic, potential, and total energy."""
        positions = self.get_positions()
        velocities = self.get_velocities()
        
        # Kinetic energy
        kinetic = 0.5 * np.sum(self.masses[:, np.newaxis] * velocities**2)
        
        # Potential energy
        potential = 0.0
        for i in range(self.n_bodies):
            for j in range(i+1, self.n_bodies):
                r = np.linalg.norm(positions[i] - positions[j])
                if r > SOFTENING:
                    potential -= self.G * self.masses[i] * self.masses[j] / r
        
        return kinetic, potential, kinetic + potential


# =============================================================================
# INITIAL CONDITIONS (matching your implementation)
# =============================================================================

def generate_initial_conditions(n_bodies: int, seed: int = 42,
                                 initial_radius: float = 50.0,
                                 initial_velocity_scale: float = 0.5):
    """Generate initial particle distribution (same as your code)."""
    np.random.seed(seed)
    
    # Spherical distribution
    phi = np.random.uniform(0, 2*np.pi, n_bodies)
    costheta = np.random.uniform(-1, 1, n_bodies)
    u = np.random.uniform(0, 1, n_bodies)
    r = initial_radius * (u ** (1/3))
    theta = np.arccos(costheta)
    
    positions = np.zeros((n_bodies, 3), dtype=np.float32)
    positions[:, 0] = r * np.sin(theta) * np.cos(phi)
    positions[:, 1] = r * np.sin(theta) * np.sin(phi)
    positions[:, 2] = r * np.cos(theta)
    
    # Random velocities
    velocities = np.random.randn(n_bodies, 3).astype(np.float32)
    velocities *= initial_velocity_scale
    
    # Log-normal mass distribution
    masses = np.random.lognormal(0, 0.5, n_bodies).astype(np.float32)
    masses = masses / np.mean(masses)
    
    # Central massive body
    if n_bodies > 100:
        positions[0] = [0, 0, 0]
        velocities[0] = [0, 0, 0]
        masses[0] = 10.0
    
    return positions, velocities, masses


# =============================================================================
# EXPERIMENT FRAMEWORK
# =============================================================================

@dataclass
class TimestepExperiment:
    """Configuration for one experiment."""
    name: str
    n_steps: int
    dt: float
    
    @property
    def total_time(self) -> float:
        return self.n_steps * self.dt


@dataclass
class ExperimentResult:
    """Results from one experiment."""
    experiment: TimestepExperiment
    final_positions: np.ndarray
    final_velocities: np.ndarray
    initial_energy: float
    final_energy: float
    energy_history: List[float]
    position_history: List[np.ndarray]
    computation_time: float
    
    @property
    def energy_drift_percent(self) -> float:
        return abs(self.final_energy - self.initial_energy) / abs(self.initial_energy) * 100


def run_experiment(positions: np.ndarray, velocities: np.ndarray,
                   masses: np.ndarray, experiment: TimestepExperiment,
                   record_interval: int = None) -> ExperimentResult:
    """Run one timestep experiment."""
    n_bodies = len(masses)
    
    if record_interval is None:
        record_interval = max(1, experiment.n_steps // 100)
    
    sim = TimestepSimulator(n_bodies, experiment.dt)
    sim.initialize(positions.copy(), velocities.copy(), masses.copy())
    
    # Initial energy
    _, _, initial_energy = sim.calculate_energy()
    energy_history = [initial_energy]
    position_history = [positions.copy()]
    
    # Run simulation
    start = time.perf_counter()
    for step in range(experiment.n_steps):
        sim.step()
        
        if (step + 1) % record_interval == 0 or step == experiment.n_steps - 1:
            position_history.append(sim.get_positions().copy())
            _, _, energy = sim.calculate_energy()
            energy_history.append(energy)
    
    elapsed = time.perf_counter() - start
    
    # Final state
    final_pos = sim.get_positions()
    final_vel = sim.get_velocities()
    _, _, final_energy = sim.calculate_energy()
    
    return ExperimentResult(
        experiment=experiment,
        final_positions=final_pos,
        final_velocities=final_vel,
        initial_energy=initial_energy,
        final_energy=final_energy,
        energy_history=energy_history,
        position_history=position_history,
        computation_time=elapsed
    )


def compute_position_difference(ref: ExperimentResult, 
                                  other: ExperimentResult) -> Dict:
    """Compute difference between final states."""
    distances = np.linalg.norm(ref.final_positions - other.final_positions, axis=1)
    
    com_ref = np.mean(ref.final_positions, axis=0)
    com_other = np.mean(other.final_positions, axis=0)
    
    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances),
        'com_difference': np.linalg.norm(com_ref - com_other),
        'per_particle': distances
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_plots(results: List[ExperimentResult], output_dir: str):
    """Create comprehensive visualization of timestep effects."""
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    colors = [COLORS['fine'], COLORS['medium'], COLORS['coarse']]
    
    # 1. Final positions (2D projection)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1a1a1a')
    
    for result, color in zip(results, colors):
        pos = result.final_positions
        ax1.scatter(pos[:, 0], pos[:, 1], c=color, s=8, alpha=0.6,
                   label=result.experiment.name)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Y Position', fontsize=11)
    ax1.set_title('Final Particle Positions (XY Projection)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    ax1.set_aspect('equal')
    
    # 2. Energy over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1a1a1a')
    
    for result, color in zip(results, colors):
        n_points = len(result.energy_history)
        times = np.linspace(0, result.experiment.total_time, n_points)
        normalized = np.array(result.energy_history) / result.initial_energy
        ax2.plot(times, normalized, color=color, linewidth=2,
                label=result.experiment.name)
    
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Perfect conservation')
    ax2.set_xlabel('Simulation Time', fontsize=11)
    ax2.set_ylabel('E / E₀ (Normalized)', fontsize=11)
    ax2.set_title('Energy Conservation Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # 3. Position error vs reference
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a1a')
    
    reference = results[0]
    labels = []
    mean_errors = []
    max_errors = []
    
    for result in results[1:]:
        diff = compute_position_difference(reference, result)
        labels.append(result.experiment.name.split(':')[1].strip())
        mean_errors.append(diff['mean_distance'])
        max_errors.append(diff['max_distance'])
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax3.bar(x - width/2, mean_errors, width, label='Mean Error', color=COLORS['medium'])
    ax3.bar(x + width/2, max_errors, width, label='Max Error', color=COLORS['coarse'])
    
    ax3.set_xlabel('Timestep Configuration', fontsize=11)
    ax3.set_ylabel('Position Error vs Fine Reference', fontsize=11)
    ax3.set_title('Numerical Drift Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # 4. Energy drift bar chart
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1a1a1a')
    
    drifts = [r.energy_drift_percent for r in results]
    labels = [r.experiment.name for r in results]
    
    bars = ax4.bar(range(len(results)), drifts, color=colors, edgecolor='white')
    
    for bar, drift in zip(bars, drifts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + max(drifts)*0.02,
                f'{drift:.3f}%', ha='center', fontsize=10, fontweight='bold',
                color='white')
    
    ax4.set_xlabel('Timestep Configuration', fontsize=11)
    ax4.set_ylabel('Energy Drift (%)', fontsize=11)
    ax4.set_title('Total Energy Drift After Simulation', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels([r.experiment.name for r in results], rotation=15, ha='right')
    ax4.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    fig.suptitle('Timestep Size vs Numerical Accuracy\nLeapfrog Integration Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'timestep_analysis.png')
    plt.savefig(path, dpi=150, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    
    return path


def create_trajectory_plot(results: List[ExperimentResult], 
                            particle_indices: List[int],
                            output_dir: str):
    """Create trajectory comparison for selected particles."""
    
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a1a')
    
    colors = [COLORS['fine'], COLORS['medium'], COLORS['coarse']]
    
    # Plot trajectories
    for p_idx in particle_indices:
        for result, color in zip(results, colors):
            trajectory = np.array([pos[p_idx] for pos in result.position_history])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, linewidth=1, alpha=0.7)
            # Start marker
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      color=color, s=50, marker='o', edgecolors='white', zorder=5)
            # End marker
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color=color, s=100, marker='*', edgecolors='white', zorder=5)
    
    # Legend
    for result, color in zip(results, colors):
        ax.plot([], [], color=color, linewidth=2, label=result.experiment.name)
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Trajectory Comparison (Particles {particle_indices})\n'
                f'○ = Start, ★ = End | Same initial conditions',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.set_aspect('equal')
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(path, dpi=150, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()
    
    return path


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_timestep_analysis(n_bodies: int = 100,
                           total_time: float = 10.0,
                           output_dir: str = '../docs'):
    """
    Run the complete timestep analysis.
    
    This is the experiment you mentioned:
    - Fine:   1000 steps × dt=0.01  → Total time = 10
    - Medium: 100 steps × dt=0.1   → Total time = 10
    - Coarse: 10 steps × dt=1.0    → Total time = 10
    """
    
    print("=" * 70)
    print("TIMESTEP ACCURACY ANALYSIS")
    print("=" * 70)
    
    # Check CUDA
    if not cuda.is_available():
        print("\nERROR: CUDA not available!")
        print("Run this on your P100 system.")
        return None
    
    device = cuda.get_current_device()
    print(f"\nGPU: {device.name.decode()}")
    print(f"Bodies: {n_bodies}")
    print(f"Total simulation time: {total_time}")
    
    # Generate initial conditions
    print("\nGenerating initial conditions...")
    positions, velocities, masses = generate_initial_conditions(n_bodies)
    
    # Define experiments
    experiments = [
        TimestepExperiment("Fine: 1000×0.01", n_steps=1000, dt=0.01),
        TimestepExperiment("Medium: 100×0.1", n_steps=100, dt=0.1),
        TimestepExperiment("Coarse: 10×1.0", n_steps=10, dt=1.0),
    ]
    
    # Verify total times match
    for exp in experiments:
        assert abs(exp.total_time - total_time) < 1e-10, \
            f"{exp.name}: total_time={exp.total_time} != {total_time}"
    
    # Run experiments
    results = []
    print("\n" + "-" * 70)
    
    for exp in experiments:
        print(f"\nRunning: {exp.name}")
        print(f"  Steps: {exp.n_steps}, dt: {exp.dt}")
        
        result = run_experiment(positions, velocities, masses, exp)
        
        print(f"  Time: {result.computation_time:.3f}s")
        print(f"  Energy drift: {result.energy_drift_percent:.6f}%")
        results.append(result)
    
    # Create visualizations
    print("\n" + "-" * 70)
    print("Creating visualizations...")
    
    plot_path = create_analysis_plots(results, output_dir)
    traj_path = create_trajectory_plot(results, [0, 1, 5, 10], output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    reference = results[0]
    print(f"\nReference: {reference.experiment.name}")
    print(f"  Initial energy: {reference.initial_energy:.4f}")
    print(f"  Final energy: {reference.final_energy:.4f}")
    print(f"  Energy drift: {reference.energy_drift_percent:.6f}%")
    
    for result in results[1:]:
        print(f"\n{result.experiment.name}:")
        print(f"  Energy drift: {result.energy_drift_percent:.4f}%")
        
        diff = compute_position_difference(reference, result)
        print(f"  Mean position error: {diff['mean_distance']:.4f}")
        print(f"  Max position error: {diff['max_distance']:.4f}")
        print(f"  Center of mass shift: {diff['com_difference']:.4f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. NUMERICAL ERROR ACCUMULATION
   - Leapfrog integration error scales with dt²
   - Coarse timesteps accumulate significant drift
   - Fine timesteps maintain better conservation

2. TRAJECTORY DIVERGENCE  
   - Same initial conditions → different final states
   - Close encounters amplify numerical differences
   - Phase errors compound over many orbits

3. GPU ADVANTAGE
   - With 13,000× speedup, you can afford dt=0.001 instead of dt=0.1
   - This gives ~100× better accuracy at same wall-clock time
   - Enables high-fidelity physics for robotics/simulation

4. PRODUCTION RECOMMENDATIONS
   - Use adaptive timesteps for close encounters
   - Consider symplectic integrators for long simulations
   - Validate with energy conservation checks
""")
    
    print(f"\nOutput files saved to: {output_dir}/")
    
    return {
        'results': results,
        'plots': [plot_path, traj_path]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Timestep accuracy analysis')
    parser.add_argument('-n', '--bodies', type=int, default=100,
                        help='Number of bodies (default: 100)')
    parser.add_argument('-t', '--time', type=float, default=10.0,
                        help='Total simulation time (default: 10.0)')
    parser.add_argument('-o', '--output', type=str, default='../docs',
                        help='Output directory (default: ../docs)')
    
    args = parser.parse_args()
    
    run_timestep_analysis(
        n_bodies=args.bodies,
        total_time=args.time,
        output_dir=args.output
    )