#!/usr/bin/env python3
"""
Numerical Accuracy Analysis: Timestep Comparison (Improved Version)
====================================================================

Key improvements over v1:
1. Adaptive softening based on particle count
2. Smarter timestep selection for stability
3. Virialized initial conditions (stable system)
4. Configurable experiment parameters
5. Suppressed Numba warnings

Author: Andrey Maltsev
Usage: python3 timestep_analysis_v3.py -n 1000 -t 10.0
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import time
import os
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Suppress Numba performance warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Configure matplotlib
import matplotlib
matplotlib.use('Agg')
plt.style.use('dark_background')

# =============================================================================
# CONSTANTS
# =============================================================================
G = 1.0

COLORS = {
    'fine': '#96CEB4',
    'medium': '#4ECDC4',
    'coarse': '#FF6B6B',
    'accent': '#FFEAA7',
    'grid': '#2C3E50'
}


def calculate_softening(n_bodies: int, radius: float = 50.0) -> float:
    """
    Calculate appropriate softening parameter based on system size.
    
    Softening should be ~mean inter-particle distance / 10
    to prevent singularities while preserving physics.
    """
    # Mean inter-particle distance in sphere
    volume = (4/3) * np.pi * radius**3
    mean_separation = (volume / n_bodies) ** (1/3)
    
    # Softening = fraction of mean separation
    softening = mean_separation * 0.1
    
    return max(softening, 0.01)  # Minimum softening


def calculate_safe_timestep(n_bodies: int, radius: float = 50.0, 
                            g: float = 1.0, safety_factor: float = 0.1) -> float:
    """
    Calculate safe timestep based on dynamical time.
    
    Dynamical time ~ sqrt(R³ / (G*M))
    Safe dt ~ dynamical_time * safety_factor
    """
    # Estimate total mass (normalized masses average ~1)
    total_mass = n_bodies * 1.0
    
    # Dynamical time
    t_dyn = np.sqrt(radius**3 / (g * total_mass))
    
    # Safe timestep (fraction of dynamical time)
    dt = t_dyn * safety_factor
    
    return min(dt, 0.01)  # Cap at 0.01 for safety


# =============================================================================
# CUDA KERNELS
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
# SIMULATOR
# =============================================================================

class TimestepSimulator:
    """GPU simulator with configurable softening."""
    
    def __init__(self, n_bodies: int, dt: float, softening: float):
        self.n_bodies = n_bodies
        self.dt = dt
        self.softening = softening
        self.G = G
        
        # CUDA config
        if n_bodies <= 256:
            self.threads = 64
        elif n_bodies <= 1024:
            self.threads = 128
        else:
            self.threads = 256
        self.blocks = (n_bodies + self.threads - 1) // self.threads
        
    def initialize(self, positions: np.ndarray, velocities: np.ndarray, 
                   masses: np.ndarray):
        self.positions = positions.astype(np.float32)
        self.velocities = velocities.astype(np.float32)
        self.masses = masses.astype(np.float32)
        
        self.d_positions = cuda.to_device(self.positions)
        self.d_velocities = cuda.to_device(self.velocities)
        self.d_masses = cuda.to_device(self.masses)
        self.d_accelerations = cuda.device_array_like(self.velocities)
    
    def step(self):
        compute_forces_kernel[self.blocks, self.threads](
            self.d_positions, self.d_masses, self.d_accelerations,
            self.n_bodies, self.G, self.softening
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
        positions = self.get_positions()
        velocities = self.get_velocities()
        
        kinetic = 0.5 * np.sum(self.masses[:, np.newaxis] * velocities**2)
        
        potential = 0.0
        for i in range(self.n_bodies):
            for j in range(i+1, self.n_bodies):
                r = np.linalg.norm(positions[i] - positions[j])
                if r > self.softening:
                    potential -= self.G * self.masses[i] * self.masses[j] / r
        
        return kinetic, potential, kinetic + potential


# =============================================================================
# INITIAL CONDITIONS (Improved - Virialized System)
# =============================================================================

def generate_virialized_sphere(n_bodies: int, seed: int = 42,
                                radius: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a virialized (stable) spherical system.
    
    Virial theorem: 2K + U = 0 (for equilibrium)
    This means total kinetic energy = -0.5 * potential energy
    """
    np.random.seed(seed)
    
    # Positions: uniform density sphere
    phi = np.random.uniform(0, 2*np.pi, n_bodies)
    costheta = np.random.uniform(-1, 1, n_bodies)
    u = np.random.uniform(0, 1, n_bodies)
    r = radius * (u ** (1/3))  # Uniform density
    theta = np.arccos(costheta)
    
    positions = np.zeros((n_bodies, 3), dtype=np.float32)
    positions[:, 0] = r * np.sin(theta) * np.cos(phi)
    positions[:, 1] = r * np.sin(theta) * np.sin(phi)
    positions[:, 2] = r * np.cos(theta)
    
    # Masses: narrow distribution (more stable than log-normal)
    masses = np.random.uniform(0.8, 1.2, n_bodies).astype(np.float32)
    total_mass = np.sum(masses)
    
    # Calculate potential energy for virial scaling
    softening = calculate_softening(n_bodies, radius)
    potential = 0.0
    for i in range(min(n_bodies, 500)):  # Sample for speed
        for j in range(i+1, min(n_bodies, 500)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist > softening:
                potential -= G * masses[i] * masses[j] / dist
    
    # Scale to full system
    if n_bodies > 500:
        scale = (n_bodies / 500) ** 2
        potential *= scale
    
    # Target kinetic energy from virial theorem: K = -U/2
    target_kinetic = -potential / 2
    
    # Generate random velocities
    velocities = np.random.randn(n_bodies, 3).astype(np.float32)
    
    # Scale velocities to match target kinetic energy
    current_kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
    if current_kinetic > 0:
        scale = np.sqrt(target_kinetic / current_kinetic)
        velocities *= scale * 0.5  # Factor 0.5 for extra stability
    
    # Remove net momentum (center of mass stays fixed)
    total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
    velocities -= total_momentum / total_mass
    
    # Central massive body (optional, adds stability)
    if n_bodies > 100:
        positions[0] = [0, 0, 0]
        velocities[0] = [0, 0, 0]
        masses[0] = 5.0  # Moderate central mass
    
    return positions, velocities, masses


def generate_disk_galaxy(n_bodies: int, seed: int = 42,
                          radius: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a rotating disk galaxy (very stable).
    
    Particles orbit a central mass with circular velocities.
    """
    np.random.seed(seed)
    
    positions = np.zeros((n_bodies, 3), dtype=np.float32)
    velocities = np.zeros((n_bodies, 3), dtype=np.float32)
    masses = np.zeros(n_bodies, dtype=np.float32)
    
    # Central black hole
    central_mass = 50.0
    masses[0] = central_mass
    positions[0] = [0, 0, 0]
    velocities[0] = [0, 0, 0]
    
    # Disk particles
    for i in range(1, n_bodies):
        # Exponential disk profile
        r = radius * 0.1 + radius * 0.8 * np.sqrt(np.random.rand())
        theta = np.random.rand() * 2 * np.pi
        
        positions[i, 0] = r * np.cos(theta)
        positions[i, 1] = r * np.sin(theta)
        positions[i, 2] = (np.random.rand() - 0.5) * radius * 0.05  # Thin disk
        
        # Circular orbital velocity
        v_circ = np.sqrt(G * central_mass / r)
        velocities[i, 0] = -v_circ * np.sin(theta)
        velocities[i, 1] = v_circ * np.cos(theta)
        velocities[i, 2] = 0
        
        # Small velocity dispersion
        velocities[i] += np.random.randn(3) * v_circ * 0.05
        
        masses[i] = np.random.uniform(0.01, 0.05)
    
    return positions, velocities, masses


# =============================================================================
# EXPERIMENT FRAMEWORK
# =============================================================================

@dataclass
class TimestepExperiment:
    name: str
    n_steps: int
    dt: float
    
    @property
    def total_time(self) -> float:
        return self.n_steps * self.dt


@dataclass
class ExperimentResult:
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
        if self.initial_energy == 0:
            return 0.0
        return abs(self.final_energy - self.initial_energy) / abs(self.initial_energy) * 100


def run_experiment(positions: np.ndarray, velocities: np.ndarray,
                   masses: np.ndarray, experiment: TimestepExperiment,
                   softening: float, record_interval: int = None) -> ExperimentResult:
    """Run one timestep experiment."""
    n_bodies = len(masses)
    
    if record_interval is None:
        record_interval = max(1, experiment.n_steps // 100)
    
    sim = TimestepSimulator(n_bodies, experiment.dt, softening)
    sim.initialize(positions.copy(), velocities.copy(), masses.copy())
    
    _, _, initial_energy = sim.calculate_energy()
    energy_history = [initial_energy]
    position_history = [positions.copy()]
    
    start = time.perf_counter()
    for step in range(experiment.n_steps):
        sim.step()
        
        if (step + 1) % record_interval == 0 or step == experiment.n_steps - 1:
            position_history.append(sim.get_positions().copy())
            _, _, energy = sim.calculate_energy()
            energy_history.append(energy)
    
    elapsed = time.perf_counter() - start
    
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
    distances = np.linalg.norm(ref.final_positions - other.final_positions, axis=1)
    com_ref = np.mean(ref.final_positions, axis=0)
    com_other = np.mean(other.final_positions, axis=0)
    
    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'std_distance': np.std(distances),
        'com_difference': np.linalg.norm(com_ref - com_other),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_analysis_plots(results: List[ExperimentResult], output_dir: str,
                          n_bodies: int, softening: float):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    colors = [COLORS['fine'], COLORS['medium'], COLORS['coarse']]
    
    # 1. Final positions
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
    
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Perfect')
    ax2.set_xlabel('Simulation Time', fontsize=11)
    ax2.set_ylabel('E / E₀ (Normalized)', fontsize=11)
    ax2.set_title('Energy Conservation Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # 3. Position error
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a1a')
    
    reference = results[0]
    labels, mean_errors, max_errors = [], [], []
    
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
    ax3.set_ylabel('Position Error vs Fine', fontsize=11)
    ax3.set_title('Numerical Drift Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # 4. Energy drift
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
    
    fig.suptitle(f'Timestep Size vs Numerical Accuracy\n'
                 f'N={n_bodies} bodies | Softening={softening:.4f} | Virialized System',
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
    """Create trajectory comparison."""
    
    fig = plt.figure(figsize=(12, 10), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a1a')
    
    colors = [COLORS['fine'], COLORS['medium'], COLORS['coarse']]
    
    for p_idx in particle_indices:
        for result, color in zip(results, colors):
            trajectory = np.array([pos[p_idx] for pos in result.position_history])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, linewidth=1, alpha=0.7)
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      color=color, s=50, marker='o', edgecolors='white', zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color=color, s=100, marker='*', edgecolors='white', zorder=5)
    
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
# MAIN
# =============================================================================

def run_analysis(n_bodies: int = 1000,
                 total_time: float = 10.0,
                 output_dir: str = '../docs',
                 system_type: str = 'virialized'):
    """
    Run improved timestep analysis.
    
    Args:
        n_bodies: Number of particles
        total_time: Total simulation time
        output_dir: Output directory for plots
        system_type: 'virialized' (stable sphere) or 'disk' (galaxy)
    """
    
    print("=" * 70)
    print("TIMESTEP ACCURACY ANALYSIS (Improved Version)")
    print("=" * 70)
    
    if not cuda.is_available():
        print("\nERROR: CUDA not available!")
        return None
    
    device = cuda.get_current_device()
    print(f"\nGPU: {device.name.decode()}")
    print(f"Bodies: {n_bodies}")
    print(f"Total time: {total_time}")
    print(f"System type: {system_type}")
    
    # Calculate appropriate parameters
    softening = calculate_softening(n_bodies)
    base_dt = calculate_safe_timestep(n_bodies)
    
    print(f"\nAuto-calculated parameters:")
    print(f"  Softening: {softening:.4f}")
    print(f"  Base dt: {base_dt:.4f}")
    
    # Generate initial conditions
    print(f"\nGenerating {system_type} initial conditions...")
    if system_type == 'disk':
        positions, velocities, masses = generate_disk_galaxy(n_bodies)
    else:
        positions, velocities, masses = generate_virialized_sphere(n_bodies)
    
    # Define experiments with appropriate timesteps
    # Fine: base_dt, Medium: 10×, Coarse: 100×
    experiments = [
        TimestepExperiment(
            f"Fine: {int(total_time/base_dt)}×{base_dt:.4f}", 
            n_steps=int(total_time / base_dt), 
            dt=base_dt
        ),
        TimestepExperiment(
            f"Medium: {int(total_time/(base_dt*10))}×{base_dt*10:.4f}", 
            n_steps=int(total_time / (base_dt * 10)), 
            dt=base_dt * 10
        ),
        TimestepExperiment(
            f"Coarse: {int(total_time/(base_dt*100))}×{base_dt*100:.4f}", 
            n_steps=max(1, int(total_time / (base_dt * 100))), 
            dt=base_dt * 100
        ),
    ]
    
    # Run experiments
    results = []
    print("\n" + "-" * 70)
    
    for exp in experiments:
        print(f"\nRunning: {exp.name}")
        print(f"  Steps: {exp.n_steps}, dt: {exp.dt:.4f}")
        
        result = run_experiment(positions, velocities, masses, exp, softening)
        
        print(f"  Time: {result.computation_time:.3f}s")
        print(f"  Energy drift: {result.energy_drift_percent:.4f}%")
        results.append(result)
    
    # Create visualizations
    print("\n" + "-" * 70)
    print("Creating visualizations...")
    
    plot_path = create_analysis_plots(results, output_dir, n_bodies, softening)
    traj_path = create_trajectory_plot(results, [0, 1, 5, 10], output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    reference = results[0]
    print(f"\nReference: {reference.experiment.name}")
    print(f"  Initial energy: {reference.initial_energy:.4f}")
    print(f"  Final energy: {reference.final_energy:.4f}")
    print(f"  Energy drift: {reference.energy_drift_percent:.4f}%")
    
    for result in results[1:]:
        print(f"\n{result.experiment.name}:")
        print(f"  Energy drift: {result.energy_drift_percent:.4f}%")
        
        diff = compute_position_difference(reference, result)
        print(f"  Mean position error: {diff['mean_distance']:.4f}")
        print(f"  Max position error: {diff['max_distance']:.4f}")
    
    # Assessment
    print("\n" + "=" * 70)
    print("STABILITY ASSESSMENT")
    print("=" * 70)
    
    fine_drift = results[0].energy_drift_percent
    if fine_drift < 1.0:
        status = "✓ STABLE"
        color = "green"
    elif fine_drift < 10.0:
        status = "⚠ MARGINALLY STABLE"
        color = "yellow"
    else:
        status = "✗ UNSTABLE"
        color = "red"
    
    print(f"\nFine timestep energy drift: {fine_drift:.2f}% → {status}")
    print(f"\nFor better results, try:")
    print(f"  - Smaller base_dt (current: {base_dt:.4f})")
    print(f"  - Larger softening (current: {softening:.4f})")
    print(f"  - Disk galaxy system (--system disk)")
    
    return {
        'results': results,
        'plots': [plot_path, traj_path],
        'softening': softening,
        'base_dt': base_dt
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved timestep analysis')
    parser.add_argument('-n', '--bodies', type=int, default=1000,
                        help='Number of bodies (default: 1000)')
    parser.add_argument('-t', '--time', type=float, default=10.0,
                        help='Total simulation time (default: 10.0)')
    parser.add_argument('-o', '--output', type=str, default='../docs',
                        help='Output directory (default: ../docs)')
    parser.add_argument('--system', type=str, default='virialized',
                        choices=['virialized', 'disk'],
                        help='System type: virialized (stable sphere) or disk (galaxy)')
    
    args = parser.parse_args()
    
    run_analysis(
        n_bodies=args.bodies,
        total_time=args.time,
        output_dir=args.output,
        system_type=args.system
    )
