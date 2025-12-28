#!/usr/bin/env python3
"""
Numerical Accuracy Analysis: Timestep Comparison
=================================================

This module investigates how timestep size affects simulation accuracy.
Compares fine vs coarse timesteps over the same total simulation time.

Experiment: 
- Same total simulated time, different granularity
- Fine:   1000 steps × dt=0.01  → Total time = 10
- Coarse: 10 steps × dt=1.0    → Total time = 10

Key findings demonstrate:
1. Euler integration drift with large timesteps
2. Energy conservation vs computational efficiency tradeoff
3. Phase errors in orbital mechanics

Author: Andrey Maltsev
"""

import numpy as np
import matplotlib .pyplot as plt
import matplotlib.animation as  animation
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbody_cuda import (
    NBodySimulatorGPU, SimulationConfig,
    generate_galaxy_disk, generate_random_particles
)
from numba import cuda

import matplotlib
matplotlib.use('Agg')
plt.style.use('darl_background')

#dataclass
class TimestepExperiment:
    """Configuration for timestep experiment."""
    name: str
    n_steps: int
    dt: float

    @property
    def total_time(self) -> float:
        return self.n_steps *self.dt

@dataclass    
class ExperimentResult:
    """Results from a timestep experiment."""
    experiment: TimestepExperiment
    final_positions: np.ndarray
    final_velocities: np.ndarray
    final_energy: float
    initial_energy: float
    energy_history: List[float]
    position_history: List[np.ndarray]

    @property
    def energy_drift_percent(self) -> float:
        return abs(self.final_energy - self.initial_energy) / abs(self.initial_energy) * 100

    @property
    def total_time(self) -> float:
        return self.experiment.total_time

def run_timestep_experiment(positions: np.ndarray,
                           velocities: np.ndarray,
                           masses: np.ndarray,
                           experiment: TimestepExperiment,
                           record_interval: int=10) -> ExperimentResult:
    """
    Run simulation with specific timestep configuration.
    
    Args:
        positions: Initial positions (N, 3)
        velocities: Initial velocities (N, 3)
        masses: Particle masses (N,)
        experiment: Timestep configuration
        record_interval: How often to record state
        
    Returns:
        ExperimentResult with full history
    """
    config = SimulationConfig(
        n_particles=len(masses),
        dt=experiment.dt
    )

    simulator = NBodySimulatorGPU(config, version=3)
    simulator.initialize(positions.copy(), velocities.copy(), masses.copy())

    # Record initial state
    ke, pe, initial_energy = simulator.compute_energy()
    energy_history = [initial_energy]
    position_history = [positions.copy()]

    # Run simulation
    for step in range(experiment.n_steps):
        simulator.step()

        if (step + 1) % record_interval == 0 or step == experiment.n_steps -1:
            pos, vel = simulator.get_state()
            position_history.append(pos.copy())
            ke, pe, te = simulator.compute_energy()
            energy_history.append(te)

    # Final state
    final_pos, final_vel = simulator.get_state()
    ke, pe, final_energy = simulator.compute_energy()

    return ExperimentResult(
       experiment=experiment,
        final_positions=final_pos,
        final_velocities=final_vel,
        final_energy=final_energy,
        initial_energy=initial_energy,
        energy_history=energy_history,
        position_history=position_history 
    )

def compute_position_difference(result1: ExperimentResult,
                               result2: ExperimentResult) -> Dict:
    """
    Compute difference between final states of two experiments.
    
    Returns:
        Dictionary with various difference metrics
    """
    pos1 = result1.final_positions
    pos2 = result2.final_positions

    # Per-particle distance
    distances = np.linalg.norm(pos1 - pos2, axis=1)

    # Center of mass difference
    com1 = np-mean(pos1, axis=0)
    com2 = np.mean(pos2, axis=0)
    com_diff = np.linalg.norm(com1 - com2)

    return {
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'std_distance': np.std(distances),
        'com_difference': com_diff,
        'per_particle_distances': distances
    }

def run_timestep_comparison(n_particles: int = 100,
                           total_time: float = 10.0,
                           scenarios: List[Tuple[int, float]] = None) -> List[ExperimentResult]:
    """
    Compare multiple timestep configurations.
    
    Args:
        n_particles: Number of particles
        total_time: Total simulation time (same for all)
        scenarios: List of (n_steps, dt) tuples
        
    Returns:
        List of experiment results
    """
    if scenarios is None:
        scenarios = [
            (1000, 0.01),   # Fine: 1000 steps × 0.01
            (100, 0.1),    # Medium: 100 steps × 0.1  
            (10, 1.0),     # Coarse: 10 steps × 1.0
        ]
    # Verify all scenarios have same total time
    for n_steps, dt in scenarios:
        assert abs(n_steps * dt - total_time) < 1e-10, \
            f"Scenario  {n_steps} x {dt} doesn't match total_time = {total_time}"

    print(f"\n{'='*60}")
    print(f"TIMESTEP ACCURACY COMPARISON")
    print(f"N = {n_particles} particles, Total time = {total_time}")
    print(f"{'='*60}")

    # Generate initial condifions (same for all)
    positions, velocities, masses = generate_galaxy_disk(n_particles, seed=42)

    results = []
    for n_steps, dt in scenarios:
        exp = TimestepExperiment(
            name=f"{n_steps} steps x dt = {dt}",
            dt=dt
        )

        print(f"\nRunning: {exp.name}")
        result = run_timestep_experiment(
            positions, velocities, masses, exp,
            record_interval=max(1, n_steps // 100)
        )

        print(f" Energy drift: {result.energy_drift_percent:.4f}%")
        results.append(result)

    return results

def create_timestep_comparison_plot(results: List[ExperimentResult],
                                   output_dir: str = '../docs') -> str:
    """
    Create visualization comparing timestep effects.
    """
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    colors = ['#96CEB4', '#4ECDC4', '#FF6B6B']
    
    # 1. Final position comparison (2D view)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor('#1a1a1a')
    
    for i, (result, color) in enumerate(zip(results, colors)):
        pos = result.final_positions
        ax1.scatter(pos[:, 0], pos[:, 1], c=color, s=10, alpha=0.6,
                   label=result.experiment.name)
    
    ax1.set_xlabel('X Position', fontsize=11)
    ax1.set_ylabel('Y Position', fontsize=11)
    ax1.set_title('Final Particle Positions', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    #2. Energy conservation over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor('#1a1a1a')

    for result, color in zip(results, colors):
        n_points = len(result.energy_history)
        times = np.linspace(0, result.total_time, n_points)
        
        # Normalize to initial energy
        normalized = np.array(result.energy_history) / result.initial_energy
        ax2.plot(times, normalized, color=color, linewidth=2, 
                label=result.experiment.name)
    
    ax2.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Simulation Time', fontsize=11)
    ax2.set_ylabel('E / E₀ (Normalized Energy)', fontsize=11)
    ax2.set_title('Energy Conservation', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Position difference from reference (fine timestep)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor('#1a1a1a')

    reference = results[0] # Fine timestep is reference

    bar_data = []
    labels = []
    for result in results[1:]:
        diff = compute_position_difference(reference, result)
        bar_data.append([diff['mean_distance'], diff['max_distance']])
        labels.append(result.experiment.name)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax3.bar(x - width/2, [d[0] for d in bar_data], width,
                   label='Mean Distance', color = '#4ECDC4')
    bars2 = ax3.bar(x + width/2, [d[1] for d in bar_data], width,
                    label='Max Distance', color='#FF6B6B')
                          
    ax3.set_xlabel('Timestep Configuration', fontsize=11)
    ax3.set_ylabel('Position Difference (vs Fine)', fontsize=11)
    ax3.set_title('Numerical Drift vs Reference', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Energy drift bar chart
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor('#1a1a1a')
    
    drift_values = [r.energy_drift_percent for r in results]
    labels = [r.experiment.name for r in results]
    
    bars = ax4.bar(range(len(results)), drift_values, color=colors,
                   edgecolor='white', linewidth=0.5)
    
    for bar, drift in zip(bars, drift_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{drift:.3f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='white')
    
    ax4.set_xlabel('Timestep Configuration', fontsize=11)
    ax4.set_ylabel('Energy Drift (%)', fontsize=11)
    ax4.set_title('Total Energy Drift', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(results)))
    ax4.set_xticklabels(labels, rotation=15, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Timestep Size vs Numerical Accuracy\nEuler Integration Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'timestep_analysis.png')
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', edgecolor='none',
                bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    plt.close()
    return output_path

def create_trajectory_comparison(results: List[ExperimentResult],
                                particle_indices: List[int] = None,
                                output_dir: str = '../docs') -> str:
    """
    Create animation showing trajectory divergence over time.
    """
    if particle_indices is None:
        particle_indices = [0, 1, 5, 10] # Track specific particles

    fig = plt.figure(figsize=(12, 10), dpi=100)
    fig.patch.set_facecolor('#0a0a0a')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a1a')
    
    colors = ['#96CEB4', '#4ECDC4', '#FF6B6B']
    
    # Get trajectory bounds
    all_x = []
    all_y = []
    for result in results:
        for pos in result.position_history:
            all_x.extend(pos[particle_indices, 0])
            all_y.extend(pos[particle_indices, 1])

    margin = 0.1 * max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Plot trajectories for each particle
    for p_idx in particle_indices:
        for result, color in zip(results, colors):
            trajectory = np.array([pos[p_idx] for pos in result.position_history])
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, linewidth=1, alpha=0.7)
            # Mark start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1], 
                      color=color, s=50, marker='o', edgecolors='white')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color=color, s=100, marker='*', edgecolors='white')
    
    # Legend
    for result, color in zip(results, colors):
        ax.plot([], [], color=color, linewidth=2, label=result.experiment.name)
    ax.legend(loc='upper right')
    
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(f'Trajectory Comparison (Particles {particle_indices})\nSame initial conditions, different timesteps',
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', edgecolor='none',
                bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()
    return output_path


def run_extended_analysis(n_particles: int = 100,
                           output_dir: str = '../docs') -> Dict:
    """
    Run comprehensive timestep analysis with multiple configurations.
    """
    print(f"\n{'='*70}")
    print("EXTENDED TIMESTEP ACCURACY ANALYSIS")
    print(f"{'='*70}")
    
    if not cuda.is_available():
        print("ERROR: CUDA not available!")
        return {}
    
    # Standard comparison (same total time)
    results = run_timestep_comparison(
        n_particles=n_particles,
        total_time=10.0,
        scenarios=[
            (1000, 0.01),   # Fine
            (100, 0.1),    # Medium
            (10, 1.0),     # Coarse
        ]
    )
    
    # Create visualizations
    plot_path = create_timestep_comparison_plot(results, output_dir)
    traj_path = create_trajectory_comparison(results, 
                                               particle_indices=[0, 1, 5, 10],
                                               output_dir=output_dir)
    
    # Print summary
    print(f"\n{'='*70}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    reference = results[0]
    print(f"\nReference: {reference.experiment.name}")
    print(f"  Energy drift: {reference.energy_drift_percent:.6f}%")
    
    for result in results[1:]:
        print(f"\n{result.experiment.name}:")
        print(f"  Energy drift: {result.energy_drift_percent:.4f}%")
        
        diff = compute_position_difference(reference, result)
        print(f"  Mean position error: {diff['mean_distance']:.4f}")
        print(f"  Max position error: {diff['max_distance']:.4f}")
        print(f"  Center of mass drift: {diff['com_difference']:.4f}")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    print("""
1. Euler integration accumulates error proportional to dt²
2. Large timesteps cause:
   - Energy drift (non-conservation)
   - Phase errors in orbital motion
   - Particles may "overshoot" close encounters
   
3. For production simulations, consider:
   - Leapfrog/Verlet integration (symplectic)
   - Adaptive timesteps based on particle separation
   - Higher-order Runge-Kutta methods
   
4. GPU speedup allows using smaller dt without sacrificing real-time
""")
    
    return {
        'results': results,
        'plot_path': plot_path,
        'trajectory_path': traj_path
    }


if __name__ == "__main__":
    run_extended_analysis(n_particles=100, output_dir='../docs')
