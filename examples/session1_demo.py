"""
Example: Complete Session 1 demonstration

This example shows how to use all Session 1 features together:
1. Propagate orbits with perturbations
2. Simulate formation flying
3. Compute laser phase measurements
4. Add realistic noise
5. Analyze noise with Allan deviation

This demonstrates a realistic gravimetry mission scenario.
"""

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from sim.dynamics import (
    two_body_dynamics,
    perturbed_dynamics,
    propagate_orbit_jax,
    propagate_relative_orbit,
    inertial_to_hill_frame,
)

from interferometry import (
    compute_phase_from_states,
    compute_phase_rate_from_states,
)

from interferometry.noise import (
    total_phase_noise_std,
    generate_noise_realization,
    noise_equivalent_range,
)

from interferometry.allan import (
    overlapping_allan_deviation,
    identify_noise_type,
)


def main():
    """Run complete Session 1 demonstration."""
    
    print("=" * 70)
    print("SESSION 1 DEMONSTRATION: Gravimetry Mission Simulation")
    print("=" * 70)
    print()
    
    # ========================================================================
    # PART 1: Orbital Dynamics
    # ========================================================================
    print("PART 1: Propagating GRACE-FO-like orbits")
    print("-" * 70)
    
    # Initial conditions for leader satellite (LEO, ~490 km altitude)
    r0_leader = jnp.array([6871.0, 0.0, 0.0])  # km
    v0_leader = jnp.array([0.0, 7.6, 0.0])      # km/s
    state0_leader = jnp.concatenate([r0_leader, v0_leader])
    
    # Propagate with J2 and drag for one orbit
    print("  Propagating leader orbit with J2 and drag...")
    
    def dynamics_perturbed(t, state):
        return perturbed_dynamics(
            t, state,
            include_j2=True,
            include_drag=True,
            cd=2.2,
            area_to_mass=0.01,  # m²/kg
        )
    
    # Propagate for ~1.5 hours (one orbit)
    times, states_leader = propagate_orbit_jax(
        dynamics_perturbed,
        state0_leader,
        t_span=(0.0, 5400.0),  # 90 minutes
        dt=10.0  # 10 second steps
    )
    
    print(f"  ✓ Propagated {len(times)} steps over {times[-1]/60:.1f} minutes")
    print(f"  Initial altitude: {jnp.linalg.norm(state0_leader[:3]) - 6378:.1f} km")
    print(f"  Final altitude: {jnp.linalg.norm(states_leader[-1, :3]) - 6378:.1f} km")
    print()
    
    # ========================================================================
    # PART 2: Formation Flying
    # ========================================================================
    print("PART 2: Simulating formation flying (220 km separation)")
    print("-" * 70)
    
    # Initial relative state: 220 km along-track separation
    delta_state0 = jnp.array([0.0, 220.0, 0.0, 0.0, 0.0, 0.0])  # km, km/s
    
    # Mean motion for leader orbit
    a_leader = jnp.linalg.norm(state0_leader[:3])
    mu = 398600.4418  # km³/s²
    n = jnp.sqrt(mu / (a_leader ** 3))
    
    print(f"  Leader orbital radius: {a_leader:.1f} km")
    print(f"  Mean motion: {n*1000:.4f} mrad/s")
    print(f"  Orbital period: {2*jnp.pi/n/60:.1f} minutes")
    
    # Propagate relative motion (Hill-Clohessy-Wiltshire)
    print("  Propagating relative orbit...")
    times_rel, delta_states = propagate_relative_orbit(
        delta_state0, n,
        t_span=(0.0, 5400.0),
        dt=10.0
    )
    
    # Extract relative positions
    delta_x = delta_states[:, 0]  # Radial
    delta_y = delta_states[:, 1]  # Along-track
    delta_z = delta_states[:, 2]  # Cross-track
    
    print(f"  ✓ Relative orbit propagated")
    print(f"  Initial separation: {jnp.linalg.norm(delta_state0[:3]):.1f} km")
    print(f"  Final separation: {jnp.linalg.norm(delta_states[-1, :3]):.1f} km")
    print(f"  Separation range: {jnp.min(jnp.linalg.norm(delta_states[:, :3], axis=1)):.1f} - {jnp.max(jnp.linalg.norm(delta_states[:, :3], axis=1)):.1f} km")
    print()
    
    # ========================================================================
    # PART 3: Laser Phase Measurements
    # ========================================================================
    print("PART 3: Computing laser interferometry measurements")
    print("-" * 70)
    
    # For simplicity, use a subset of samples for phase computation
    n_phase_samples = 100
    step = len(times) // n_phase_samples
    indices = jnp.arange(0, len(times), step)[:n_phase_samples]
    
    # Compute follower positions in inertial frame
    phases = []
    ranges = []
    
    print("  Computing phase measurements...")
    for idx in indices:
        r_leader = states_leader[idx, :3]
        v_leader = states_leader[idx, 3:]
        delta_r = delta_states[idx, :3]
        delta_v = delta_states[idx, 3:]
        
        # Compute range
        range_km = jnp.linalg.norm(delta_r)
        ranges.append(range_km)
        
        # Compute phase (note: this is simplified, ignores frame transformations)
        # In practice, you'd transform delta_r to inertial first
        # For this demo, we approximate:
        r_follower = r_leader + delta_r  # Approximate inertial position
        v_follower = v_leader + delta_v
        
        phase = compute_phase_from_states(r_leader, r_follower)
        phases.append(phase)
    
    phases = jnp.array(phases)
    ranges = jnp.array(ranges)
    
    print(f"  ✓ Computed {len(phases)} phase measurements")
    print(f"  Range: {jnp.min(ranges):.2f} - {jnp.max(ranges):.2f} km")
    print(f"  Phase range: {jnp.min(phases):.2e} - {jnp.max(phases):.2e} rad")
    print()
    
    # ========================================================================
    # PART 4: Noise Budget Analysis
    # ========================================================================
    print("PART 4: Computing noise budget (GRACE-FO-like parameters)")
    print("-" * 70)
    
    # GRACE-FO-like parameters
    power = 10e-12  # 10 pW received power
    range_km = jnp.mean(ranges)
    range_rate_km_s = 0.001  # ~1 m/s typical relative velocity
    
    # Compute total noise and breakdown
    total_noise, breakdown = total_phase_noise_std(
        power=power,
        range_km=range_km,
        range_rate_km_s=range_rate_km_s,
        frequency_stability=1e-13,  # Iodine-stabilized laser
        pointing_jitter_rad=10e-6,  # 10 μrad
        clock_stability=1e-12,       # USO
        acceleration_noise=3e-11,    # m/s²/√Hz at 0.1 Hz
        measurement_frequency=0.1,
        bandwidth=1.0,
    )
    
    print("  Noise Breakdown:")
    for source, noise in breakdown.items():
        if source != 'total':
            range_noise = noise_equivalent_range(noise)
            print(f"    {source:20s}: {noise*1e9:8.2f} nrad/√Hz  ({range_noise*1e9:8.2f} nm/√Hz)")
    
    print(f"    {'TOTAL':20s}: {total_noise*1e9:8.2f} nrad/√Hz  ({noise_equivalent_range(total_noise)*1e9:8.2f} nm/√Hz)")
    print()
    
    # ========================================================================
    # PART 5: Add Noise to Measurements
    # ========================================================================
    print("PART 5: Adding realistic noise to measurements")
    print("-" * 70)
    
    # Generate noise realization
    key = random.PRNGKey(42)
    sample_rate = 1.0 / 10.0  # 10 second sampling -> 0.1 Hz
    noise = generate_noise_realization(key, total_noise, len(phases))
    
    # Add noise to ideal phases
    phases_noisy = phases + noise
    
    print(f"  ✓ Added noise to {len(phases)} measurements")
    print(f"  Ideal phase std: {jnp.std(phases):.2e} rad")
    print(f"  Noisy phase std: {jnp.std(phases_noisy):.2e} rad")
    print(f"  Noise std (actual): {jnp.std(noise):.2e} rad")
    print(f"  Noise std (expected): {total_noise:.2e} rad/√Hz")
    print()
    
    # ========================================================================
    # PART 6: Allan Deviation Analysis
    # ========================================================================
    print("PART 6: Computing Allan deviation for noise characterization")
    print("-" * 70)
    
    # Compute overlapping Allan deviation
    tau_values = jnp.logspace(-1, 2, 20)  # 0.1 to 100 seconds
    
    print(f"  Computing overlapping Allan deviation...")
    print(f"  Tau range: {tau_values[0]:.2f} - {tau_values[-1]:.1f} seconds")
    
    oadev = overlapping_allan_deviation(
        phases_noisy - jnp.mean(phases_noisy),  # Remove mean
        sample_rate=sample_rate,
        tau_values=tau_values
    )
    
    # Identify noise type
    noise_type, slope = identify_noise_type(tau_values, oadev)
    
    print(f"  ✓ Allan deviation computed")
    print(f"  Identified noise type: {noise_type}")
    print(f"  Slope in log-log plot: {slope:.2f}")
    print()
    
    # ========================================================================
    # PART 7: Summary Statistics
    # ========================================================================
    print("=" * 70)
    print("SUMMARY: Session 1 Capabilities Demonstrated")
    print("=" * 70)
    print()
    print("✓ Orbital Dynamics:")
    print("  - Two-body propagation")
    print("  - J2 + drag perturbations")
    print("  - Formation flying (Hill-Clohessy-Wiltshire)")
    print()
    print("✓ Laser Interferometry:")
    print("  - Phase measurements from positions")
    print("  - Range-phase conversions")
    print()
    print("✓ Noise Modeling:")
    print("  - 5 noise sources (shot, frequency, pointing, clock, acceleration)")
    print("  - Noise budget analysis")
    print("  - Realistic noise generation")
    print()
    print("✓ Noise Characterization:")
    print("  - Allan deviation computation")
    print("  - Noise type identification")
    print()
    print("All Session 1 physics models are operational! 🚀")
    print()
    
    return {
        'times': times,
        'states_leader': states_leader,
        'delta_states': delta_states,
        'phases': phases,
        'phases_noisy': phases_noisy,
        'noise_breakdown': breakdown,
        'tau_values': tau_values,
        'allan_dev': oadev,
    }


if __name__ == "__main__":
    results = main()
    print("Example completed successfully!")
