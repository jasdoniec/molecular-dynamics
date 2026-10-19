/// The per-species parameters the force kernels work with: the physics
/// half of a particle, kept apart from positions and velocities (which
/// belong to the state/neighbor machinery). `Copy` so it travels by
/// value — the same shape a WGSL struct will have when the kernel is
/// ported to a compute shader.
#[derive(Clone, Copy)]
pub struct ParticleParams {
    /// Well depth of this species in reduced units (the Kelvin → reduced
    /// normalization happens in
    /// [`crate::cpu_simulation::CpuSimulation::new`]).
    pub epsilon: f32,
    /// Interaction size of this species.
    pub sigma: f32,
}

/// The force kernel: the per-pair physics a simulation runs on.
///
/// Everything around it — cell lists, Verlet lists, the integrator, the
/// thermostat — is force-agnostic: the engine only asks the kernel
/// "what is the force on this particle from that one?". New physics
/// therefore lands here as new variants and new `match` arms, never as
/// edits to the engine:
///
/// - ion interactions will add an LJ + Coulomb variant — the same
///   Lennard-Jones math *plus* a `k·qᵢ·qⱼ/r²` term, because ions still
///   need the short-range repulsion and dispersion LJ already provides;
/// - a trained neural-network potential later *replaces* the classical
///   terms entirely (the network has learned repulsion/dispersion from
///   its DFT training data), so no LJ is evaluated in that mode.
///
/// On the GPU this enum becomes a `switch` inside the force compute
/// shader.
pub enum Potential {
    /// Pure Lennard-Jones: per-species ε and σ, mixed per pair with the
    /// standard rules (arithmetic mean σ, geometric mean ε).
    LennardJones,
}

impl Potential {
    /// Force on the particle described by `a` due to the particle
    /// described by `b`.
    ///
    /// `disp` is the pair's minimum-image displacement (`a − b`, wrapped
    /// to ±box/2) and `distance` its length. The engine owns the
    /// boundary-condition wrapping, which keeps every kernel a pure pair
    /// function over a local displacement — exactly the input a compute
    /// shader wants.
    pub fn pair_force(
        &self,
        a: ParticleParams,
        b: ParticleParams,
        disp: [f32; 3],
        distance: f32,
    ) -> [f32; 3] {
        match self {
            Potential::LennardJones => lennard_jones(a, b, disp, distance),
        }
    }
}

/// `|F| = 24 ε / r · (2 (σ/r)^12 − (σ/r)^6)` along the pair axis, with
/// the mixing rules applied per pair. Exact overlap is guarded (it
/// would otherwise divide by zero and NaN the whole run).
fn lennard_jones(a: ParticleParams, b: ParticleParams, disp: [f32; 3], distance: f32) -> [f32; 3] {
    if distance < 1e-6 {
        return [0.0; 3];
    }

    let sigma = (a.sigma + b.sigma) / 2.0;
    let epsilon = (a.epsilon * b.epsilon).sqrt();

    let ratio = sigma / distance;

    let magnitude = 24.0 * epsilon / distance * (2.0 * ratio.powi(12) - ratio.powi(6));

    let mut f = [0.0; 3];
    for i in 0..3 {
        f[i] = magnitude * (disp[i] / distance);
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: ParticleParams = ParticleParams {
        epsilon: 1.0,
        sigma: 1.0,
    };

    /// Force between `a` and `b` at separation `r` along +x.
    fn force_at(r: f32, a: ParticleParams, b: ParticleParams) -> [f32; 3] {
        Potential::LennardJones.pair_force(a, b, [r, 0.0, 0.0], r)
    }

    /// The LJ force crosses zero at the potential minimum r = 2^(1/6)·σ,
    /// is repulsive (positive along `a − b`) below it and attractive
    /// above it; at r = σ the closed form gives |F| = 24ε/σ.
    #[test]
    fn lj_force_signs_and_scale() {
        let r_min = 2.0f32.powf(1.0 / 6.0);

        let f = force_at(r_min, A, A);
        assert!(
            f[0].abs() < 1e-5,
            "force must vanish at the potential minimum, got {f:?}"
        );

        assert!(
            force_at(0.9 * r_min, A, A)[0] > 0.0,
            "a compressed pair must repel"
        );
        assert!(
            force_at(1.5 * r_min, A, A)[0] < 0.0,
            "a stretched pair must attract"
        );

        let f = force_at(1.0, A, A);
        assert!(
            (f[0] - 24.0).abs() < 1e-3,
            "F(σ) must equal 24ε/σ = 24, got {f:?}"
        );
    }

    /// Mixing rules: arithmetic mean σ, geometric mean ε. A wrong mixing
    /// rule shifts the zero crossing, so probing it pins both at once.
    #[test]
    fn lj_mixing_rules() {
        let b = ParticleParams {
            epsilon: 4.0,
            sigma: 3.0,
        };
        // σ_mix = (1+3)/2 = 2, ε_mix = √(1·4) = 2 → zero at 2^(7/6).
        let r_min = 2.0 * 2.0f32.powf(1.0 / 6.0);
        let f = force_at(r_min, A, b);
        assert!(
            f[0].abs() < 1e-4,
            "mixed pair must vanish at 2^(1/6)·σ_mix, got {f:?}"
        );
    }

    /// Two exactly overlapping particles must give zero force, not NaN —
    /// the guard the long-run regression test has always relied on.
    #[test]
    fn lj_guards_exact_overlap() {
        let f = Potential::LennardJones.pair_force(A, A, [0.0; 3], 0.0);
        assert_eq!(f, [0.0; 3]);
    }
}
