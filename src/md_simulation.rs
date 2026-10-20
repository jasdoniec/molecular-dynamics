/// Per-particle data the visualizer needs to paint one frame.
pub struct ParticlePaint {
    /// World-space position, folded into `[0, box_length)`.
    pub position: [f32; 3],
    /// Billboard radius in world units (the simulation decides this —
    /// typically derived from sigma so different species render at
    /// their true size).
    pub radius: f32,
    /// RGBA colour in `[0, 1]`, assigned once at simulation setup.
    pub color: [f32; 4],
}

pub trait MdSimulation {
    fn simulation_step(&mut self) -> Vec<ParticlePaint>;

    /// Instantaneous kinetic temperature of the current state, in Kelvin:
    /// `(2 / 3N) * Σ ½·m·v² * (ε_ref / k_B)` with `k_B = 1` internally.
    ///
    /// This is the *measured* temperature — for an energy-conserving run it
    /// drifts away from the setup target as the system equilibrates.
    fn temperature_kelvin(&self) -> f32;

    fn change_temp(&mut self, change: f32);

    fn add_particle(&mut self, particle: ParticleInit);

    fn remove_particle(&mut self, usize: num);
}
