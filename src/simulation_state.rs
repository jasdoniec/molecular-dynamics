pub struct SimulationState {
    pub particles: Vec<(usize, ParticleInit)>,
    /// Target temperature in **Kelvin**. Applied once at setup (the run
    /// itself is energy-conserving, so the measured temperature drifts as
    /// the system equilibrates).
    pub temperature: f32,
    pub env_size: f32,
}

#[derive(Clone)]
pub struct ParticleInit {
    pub sigma: f32,
    /// Well depth of this species expressed as **ε/k_B in Kelvin** (the way
    /// it is usually tabulated, e.g. ~119.8 for argon, ~229 for xenon).
    ///
    /// Internally the simulation normalizes all well depths by the strongest
    /// species' epsilon, so mixing species with very different values is
    /// fine — each species then sits at its own reduced temperature
    /// T* = T_kelvin / (epsilon/k_B) while sharing one physical temperature.
    pub epsilon: f32,
    pub mass: f32,
    /// RGBA colour used by the visualizer; assigned to every particle of
    /// this species at setup.
    pub color: [f32; 4],
}
