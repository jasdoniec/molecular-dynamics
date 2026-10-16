pub struct SimulationState {
    pub particles: Vec<(usize, ParticleInit)>,
    pub temperature: f32,
    pub env_size: f32,
}

#[derive(Clone)]
pub struct ParticleInit {
    pub sigma: f32,
    pub epsilon: f32,
    pub mass: f32,
    /// RGBA colour used by the visualizer; assigned to every particle of
    /// this species at setup.
    pub color: [f32; 4],
}
