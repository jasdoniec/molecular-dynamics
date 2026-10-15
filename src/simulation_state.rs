pub struct SimulationState {
    pub particles: Vec<(usize, ParticleInit)>,
    pub temperature: f32,
    pub env_size: f32,
}

pub struct ParticleInit {
    pub sigma: f32,
    pub epsilon: f32,
    pub mass: f32,
}
