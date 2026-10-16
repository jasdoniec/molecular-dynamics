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
}
