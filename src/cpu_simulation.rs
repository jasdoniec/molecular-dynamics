pub struct CpuSimulation {
    particles: Vec<crate::Particle>,
}

impl crate::MdSimulation for CpuSimulation {
    fn simulation_step(&mut self) -> Vec<[f32; 3]> {
        vec![[0 as f32, 0 as f32, 0 as f32]]
    }
}

impl CpuSimulation {
    fn position_calculation(&mut self) {}
}
