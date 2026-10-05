pub trait MdSimulation {
    fn simulation_step(&mut self) -> Vec<[f32; 3]>;
}
