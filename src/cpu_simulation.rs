use crate::particle;

pub struct CpuSimulation {
    particles: Vec<crate::Particle>,
}

impl crate::MdSimulation for CpuSimulation {
    fn simulation_step(&mut self) -> Vec<[f32; 3]> {
        vec![[0 as f32, 0 as f32, 0 as f32]]
    }
}

impl CpuSimulation {
    fn position_calculation(&mut self, delta_time: f32) {
        for particle in &mut self.particles {
            for i in 0..3 {
                particle.curr_position[i] = particle.prev_postion[i]
                    + particle.prev_velocity[i] * delta_time
                    + particle.prev_acceleration[i] * delta_time * delta_time / 2.0;
            }
        }
    }

    fn force_acc_velocity_calculation(&mut self) {
        let length = self.particles.len() / 2;

        for i in 0..length {}
    }
}
