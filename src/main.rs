use molecular_dynamics::cpu_simulation::CpuSimulation;
use molecular_dynamics::md_simulation::MdSimulation;
use molecular_dynamics::render::Visualizer;
use molecular_dynamics::simulation_state::{ParticleInit, SimulationState};

fn main() {
    let env_size = 14.0;
    let num_particles = 1500;

    let state = SimulationState {
        particles: vec![(
            num_particles,
            ParticleInit {
                sigma: 1.0,
                epsilon: 1.0,
                mass: 1.0,
            },
        )],
        temperature: 1.0,
        env_size,
    };

    let engine = CpuSimulation::new(0.002, 4, state)
        .expect("failed to set up the simulation");

    Visualizer::new(engine, env_size).run();
}

/// Headless helper: run a simulation for a number of steps without rendering.
#[allow(dead_code)]
fn run_simulation<E: MdSimulation>(mut engine: E, num_steps: usize) {
    for _ in 0..num_steps {
        engine.simulation_step();
    }
}
