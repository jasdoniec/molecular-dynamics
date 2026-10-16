use molecular_dynamics::cpu_simulation::CpuSimulation;
use molecular_dynamics::md_simulation::MdSimulation;
use molecular_dynamics::render::Visualizer;
use molecular_dynamics::simulation_state::{ParticleInit, SimulationState};

fn main() {
    let env_size = 16.0;

    // Two species so the per-particle colour and radius are visible:
    // small blue particles and bigger, heavier orange ones.
    let state = SimulationState {
        particles: vec![
            (
                1000,
                ParticleInit {
                    sigma: 1.0,
                    epsilon: 1.0,
                    mass: 1.0,
                    color: [0.26, 0.52, 0.96, 1.0], // blue
                },
            ),
            (
                250,
                ParticleInit {
                    sigma: 1.4,
                    epsilon: 1.0,
                    mass: 2.0,
                    color: [0.98, 0.45, 0.13, 1.0], // orange
                },
            ),
        ],
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
