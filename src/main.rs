use molecular_dynamics::cpu_simulation::CpuSimulation;
use molecular_dynamics::md_simulation::MdSimulation;
use molecular_dynamics::render::Visualizer;
use molecular_dynamics::simulation_state::{ParticleInit, SimulationState};

fn main() {
    let env_size = 16.0;

    // Temperatures and well depths are in Kelvin. Each species then sits at
    // its own reduced temperature: at 120 K the argon-like blue particles
    // are near T* = 1.0 (fluid), while the xenon-like orange ones are at
    // T* ≈ 0.52 (colder than their well depth -> they cluster).
    let state = SimulationState {
        particles: vec![
            (
                1000,
                ParticleInit {
                    sigma: 1.0,
                    epsilon: 119.8, // argon-like well depth (epsilon/k_B)
                    mass: 1.0,
                    color: [0.26, 0.52, 0.96, 1.0], // blue
                },
            ),
            (
                250,
                ParticleInit {
                    sigma: 1.4,
                    epsilon: 229.0, // xenon-like well depth (epsilon/k_B)
                    mass: 2.0,
                    color: [0.98, 0.45, 0.13, 1.0], // orange
                },
            ),
        ],
        temperature: 120.0,
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
