use dialoguer::Confirm;
use dialoguer::Select;
use molecular_dynamics::cpu_simulation::CpuSimulation;
use molecular_dynamics::md_simulation::MdSimulation;
use molecular_dynamics::render::Visualizer;
use molecular_dynamics::simulation_state::{ParticleInit, SimulationState};

fn main() {
    let is_demo = Confirm::new()
        .with_prompt("Run the demo(y) or make your own (n)")
        .interact()
        .unwrap();

    let state = if is_demo { demo() } else { user_simulation() };

    let env_size = state.env_size;

    let engine = CpuSimulation::new(0.002, 4, state).expect("failed to set up the simulation");

    Visualizer::new(engine, env_size).run();
}

/// Headless helper: run a simulation for a number of steps without rendering.
#[allow(dead_code)]
fn run_simulation<E: MdSimulation>(mut engine: E, num_steps: usize) {
    for _ in 0..num_steps {
        engine.simulation_step();
    }
}

fn demo() -> SimulationState {
    let env_size = 16.0;

    // Two species so the per-particle colour and radius are visible:
    // small blue particles and bigger, heavier orange ones.
    SimulationState {
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
    }
}

fn user_simulation() -> SimulationState {
    let particles: Vec<(usize, ParticleInit)> = Vec::new();
    let options = vec![
        ("Helium", ParticleInit{
            sigma: 2.55,
        }),
        ("Neon", ParticleInit{
            sigma: 2.82,
        }),
        ("Argon", ParticleInit{
            sigma: 3.405,
        }),
        ("Krypton", ParticleInit{
            sigma: 3.66,
        }),
        ("Xenon", ParticleInit{
            sigma: 4.05,
        }),
        ("Radon", ParticleInit{
            sigma:
        })
    ]
}
