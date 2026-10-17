use dialoguer::Confirm;
use dialoguer::Input;
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

    // Temperatures and well depths are in Kelvin. Each species then sits at
    // its own reduced temperature: at 120 K the argon-like blue particles
    // are near T* = 1.0 (fluid), while the xenon-like orange ones are at
    // T* ≈ 0.52 (colder than their well depth -> they cluster).
    // Two species so the per-particle colour and radius are visible: small blue particles and bigger, heavier orange ones.
    SimulationState {
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
    }
}

fn user_simulation() -> SimulationState {
    let mut particles: Vec<(usize, ParticleInit)> = Vec::new();
    let mut options: Vec<(String, ParticleInit, usize)> = vec![
        (
            "Helium".to_string(),
            ParticleInit {
                sigma: 2.55,
                epsilon: 10.2,
                mass: 1.0,
                color: [1.0, 0.55, 0.20, 1.0],
            },
            0,
        ),
        (
            "Neon".to_string(),
            ParticleInit {
                sigma: 2.82,
                epsilon: 35.6,
                mass: 5.04,
                color: [1.0, 0.27, 0.0, 1.0],
            },
            0,
        ),
        (
            "Argon".to_string(),
            ParticleInit {
                sigma: 3.405,
                epsilon: 119.8,
                mass: 9.98,
                color: [1.0, 0.27, 0.0, 1.0],
            },
            0,
        ),
        (
            "Krypton".to_string(),
            ParticleInit {
                sigma: 3.66,
                epsilon: 171.0,
                mass: 20.94,
                color: [0.75, 0.85, 1.0, 1.0],
            },
            0,
        ),
        (
            "Xenon".to_string(),
            ParticleInit {
                sigma: 4.05,
                epsilon: 221.0,
                mass: 32.80,
                color: [0.25, 0.50, 1.0, 1.0],
            },
            0,
        ),
    ];

    let mut display: Vec<String> = options.iter().map(|(name, _, _)| name.clone()).collect();

    let done = "Done".to_string();

    display.push(done);

    loop {
        let selection = Select::new()
            .with_prompt("Pick a particle to add")
            .items(&display) // Dialoguer reads the string titles perfectly here
            .default(0)
            .interact()
            .unwrap();

        if selection == 5 {
            break;
        }

        let num = Input::<usize>::new()
            .with_prompt(format!("How many {} particles", display[selection]))
            .default(100)
            .interact_text()
            .unwrap();

        options[selection].2 = num;
    }

    for i in 0..5 {
        if options[i].2 != 0 {
            particles.push((options[i].2, options[i].1.clone()));
        }
    }

    let temperature = Input::<f32>::new()
        .with_prompt("Select temperature")
        .default(100.0)
        .interact_text()
        .unwrap();

    let env_size = Input::<f32>::new()
        .with_prompt("Select the box width")
        .interact_text()
        .unwrap();

    SimulationState {
        particles,
        temperature,
        env_size,
    }
}
