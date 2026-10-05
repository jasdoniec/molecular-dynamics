use molecular_dynamics::md_simulation::MdSimulation;

fn main() {
    println!("Hello, world!");
}

fn run_simulation<E: MdSimulation>(mut engine: E, num_steps: usize) {
    for _ in 0..num_steps {
        engine.simulation_step();
    }
}
