use crate::particle;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct CpuSimulation {
    particles: Vec<crate::Particle>,
    env_size: f32,
    r_cut: f32,
    skin: f32,
    cell_list: Box<[Vec<usize>]>,
    list_size: usize,
    delta_time: f32,
    steps_between_paint: usize,
    /// ε/k_B (in Kelvin) of the strongest species — the internal energy
    /// unit. Converts the reduced internal temperature back to Kelvin.
    epsilon_ref_kelvin: f32,
}

impl crate::MdSimulation for CpuSimulation {
    fn simulation_step(&mut self) -> Vec<crate::md_simulation::ParticlePaint> {
        for _ in 0..self.steps_between_paint {
            self.position_calculation();
            if self.check_verlet() {
                self.create_verlet_list();
            }
            self.force_acc_velocity_calculation();
            for particle in &mut self.particles {
                std::mem::swap(&mut particle.curr, &mut particle.prev);
            }
        }

        // After the swap at the end of each physics step the newest state
        // lives in `prev` (returning `curr` would paint one step behind).
        // The radius is derived from sigma so species render at true size.
        self.particles
            .iter()
            .map(|p| crate::md_simulation::ParticlePaint {
                position: p.prev.position,
                radius: p.sigma * 0.5,
                color: p.color,
            })
            .collect()
    }

    fn temperature_kelvin(&self) -> f32 {
        let n = self.particles.len();
        if n == 0 {
            return 0.0;
        }
        // `prev` holds the current state; k_B = 1 in the internal units.
        // sum(m v²) = 2·KE, so T* = sum / (3N).
        let sum_m_v_squared: f32 = self
            .particles
            .iter()
            .map(|p| {
                let v = p.prev.velocity;
                p.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            })
            .sum();
        (sum_m_v_squared / (3.0 * n as f32)) * self.epsilon_ref_kelvin
    }
}

impl CpuSimulation {
    fn r_list(&self) -> f32 {
        self.r_cut + self.skin
    }

    pub fn new(
        delta_time: f32,
        steps_between_paint: usize,
        state: crate::SimulationState,
    ) -> Result<Self, String> {
        let mut state = state;

        let env_size = state.env_size;

        // Kelvin -> reduced units. The strongest species' epsilon (ε/k_B in
        // Kelvin) becomes the internal energy unit: every well depth is
        // divided by it, and so is the input temperature. The internal
        // dynamics then stay in the familiar sigma/epsilon ~ 1 regime no
        // matter the physical scale, which keeps `delta_time` valid, while
        // mixtures correctly give each species its own reduced temperature
        // T* = T_kelvin / (epsilon_i / k_B).
        let epsilon_ref_kelvin = state
            .particles
            .iter()
            .map(|(_, init)| init.epsilon)
            .fold(f32::MIN, f32::max);
        if epsilon_ref_kelvin <= 0.0 {
            return Err("at least one species needs a positive epsilon (in Kelvin)".to_string());
        }
        for (_, init) in state.particles.iter_mut() {
            init.epsilon /= epsilon_ref_kelvin;
        }
        let temperature_reduced = state.temperature / epsilon_ref_kelvin;

        let mut particle_num = 0;

        for i in 0..state.particles.len() {
            particle_num += state.particles[i].0;
        }

        let particles_width = ((particle_num as f32).powf(1.0 / 3.0).ceil()) as usize;

        let spacing = env_size / (particles_width as f32);

        let mut biggest_sigma = state.particles[0].1.sigma;
        for particle in &state.particles {
            if particle.1.sigma > biggest_sigma {
                biggest_sigma = particle.1.sigma;
            }
        }

        if biggest_sigma > spacing {
            return Err(
                "env too small/particles too big, can't create particle without overlap"
                    .to_string(),
            );
        }

        let mut particles: Vec<particle::Particle> = Vec::new();

        let mut rng = rand::rng();

        for i in 0..particles_width {
            for j in 0..particles_width {
                for k in 0..particles_width {
                    let y = rng.random_range(0..state.particles.len());
                    let mut pos = [0.0; 3];
                    pos[0] = spacing * (i as f32) + spacing / 2.0;
                    pos[1] = spacing * (j as f32) + spacing / 2.0;
                    pos[2] = spacing * (k as f32) + spacing / 2.0;
                    let std_dev = (temperature_reduced / state.particles[y].1.mass).sqrt();
                    let normal = Normal::new(0.0, std_dev).unwrap();
                    let velocity = [
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                        normal.sample(&mut rng),
                    ];
                    particles.push(crate::Particle::new(&state.particles[y].1, pos, velocity));
                    state.particles[y].0 -= 1;
                    if state.particles[y].0 == 0 {
                        state.particles.remove(y);
                    }
                    if state.particles.is_empty() {
                        break;
                    }
                }
                if state.particles.is_empty() {
                    break;
                }
            }
            if state.particles.is_empty() {
                break;
            }
        }

        // Remove center-of-mass drift (mass-weighted)
        let mut v_com = [0.0f32; 3];
        let mut m_total = 0.0f32;
        for p in &particles {
            for i in 0..3 {
                v_com[i] += p.mass * p.prev.velocity[i];
            }
            m_total += p.mass;
        }
        if m_total > 0.0 {
            for i in 0..3 {
                v_com[i] /= m_total;
            }
        }

        // Rescale to the exact target temperature (after drift removal)
        let ke: f32 = particles
            .iter()
            .map(|p| {
                let v = [
                    p.prev.velocity[0] - v_com[0],
                    p.prev.velocity[1] - v_com[1],
                    p.prev.velocity[2] - v_com[2],
                ];
                p.mass * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            })
            .sum();
        let t_actual = ke / (3.0 * particles.len() as f32);

        if t_actual > 0.0 {
            let scale = (temperature_reduced / t_actual).sqrt();
            for p in &mut particles {
                for i in 0..3 {
                    p.prev.velocity[i] = (p.prev.velocity[i] - v_com[i]) * scale;
                }
            }
        }

        let r_cut = biggest_sigma * 2.7;
        let skin = biggest_sigma * 0.3;

        let width = (env_size / (r_cut + skin)) as usize;

        if width < 3 {
            return Err("Env to small/particles too big, possible self referencing".to_string());
        }

        let cell_num = width * width * width;

        let mut cells = Vec::with_capacity(cell_num);

        for _ in 0..cell_num {
            cells.push(Vec::new())
        }

        let mut simulation = Self {
            particles,
            env_size,
            r_cut,
            skin,
            cell_list: cells.into_boxed_slice(),
            list_size: width,
            delta_time,
            steps_between_paint,
            epsilon_ref_kelvin,
        };

        simulation.create_verlet_list();

        // Velocity Verlet needs the accelerations at t = 0 (the first
        // position update already uses them). The freshly placed state lives
        // in `prev` while the force code reads `curr`, so swap, compute the
        // forces, and swap back.
        for particle in &mut simulation.particles {
            std::mem::swap(&mut particle.curr, &mut particle.prev);
        }
        simulation.compute_forces();
        for particle in &mut simulation.particles {
            std::mem::swap(&mut particle.curr, &mut particle.prev);
        }

        Ok(simulation)
    }

    fn check_verlet(&self) -> bool {
        let mut out = false;
        for p in &self.particles {
            if p.moved_too_much(self.skin, self.env_size) {
                out = true;
                break;
            }
        }

        out
    }

    fn get_cell_id(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.list_size * y + self.list_size * self.list_size * z
    }

    fn position_calculation(&mut self) {
        let delta_time = self.delta_time;
        for particle in &mut self.particles {
            for i in 0..3 {
                particle.curr.position[i] = particle.prev.position[i]
                    + particle.prev.velocity[i] * delta_time
                    + particle.prev.acceleration[i] * delta_time * delta_time / 2.0;
                // Fold into [0, env_size). This MUST be a pure +/- box-length
                // shift: the forces, cell lists and verlet lists all work in
                // the minimum-image convention, where only multiples of the
                // box length are no-ops. A teleport to a fixed point (like the
                // old `> env + sigma` -> `-sigma` wrap) displaces particles by
                // ~2 sigma in the physics frame and can land them on top of
                // neighbours, detonating the 1/r^13 Lennard-Jones repulsion.
                particle.curr.position[i] = particle.curr.position[i].rem_euclid(self.env_size);
            }
        }
    }

    /// Sums the pairwise forces from the verlet lists into `curr.force` and
    /// derives `curr.acceleration`. No integration happens here, so this can
    /// also be used to seed the initial accelerations.
    fn compute_forces(&mut self) {
        for j in 0..self.particles.len() {
            let mut force = [0.0; 3];
            for i in 0..self.particles[j].verlet.particles.len() {
                let particle = &self.particles[j];
                let p = particle.verlet.particles[i];
                let clone = self.particles[p].clone();
                let f = particle.force(
                    &clone,
                    particle.distance_squared(&clone, self.env_size).sqrt(),
                    self.env_size,
                );
                for l in 0..3 {
                    force[l] += f[l];
                }
            }
            self.particles[j].add_force(force);
            self.particles[j].acceleration();
        }
    }

    fn force_acc_velocity_calculation(&mut self) {
        self.compute_forces();
        for particle in &mut self.particles {
            particle.velocity(self.delta_time);
        }
    }

    fn create_cell_list(&mut self) {
        for i in 0..self.cell_list.len() {
            self.cell_list[i].clear();
        }

        for i in 0..self.particles.len() {
            self.particles[i].verlet.particles.clear();
            let p = self.particles[i].curr.position;
            self.particles[i].verlet.org_pos = p;
            let c = |v: f32| -> usize {
                let v = v.rem_euclid(self.env_size);
                ((v / self.r_list()) as usize).min(self.list_size - 1)
            };
            self.cell_list[self.get_cell_id(c(p[0]), c(p[1]), c(p[2]))].push(i);
        }
    }

    fn create_verlet_list(&mut self) {
        self.create_cell_list();

        for x in 0..self.list_size {
            for y in 0..self.list_size {
                for z in 0..self.list_size {
                    self.create_list_one_cell(x, y, z);
                }
            }
        }
    }

    fn create_list_one_cell(&mut self, x: usize, y: usize, z: usize) {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let jx = (x as isize + dx).rem_euclid(self.list_size as isize) as usize;
                    let jy = (y as isize + dy).rem_euclid(self.list_size as isize) as usize;
                    let jz = (z as isize + dz).rem_euclid(self.list_size as isize) as usize;
                    for &a in &self.cell_list[self.get_cell_id(x, y, z)] {
                        for &b in &self.cell_list[self.get_cell_id(jx, jy, jz)] {
                            if a == b {
                                continue;
                            }
                            if self.particles[a]
                                .distance_squared(&self.particles[b].clone(), self.env_size)
                                < (self.r_list() * self.r_list())
                            {
                                self.particles[a].verlet.particles.push(b);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::md_simulation::MdSimulation;
    use crate::simulation_state::{ParticleInit, SimulationState};

    use super::CpuSimulation;

    /// Long NVE run with two species: every boundary crossing must be a pure
    /// multiple of the box length (the minimum-image convention used by the
    /// forces/cell/verlet lists treats any other wrap as a real ~2-sigma
    /// displacement, which can land particles on top of neighbours and
    /// detonate the Lennard-Jones 1/r^13 repulsion — velocities blow up).
    /// Also checks that the paint data carries each species' true radius
    /// (sigma/2) and its setup colour, and that the Kelvin temperature
    /// plumbing (epsilon normalization + readout) is correct.
    #[test]
    fn long_run_stays_bounded() {
        let env_size = 13.0;
        let blue = [0.26, 0.52, 0.96, 1.0];
        let orange = [0.98, 0.45, 0.13, 1.0];
        let state = SimulationState {
            particles: vec![
                (
                    70,
                    ParticleInit {
                        sigma: 1.0,
                        epsilon: 119.8, // argon-like (Kelvin)
                        mass: 1.0,
                        color: blue,
                    },
                ),
                (
                    30,
                    ParticleInit {
                        sigma: 1.4,
                        epsilon: 229.0, // xenon-like (Kelvin)
                        mass: 2.0,
                        color: orange,
                    },
                ),
            ],
            temperature: 120.0,
            env_size,
        };
        // 10_000 physics steps (8 per simulation_step call), ~20 time units.
        let mut sim = CpuSimulation::new(0.002, 8, state).expect("simulation setup failed");

        // The constructor rescales the velocities to exactly the target
        // temperature, so the initial readout must match it.
        let measured = sim.temperature_kelvin();
        assert!(
            (measured - 120.0).abs() < 1.0,
            "measured {measured} K right after setup, expected ~120 K"
        );

        for step in 0..(10_000 / 8) {
            let paints = sim.simulation_step();
            assert_eq!(paints.len(), 100, "particle count changed");

            if step == 0 {
                // Colours and radii must match the species setup.
                let num_blue = paints.iter().filter(|p| p.color == blue).count();
                let num_orange = paints.iter().filter(|p| p.color == orange).count();
                assert_eq!(num_blue, 70, "blue count wrong");
                assert_eq!(num_orange, 30, "orange count wrong");
                for paint in &paints {
                    let expected = if paint.color == blue { 0.5 } else { 0.7 };
                    assert!(
                        (paint.radius - expected).abs() < 1e-6,
                        "radius {} doesn't match sigma/2 = {}",
                        paint.radius,
                        expected
                    );
                }
            }

            for paint in &paints {
                for c in paint.position {
                    assert!(
                        (0.0..env_size).contains(&c),
                        "position {c} escaped the box [0, {env_size}) at call {step}"
                    );
                }
            }
        }

        // The run is energy-conserving, so the measured temperature drifts
        // as the lattice relaxes — but it must stay within sane bounds.
        // (A missing epsilon normalization would read ~0.5 K here, a
        // double-applied one ~27,600 K.)
        let measured = sim.temperature_kelvin();
        assert!(
            (60.0..600.0).contains(&measured),
            "measured temperature {measured} K left the plausible range"
        );

        // After the last step the current state lives in `prev` (swapped at
        // the end of every physics step).
        let mut max_speed_sq = 0.0f32;
        for particle in &sim.particles {
            let v = particle.prev.velocity;
            let speed_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            assert!(speed_sq.is_finite(), "velocity went NaN/inf");
            max_speed_sq = max_speed_sq.max(speed_sq);
        }
        // Equilibrated speeds are ~2-6; 30 is generous headroom that still
        // catches the >100 explosion of a broken wrap.
        assert!(
            max_speed_sq < 30.0 * 30.0,
            "velocities blew up: |v| = {}",
            max_speed_sq.sqrt()
        );
    }
}
