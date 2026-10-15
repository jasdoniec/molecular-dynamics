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
}

impl crate::MdSimulation for CpuSimulation {
    fn simulation_step(&mut self) -> Vec<[f32; 3]> {
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

        self.particles.iter().map(|p| p.curr.position).collect()
    }
}

impl CpuSimulation {
    fn r_list(&self) -> f32 {
        self.r_cut + self.skin
    }

    fn new(
        delta_time: f32,
        steps_between_paint: usize,
        state: crate::SimulationState,
    ) -> Result<Self, String> {
        let mut state = state;

        let env_size = state.env_size;

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
                    let std_dev = (state.temperature / state.particles[y].1.mass).sqrt();
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
            let scale = (state.temperature / t_actual).sqrt();
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
        };

        simulation.create_verlet_list();

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
                if particle.curr.position[i] > self.env_size + particle.sigma {
                    particle.curr.position[i] = -particle.sigma;
                } else if particle.curr.position[i] < -particle.sigma {
                    particle.curr.position[i] = self.env_size + particle.sigma;
                }
            }
        }
    }

    fn force_acc_velocity_calculation(&mut self) {
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
            self.particles[j].velocity(self.delta_time);
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
