use crate::particle::Particle;

pub struct CpuSimulation {
    particles: Vec<crate::Particle>,
    env_size: f32,
    r_list: f32,
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
    fn new(env_size: f32, delta_time: f32, steps_between_paint: usize) -> Self {
        //TODO Make particles
        //TODO Get env_size and r_list from constructor
        //TODO initialize cell_list

        let r_list = 0.0;

        let width = (env_size / r_list) as usize;

        let cell_num = width * width * width;

        let mut cells = Vec::with_capacity(cell_num);

        for _ in 0..cell_num {
            cells.push(Vec::new())
        }

        Self {
            particles: Vec::new(),
            env_size,
            r_list: 0.0,
            cell_list: cells.into_boxed_slice(),
            list_size: width,
            delta_time,
            steps_between_paint,
        }
    }

    fn check_verlet(&self) -> bool {
        let mut out = false;
        for p in &self.particles {
            if p.moved_too_much(self.r_list, self.env_size) {
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
                ((v / self.r_list) as usize).min(self.list_size - 1)
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
                                < (self.r_list * self.r_list)
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
