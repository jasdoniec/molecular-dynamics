use crate::particle;

pub struct CpuSimulation {
    particles: Vec<crate::Particle>,
    env_size: f32,
    r_list: f32,
    cell_list: Box<[Vec<usize>]>,
    list_size: usize,
}

impl crate::MdSimulation for CpuSimulation {
    fn simulation_step(&mut self) -> Vec<[f32; 3]> {
        vec![[0 as f32, 0 as f32, 0 as f32]]
    }
}

impl CpuSimulation {
    fn new(env_size: f32) -> Self {
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
        }
    }

    fn get_cell_id(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.list_size * y + self.list_size * self.list_size * z
    }

    fn position_calculation(&mut self, delta_time: f32) {
        for particle in &mut self.particles {
            for i in 0..3 {
                particle.curr.position[i] = particle.prev.position[i]
                    + particle.prev.velocity[i] * delta_time
                    + particle.prev.acceleration[i] * delta_time * delta_time / 2.0;
            }
        }
    }

    fn force_acc_velocity_calculation(&mut self) {
        let length = self.particles.len() / 2;

        for i in 0..length {}
    }

    fn create_cell_list(&mut self) {
        for x in 0..self.list_size {
            for y in 0..self.list_size {
                for z in 0..self.list_size {
                    self.cell_list[x + y * self.list_size + z * self.list_size * self.list_size]
                        .clear();
                }
            }
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

    fn create_list(&mut self) {
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
                            if self.particles[a].distance_squared(&self.particles[b], self.env_size)
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
