pub struct VerletList {
    pub particles: Vec<usize>,
    pub org_pos: [f32; 3],
}

pub struct Particle {
    pub curr: ParticleState,
    pub prev: ParticleState,
    pub verlet: VerletList,
}

pub struct ParticleState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
    pub force: [f32; 3],
}

impl Particle {
    pub fn distance_squared(&self, other: &Particle, box_length: f32) -> f32 {
        let mut out = 0.0 as f32;
        for i in 0..other.curr.position.len() {
            let mut diff = other.curr.position[i] - self.curr.position[i];
            //wraps the displacement across the repeating box
            diff -= box_length * (diff / box_length).round();
            out = out + diff * diff;
        }
        out
    }

    pub fn moved_too_much(&self, distance: f32) -> bool {
        let mut displacement_squared = 0_f32;
        for i in 0..self.curr.position.len() {
            let diff = self.verlet.org_pos[i] - self.curr.position[i];
            displacement_squared += diff * diff;
        }
        distance * distance / 4.0 >= displacement_squared
    }
}
