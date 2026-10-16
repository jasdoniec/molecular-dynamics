pub struct VerletList {
    pub particles: Vec<usize>,
    pub org_pos: [f32; 3],
}

pub struct Particle {
    pub curr: ParticleState,
    pub prev: ParticleState,
    pub verlet: VerletList,
    pub epsilon: f32,
    pub sigma: f32,
    pub mass: f32,
    /// RGBA colour from [`crate::simulation_state::ParticleInit`],
    /// used when painting this particle.
    pub color: [f32; 4],
}

pub struct ParticleState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub acceleration: [f32; 3],
    pub force: [f32; 3],
}

pub struct ParticleClone {
    pub position: [f32; 3],
    pub epsilon: f32,
    pub sigma: f32,
    pub mass: f32,
}

impl ParticleState {
    pub fn new() -> Self {
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            acceleration: [0.0; 3],
            force: [0.0; 3],
        }
    }
}

impl Particle {
    pub fn new(
        particle: &crate::simulation_state::ParticleInit,
        position: [f32; 3],
        velocity: [f32; 3],
    ) -> Self {
        let mut prev = ParticleState::new();
        let curr = ParticleState::new();
        prev.position = position.clone();
        prev.velocity = velocity;
        Self {
            curr,
            prev,
            verlet: VerletList {
                particles: Vec::new(),
                org_pos: position.clone(),
            },
            epsilon: particle.epsilon,
            sigma: particle.sigma,
            mass: particle.mass,
            color: particle.color,
        }
    }

    pub fn clone(&self) -> ParticleClone {
        ParticleClone {
            position: self.curr.position.clone(),
            epsilon: self.epsilon,
            sigma: self.sigma,
            mass: self.mass,
        }
    }

    pub fn distance_squared(&self, other: &ParticleClone, box_length: f32) -> f32 {
        let mut out = 0.0 as f32;
        for i in 0..other.position.len() {
            let mut diff = other.position[i] - self.curr.position[i];
            //wraps the displacement across the repeating box
            diff -= box_length * (diff / box_length).round();
            out = out + diff * diff;
        }
        out
    }

    pub fn moved_too_much(&self, distance: f32, box_length: f32) -> bool {
        let mut displacement_squared = 0_f32;
        for i in 0..self.curr.position.len() {
            let mut diff = self.verlet.org_pos[i] - self.curr.position[i];
            diff -= box_length * (diff / box_length).round();
            displacement_squared += diff * diff;
        }
        distance * distance / 4.0 <= displacement_squared
    }

    pub fn force(&self, other: &ParticleClone, distance: f32, box_length: f32) -> [f32; 3] {
        // Guard against exact overlap (distance ≈ 0) which would cause NaN
        if distance < 1e-6 {
            return [0.0; 3];
        }

        let sigma = (self.sigma + other.sigma) / 2.0;
        let epsilon = (self.epsilon * other.epsilon).sqrt();

        let ratio = sigma / distance;

        let force = 24.0 * epsilon / distance * (2.0 * ratio.powi(12) - ratio.powi(6));

        let u = |diff: f32| -> f32 {
            let mut diff = diff;
            diff -= box_length * (diff / box_length).round();
            diff / distance
        };

        let mut f = [0.0; 3];

        for i in 0..self.curr.force.len() {
            f[i] = force * u(self.curr.position[i] - other.position[i]);
        }

        f
    }

    pub fn add_force(&mut self, force: [f32; 3]) {
        self.curr.force = force;
    }

    pub fn acceleration(&mut self) {
        for i in 0..3 {
            self.curr.acceleration[i] = self.curr.force[i] / self.mass;
        }
    }

    pub fn velocity(&mut self, delta_time: f32) {
        for i in 0..3 {
            self.curr.velocity[i] = self.prev.velocity[i]
                + delta_time / 2.0 * (self.curr.acceleration[i] + self.prev.acceleration[i]);
        }
    }
}
