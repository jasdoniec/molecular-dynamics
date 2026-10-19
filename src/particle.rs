use crate::potential::ParticleParams;

pub struct VerletList {
    pub particles: Vec<usize>,
    pub org_pos: [f32; 3],
}

pub struct Particle {
    pub curr: ParticleState,
    pub prev: ParticleState,
    pub verlet: VerletList,
    /// Per-species parameters the force kernel works with (ε, σ). Mass
    /// deliberately stays a separate field: the kernel never needs it,
    /// the integrator always does.
    pub params: ParticleParams,
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

/// Snapshot of a neighbour's position and kernel params — the packet
/// the pair geometry and the force kernels receive.
pub struct ParticleClone {
    pub position: [f32; 3],
    pub params: ParticleParams,
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
            params: ParticleParams {
                epsilon: particle.epsilon,
                sigma: particle.sigma,
            },
            mass: particle.mass,
            color: particle.color,
        }
    }

    pub fn clone(&self) -> ParticleClone {
        ParticleClone {
            position: self.curr.position,
            params: self.params,
        }
    }

    /// Minimum-image displacement from `other` to `self`
    /// (`self.curr.position − other.position`, wrapped to ±box/2 by the
    /// periodic boundary convention). Both the Verlet-list distance
    /// checks and the force kernels work off this one definition, so the
    /// wrap logic lives in exactly one place.
    pub fn displacement(&self, other: &ParticleClone, box_length: f32) -> [f32; 3] {
        let mut disp = [0.0; 3];
        for i in 0..disp.len() {
            let mut diff = self.curr.position[i] - other.position[i];
            //wraps the displacement across the repeating box
            diff -= box_length * (diff / box_length).round();
            disp[i] = diff;
        }
        disp
    }

    pub fn distance_squared(&self, other: &ParticleClone, box_length: f32) -> f32 {
        let disp = self.displacement(other, box_length);
        disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]
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
