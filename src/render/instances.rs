//! CPU-side instance generation, including periodic "ghost" copies.
//!
//! Each frame the paint data returned by
//! [`MdSimulation::simulation_step`](crate::MdSimulation::simulation_step) is
//! turned into render [`Instance`]s: each particle carries its own radius and
//! its setup colour. For every particle that sits close enough to a box face
//! that its sphere would visually cross the boundary, extra copies are
//! emitted at `position +/- box_length` — one for every boundary the
//! particle touches (faces, edges and corners produce up to 8 copies). The
//! ghosts use the same radius and colour as the original particle, so
//! wrap-around is easy to follow.

use crate::md_simulation::ParticlePaint;

/// One renderable particle billboard.
///
/// Layout must match the instance-rate vertex buffer declared in
/// `renderer.rs`: two `Float32x4` attributes (`pos_radius`, `color`) with a
/// 32-byte stride. `pos_radius` is xyz = world position, w = radius.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub pos_radius: [f32; 4],
    pub color: [f32; 4],
}

impl Instance {
    pub const SIZE: u64 = std::mem::size_of::<Instance>() as u64;
}

/// Turns paint data into instances, duplicating particles that are close to
/// a periodic boundary so they show up on both sides of the box.
///
/// Each particle renders at its own radius (the sim provides the true,
/// sigma-derived size).
///
/// Positions are kept inside `[0, box_length)` by the simulation; the
/// `rem_euclid` fold is a guard for float edge cases and for
/// `MdSimulation` implementations that don't fold themselves.
///
/// If `box_length <= 0.0` no periodicity is assumed and each particle is
/// drawn exactly once.
///
/// Returns the number of instances written (all of them pushed to `out`).
pub fn build_instances(
    particles: &[ParticlePaint],
    box_length: f32,
    out: &mut Vec<Instance>,
) -> usize {
    out.clear();
    out.reserve(particles.len());

    if box_length <= 0.0 {
        for particle in particles {
            push_instance(out, particle.position, particle.radius, particle.color);
        }
        return out.len();
    }

    for particle in particles {
        let radius = particle.radius;
        // Fold into [0, box_length).
        let q = [
            particle.position[0].rem_euclid(box_length),
            particle.position[1].rem_euclid(box_length),
            particle.position[2].rem_euclid(box_length),
        ];

        for dx in -1i32..=1 {
            if !axis_visible(q[0], dx, radius, box_length) {
                continue;
            }
            for dy in -1i32..=1 {
                if !axis_visible(q[1], dy, radius, box_length) {
                    continue;
                }
                for dz in -1i32..=1 {
                    if !axis_visible(q[2], dz, radius, box_length) {
                        continue;
                    }
                    push_instance(
                        out,
                        [
                            q[0] + dx as f32 * box_length,
                            q[1] + dy as f32 * box_length,
                            q[2] + dz as f32 * box_length,
                        ],
                        radius,
                        particle.color,
                    );
                }
            }
        }
    }

    out.len()
}

fn push_instance(out: &mut Vec<Instance>, position: [f32; 3], radius: f32, color: [f32; 4]) {
    out.push(Instance {
        pos_radius: [position[0], position[1], position[2], radius],
        color,
    });
}

/// Is the copy of a particle shifted by `offset * box_length` along one axis
/// visible (i.e. does the original sphere cross that face)?
///
/// A particle at `v` near the `+face` (so the sphere crosses it, centred at
/// `v` with radius `r`) has a periodic image at `v - box_length`, which is
/// where it "re-enters" on the `-face` side.
fn axis_visible(v: f32, offset: i32, radius: f32, box_length: f32) -> bool {
    match offset {
        -1 => v + radius >= box_length,
        1 => v <= radius,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: f32 = 10.0;
    const COLOR: [f32; 4] = [0.26, 0.52, 0.96, 1.0];

    fn paint(position: [f32; 3]) -> ParticlePaint {
        ParticlePaint {
            position,
            radius: 0.5,
            color: COLOR,
        }
    }

    #[test]
    fn interior_particle_has_no_ghosts() {
        let mut out = Vec::new();
        let n = build_instances(&[paint([5.0, 5.0, 5.0])], L, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].pos_radius, [5.0, 5.0, 5.0, 0.5]);
        assert_eq!(out[0].color, COLOR);
    }

    #[test]
    fn particle_on_face_appears_on_both_sides() {
        let mut out = Vec::new();
        // Touching the -x face: original + one ghost at +L on x.
        let n = build_instances(&[paint([0.4, 5.0, 5.0])], L, &mut out);
        assert_eq!(n, 2);
        assert!(out.iter().any(|i| i.pos_radius[0] == 0.4));
        assert!(out.iter().any(|i| i.pos_radius[0] == 10.4));
        // Ghosts keep the original's colour and radius.
        assert!(out.iter().all(|i| i.color == COLOR));
        assert!(out.iter().all(|i| i.pos_radius[3] == 0.5));
    }

    #[test]
    fn particle_on_edge_gets_corner_ghosts() {
        let mut out = Vec::new();
        // Touching -x and -y faces: (0,0) stays, plus (L,0), (0,L), (L,L) on x/y.
        let n = build_instances(&[paint([0.4, 0.4, 5.0])], L, &mut out);
        assert_eq!(n, 4);
    }

    #[test]
    fn corner_particle_gets_all_eight_copies() {
        let mut out = Vec::new();
        let n = build_instances(&[paint([0.4, 0.4, 0.4])], L, &mut out);
        assert_eq!(n, 8);
    }

    #[test]
    fn overshooting_positions_are_folded_into_the_box() {
        let mut out = Vec::new();
        // Slightly below the box (e.g. from a sim that doesn't fold itself).
        let n = build_instances(&[paint([-0.25, 5.0, 5.0])], L, &mut out);
        assert_eq!(n, 2);
        assert!(out.iter().any(|i| i.pos_radius[0] == 9.75));
        assert!(out.iter().any(|i| i.pos_radius[0] == -0.25));
    }

    #[test]
    fn no_box_means_no_ghosts() {
        let mut out = Vec::new();
        let n = build_instances(&[paint([0.4, 0.4, 0.4])], 0.0, &mut out);
        assert_eq!(n, 1);
    }

    #[test]
    fn instance_layout_matches_gpu_vertex_stride() {
        // renderer.rs declares the instance vertex buffer as two Float32x4
        // attributes with a 32-byte stride — keep this in sync.
        assert_eq!(std::mem::size_of::<Instance>(), 32);
        assert_eq!(std::mem::offset_of!(Instance, color), 16);
    }
}
