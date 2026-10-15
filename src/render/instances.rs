//! CPU-side instance generation, including periodic "ghost" copies.
//!
//! Each frame the current particle positions are turned into render
//! [`Instance`]s. For every particle that sits close enough to a box face that
//! its sphere would visually cross the boundary, extra copies are emitted at
//! `position +/- box_length` — one for every boundary the particle touches
//! (faces, edges and corners produce up to 8 copies). The ghosts use the same
//! colour as the original particle, so wrap-around is easy to follow.

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

/// Turns raw simulation positions into instances, duplicating particles that
/// are close to a periodic boundary so they show up on both sides of the box.
///
/// `positions` may lie slightly outside `[0, box_length)` (the CPU simulation
/// allows an overshoot of up to one sigma before wrapping); values are folded
/// back into the box with `rem_euclid` before ghosting.
///
/// If `box_length <= 0.0` no periodicity is assumed and each particle is
/// drawn exactly once.
///
/// Returns the number of instances written (all of them pushed to `out`).
pub fn build_instances(
    positions: &[[f32; 3]],
    radius: f32,
    box_length: f32,
    out: &mut Vec<Instance>,
) -> usize {
    out.clear();
    out.reserve(positions.len());

    if box_length <= 0.0 {
        for (i, p) in positions.iter().enumerate() {
            out.push(Instance {
                pos_radius: [p[0], p[1], p[2], radius],
                color: color_for_index(i),
            });
        }
        return out.len();
    }

    for (i, p) in positions.iter().enumerate() {
        let color = color_for_index(i);
        // Fold into [0, box_length): the simulation lets positions overshoot
        // the box by up to one sigma before it wraps them.
        let q = [
            p[0].rem_euclid(box_length),
            p[1].rem_euclid(box_length),
            p[2].rem_euclid(box_length),
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
                    out.push(Instance {
                        pos_radius: [
                            q[0] + dx as f32 * box_length,
                            q[1] + dy as f32 * box_length,
                            q[2] + dz as f32 * box_length,
                            radius,
                        ],
                        color,
                    });
                }
            }
        }
    }

    out.len()
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

/// Stable colour per particle index (golden-angle hue walk), so the same
/// particle keeps its colour across frames and its periodic ghosts are
/// recognisable as copies of the same particle.
pub fn color_for_index(i: usize) -> [f32; 4] {
    let hue = (i as f32 * 0.618_033_99).fract();
    hsv_to_rgb(hue, 0.62, 0.95)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 4] {
    let sector = (h * 6.0).floor();
    let f = h * 6.0 - sector;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let (r, g, b) = match (sector as i32) % 6 {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    const L: f32 = 10.0;

    #[test]
    fn interior_particle_has_no_ghosts() {
        let mut out = Vec::new();
        let n = build_instances(&[[5.0, 5.0, 5.0]], 0.5, L, &mut out);
        assert_eq!(n, 1);
        assert_eq!(out[0].pos_radius, [5.0, 5.0, 5.0, 0.5]);
    }

    #[test]
    fn particle_on_face_appears_on_both_sides() {
        let mut out = Vec::new();
        // Touching the -x face: original + one ghost at +L on x.
        let n = build_instances(&[[0.4, 5.0, 5.0]], 0.5, L, &mut out);
        assert_eq!(n, 2);
        assert!(out.iter().any(|i| i.pos_radius[0] == 0.4));
        assert!(out.iter().any(|i| i.pos_radius[0] == 10.4));
    }

    #[test]
    fn particle_on_edge_gets_corner_ghosts() {
        let mut out = Vec::new();
        // Touching -x and -y faces: (0,0) stays, plus (L,0), (0,L), (L,L) on x/y.
        let n = build_instances(&[[0.4, 0.4, 5.0]], 0.5, L, &mut out);
        assert_eq!(n, 4);
    }

    #[test]
    fn corner_particle_gets_all_eight_copies() {
        let mut out = Vec::new();
        let n = build_instances(&[[0.4, 0.4, 0.4]], 0.5, L, &mut out);
        assert_eq!(n, 8);
    }

    #[test]
    fn overshooting_positions_are_folded_into_the_box() {
        let mut out = Vec::new();
        // Slightly below the box (allowed by the sim's wrap threshold).
        let n = build_instances(&[[-0.25, 5.0, 5.0]], 0.5, L, &mut out);
        assert_eq!(n, 2);
        assert!(out.iter().any(|i| i.pos_radius[0] == 9.75));
        assert!(out.iter().any(|i| i.pos_radius[0] == -0.25));
    }

    #[test]
    fn no_box_means_no_ghosts() {
        let mut out = Vec::new();
        let n = build_instances(&[[0.4, 0.4, 0.4]], 0.5, 0.0, &mut out);
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
