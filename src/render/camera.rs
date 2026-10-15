//! Orbit camera: rotate around a target point, zoom, and pan.
//!
//! The camera only owns a small amount of state (target, distance, yaw, pitch)
//! and produces the 4x4 view-projection matrix (plus the camera's world-space
//! right/up vectors, used for billboard rendering) that gets uploaded to the
//! GPU every frame.

use glam::{camera::rh::proj::directx, camera::rh::view, Vec3};

/// Data uploaded to the GPU each frame.
///
/// Layout must match `struct Camera` in `shaders/particle.wgsl` and
/// `shaders/box.wgsl`: mat4x4f + vec4f + vec4f = 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [f32; 16],
    pub right: [f32; 4],
    pub up: [f32; 4],
}

/// Orbit ("turntable") camera.
///
/// - left-drag: rotate around [`OrbitCamera::target`]
/// - scroll:    zoom (changes [`OrbitCamera::distance`])
/// - right/middle-drag: pan (moves [`OrbitCamera::target`])
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Horizontal angle around the target (radians).
    pub yaw: f32,
    /// Elevation above the target (radians), clamped to +/- ~89 degrees.
    pub pitch: f32,
    /// Vertical field of view (radians).
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance: distance.max(0.1),
            yaw: 0.9,
            pitch: 0.45,
            fov_y: 45f32.to_radians(),
            near: 0.01,
            far: 100_000.0,
            min_distance: distance * 0.02,
            max_distance: distance * 10.0,
        }
    }

    /// World-space position of the camera ("eye").
    pub fn eye(&self) -> Vec3 {
        let horizontal = self.distance * self.pitch.cos();
        self.target
            + Vec3::new(
                horizontal * self.yaw.cos(),
                self.distance * self.pitch.sin(),
                horizontal * self.yaw.sin(),
            )
    }

    /// Rotate the camera around the target. `dx`/`dy` are mouse deltas in
    /// pixels (x to the right, y downwards).
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        const SPEED: f32 = 0.01;
        self.yaw += dx * SPEED;
        // Dragging the mouse up should orbit the camera upwards.
        self.pitch = (self.pitch - dy * SPEED).clamp(-1.55, 1.55);
    }

    /// Zoom. `steps` is positive when scrolling up (zoom in).
    pub fn zoom(&mut self, steps: f32) {
        let factor = 1.1f32.powf(-steps);
        self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
    }

    /// Pan the camera (move the orbit target). `dx`/`dy` are mouse deltas in
    /// pixels; `viewport_height` is the render area height in pixels.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport_height: f32) {
        let (right, up) = self.basis();
        let world_per_px =
            2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_height.max(1.0);
        // Dragging moves the scene with the cursor: target goes the other way.
        self.target += -right * (dx * world_per_px) + up * (dy * world_per_px);
    }

    /// World-space right/up basis vectors of the camera.
    fn basis(&self) -> (Vec3, Vec3) {
        let eye = self.eye();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        (right, up)
    }

    pub fn to_uniform(&self, aspect: f32) -> CameraUniform {
        let eye = self.eye();
        // WebGPU NDC has z in [0, 1] (DirectX convention), hence the
        // `proj::directx` projection; the view matrix is right-handed Y-up.
        let view = view::look_at_mat4(eye, self.target, Vec3::Y);
        let proj = directx::perspective(self.fov_y, aspect.max(0.0001), self.near, self.far);
        let (right, up) = self.basis();
        CameraUniform {
            view_proj: (proj * view).to_cols_array(),
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
        }
    }
}
