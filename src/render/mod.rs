//! Real-time 3D visualization for simulations.
//!
//! Renders the per-particle paint data returned by
//! [`MdSimulation::simulation_step`](crate::MdSimulation::simulation_step) —
//! positions, per-particle radii and setup colours — as shaded spheres
//! inside the (periodic) simulation box. Particles close to a box face are
//! additionally drawn as periodic "ghost" copies on the other side of the
//! box, so the wrap-around behaviour is visible.

pub mod camera;
pub mod hud;
pub mod instances;
pub mod renderer;
pub mod visualizer;

pub use camera::{CameraUniform, OrbitCamera};
pub use hud::{CharInstance, Hud, HudLine};
pub use instances::{build_instances, Instance};
pub use renderer::Renderer;
pub use visualizer::Visualizer;
