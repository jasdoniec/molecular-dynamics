//! Real-time 3D visualization for simulations.
//!
//! Renders the positions returned by [`crate::MdSimulation::simulation_step`]
//! as shaded spheres inside the (periodic) simulation box. Particles close to
//! a box face are additionally drawn as periodic "ghost" copies on the other
//! side of the box, so the wrap-around behaviour is visible.

pub mod camera;
pub mod instances;
pub mod renderer;
pub mod visualizer;

pub use camera::{CameraUniform, OrbitCamera};
pub use instances::{build_instances, color_for_index, Instance};
pub use renderer::Renderer;
pub use visualizer::Visualizer;
