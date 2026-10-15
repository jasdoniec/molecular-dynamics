//! The winit application that ties everything together: steps the
//! [`MdSimulation`] once per frame, rebuilds the render instances (including
//! periodic ghosts) and drives the renderer and camera input.

use std::sync::Arc;

use glam::Vec3;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::MdSimulation;

use super::camera::OrbitCamera;
use super::instances::{build_instances, Instance};
use super::renderer::Renderer;

pub struct Visualizer<E: MdSimulation> {
    engine: E,
    box_length: f32,
    particle_radius: f32,
    steps_per_frame: usize,
    paused: bool,

    camera: OrbitCamera,
    /// Positions returned by the last `simulation_step()`.
    positions: Vec<[f32; 3]>,
    /// Scratch buffer for the per-frame render instances (with ghosts).
    instances: Vec<Instance>,

    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    drag_button: Option<MouseButton>,
    last_cursor: Option<(f64, f64)>,
}

impl<E: MdSimulation + 'static> Visualizer<E> {
    pub fn new(engine: E, box_length: f32) -> Self {
        let camera = OrbitCamera::new(Vec3::splat(box_length * 0.5), box_length * 2.2);
        Self {
            engine,
            box_length,
            particle_radius: (box_length * 0.025).max(0.01),
            steps_per_frame: 1,
            paused: false,
            camera,
            positions: Vec::new(),
            instances: Vec::new(),
            window: None,
            renderer: None,
            drag_button: None,
            last_cursor: None,
        }
    }

    /// Initial billboard radius for the particles (world units).
    pub fn with_particle_radius(mut self, radius: f32) -> Self {
        self.particle_radius = radius;
        self
    }

    /// How many `simulation_step()` calls to run per rendered frame.
    pub fn with_steps_per_frame(mut self, steps: usize) -> Self {
        self.steps_per_frame = steps.max(1);
        self
    }

    pub fn paused(mut self, paused: bool) -> Self {
        self.paused = paused;
        self
    }

    pub fn run(mut self) {
        println!("controls:");
        println!("  left-drag        rotate");
        println!("  right/mid-drag   pan");
        println!("  scroll           zoom");
        println!("  space            pause / resume stepping");
        println!("  [ / ]            steps per frame - / +");
        println!("  - / =            particle radius - / +");
        println!("  R                reset camera");
        println!("  Esc              quit");

        let event_loop = EventLoop::new().expect("failed to create event loop");
        if let Err(e) = event_loop.run_app(&mut self) {
            eprintln!("event loop exited with an error: {e}");
        }
    }

    fn redraw(&mut self) {
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.positions = self.engine.simulation_step();
            }
        }

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let camera = self.camera.to_uniform(renderer.aspect());
        build_instances(
            &self.positions,
            self.particle_radius,
            self.box_length,
            &mut self.instances,
        );
        renderer.render(&camera, &self.instances);

        // Keep the render loop running.
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn update_title(&self) {
        let Some(window) = &self.window else {
            return;
        };
        let paused = if self.paused { " [paused]" } else { "" };
        window.set_title(&format!(
            "molecular-dynamics — {} particles, {} step(s)/frame, radius {:.3}{}",
            self.positions.len(),
            self.steps_per_frame,
            self.particle_radius,
            paused,
        ));
    }
}

impl<E: MdSimulation + 'static> ApplicationHandler for Visualizer<E> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return; // already initialized (resumed can fire again on some platforms)
        }

        let window = event_loop
            .create_window(Window::default_attributes().with_inner_size(
                winit::dpi::LogicalSize::new(1280.0, 800.0),
            ))
            .expect("failed to create window");
        let window = Arc::new(window);

        let renderer = pollster::block_on(Renderer::new(window.clone(), self.box_length));

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.update_title();
        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                self.drag_button = match state {
                    ElementState::Pressed => Some(button),
                    ElementState::Released => None,
                };
            }

            WindowEvent::CursorMoved { position, .. } => {
                let previous = self.last_cursor.replace((position.x, position.y));
                let Some((px, py)) = previous else {
                    return;
                };
                let Some(button) = self.drag_button else {
                    return;
                };

                let dx = (position.x - px) as f32;
                let dy = (position.y - py) as f32;
                match button {
                    MouseButton::Left => self.camera.rotate(dx, dy),
                    MouseButton::Right | MouseButton::Middle => {
                        let height = self
                            .renderer
                            .as_ref()
                            .map(|r| r.config_height())
                            .unwrap_or(800)
                            .max(1);
                        self.camera.pan(dx, dy, height as f32);
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.camera.zoom(steps);
            }

            WindowEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed {
                    return;
                }
                let mut update_title = true;
                match key_event.physical_key {
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.paused = !self.paused;
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                        update_title = false;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        self.camera =
                            OrbitCamera::new(Vec3::splat(self.box_length * 0.5), self.box_length * 2.2);
                    }
                    PhysicalKey::Code(KeyCode::Minus) => {
                        self.particle_radius = (self.particle_radius / 1.25).max(0.001);
                    }
                    PhysicalKey::Code(KeyCode::Equal) => {
                        self.particle_radius = (self.particle_radius * 1.25)
                            .min(self.box_length * 0.25);
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.steps_per_frame = (self.steps_per_frame - 1).max(1);
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.steps_per_frame = (self.steps_per_frame * 2).min(4096);
                    }
                    _ => update_title = false,
                }
                if update_title {
                    self.update_title();
                }
            }

            WindowEvent::RedrawRequested => self.redraw(),

            _ => {}
        }
    }
}
