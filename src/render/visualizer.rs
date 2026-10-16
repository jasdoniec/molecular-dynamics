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

use crate::md_simulation::{MdSimulation, ParticlePaint};

use super::camera::OrbitCamera;
use super::instances::{build_instances, Instance};
use super::renderer::Renderer;
use super::hud::HudLine;

/// Base HUD text colour.
const HUD_TEXT: [f32; 4] = [0.92, 0.94, 0.97, 1.0];
/// Highlight for the live temperature readout.
const HUD_TEMP: [f32; 4] = [0.55, 0.85, 1.0, 1.0];
/// Warning colour for paused state.
const HUD_PAUSED: [f32; 4] = [1.0, 0.72, 0.20, 1.0];

pub struct Visualizer<E: MdSimulation> {
    engine: E,
    box_length: f32,
    steps_per_frame: usize,
    paused: bool,

    camera: OrbitCamera,
    /// Paint data returned by the last `simulation_step()`.
    paints: Vec<ParticlePaint>,
    /// Scratch buffer for the per-frame render instances (with ghosts).
    instances: Vec<Instance>,
    /// Scratch buffer for the per-frame HUD lines.
    hud_lines: Vec<HudLine>,

    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    drag_button: Option<MouseButton>,
    last_cursor: Option<(f64, f64)>,
    /// Last title pushed to the window — set_title only when it changes.
    last_title: String,
}

impl<E: MdSimulation + 'static> Visualizer<E> {
    pub fn new(engine: E, box_length: f32) -> Self {
        let mut camera = OrbitCamera::new(Vec3::splat(box_length * 0.5), box_length * 2.2);
        // Near-camera fade: when zoomed in, particles right in front of the
        // eye vanish so you can look inside the cloud.
        camera.near_fade = [box_length * 0.07, box_length * 0.22];
        Self {
            engine,
            box_length,
            steps_per_frame: 1,
            paused: false,
            camera,
            paints: Vec::new(),
            instances: Vec::new(),
            hud_lines: Vec::new(),
            window: None,
            renderer: None,
            drag_button: None,
            last_cursor: None,
            last_title: String::new(),
        }
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
        println!("  right arrow      advance one step (pauses first)");
        println!("  [ / ]            steps per frame - / +");
        println!("  , / .            near fade - / + (front particles vanish when zoomed in)");
        println!("  R                reset camera");
        println!("  Esc              quit");

        let event_loop = match EventLoop::new() {
            Ok(event_loop) => event_loop,
            Err(e) => {
                eprintln!("failed to connect to a display server: {e}");
                eprintln!();
                eprintln!("your shell has no display attached — this happens over SSH or in");
                eprintln!("tmux/screen started from SSH, which don't inherit the desktop");
                eprintln!("session's environment. Either run from a terminal opened inside");
                eprintln!("the desktop session, or point the shell at the display server:");
                eprintln!();
                eprintln!("  ls $XDG_RUNTIME_DIR | grep wayland    # find the socket name");
                eprintln!("  export WAYLAND_DISPLAY=<socket>        # e.g. wayland-1");
                eprintln!("  # on X11 instead: export DISPLAY=:0");
                eprintln!();
                eprintln!("the window will then open on that machine's screen.");
                std::process::exit(1);
            }
        };
        if let Err(e) = event_loop.run_app(&mut self) {
            eprintln!("event loop exited with an error: {e}");
        }
    }

    fn redraw(&mut self) {
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.paints = self.engine.simulation_step();
            }
        }

        self.build_hud_lines();

        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let camera = self.camera.to_uniform(renderer.aspect());
        build_instances(
            &self.paints,
            self.box_length,
            &mut self.instances,
        );
        renderer.render(&camera, &self.instances, &self.hud_lines);

        self.update_title();

        // Keep the render loop running.
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Rebuilds the HUD content — push more [`HudLine`]s here to extend
    /// the readout; the renderer draws whatever it is given.
    fn build_hud_lines(&mut self) {
        self.hud_lines.clear();

        let temperature = self.engine.temperature_kelvin();
        let paused = if self.paused { " [PAUSED]" } else { "" };

        self.hud_lines.push(HudLine::new(
            format!("particles: {}", self.paints.len()),
            HUD_TEXT,
        ));
        self.hud_lines.push(HudLine::new(
            format!("steps/frame: {}{}", self.steps_per_frame, paused),
            if self.paused { HUD_PAUSED } else { HUD_TEXT },
        ));
        self.hud_lines.push(HudLine::new(
            format!("T: {temperature:.0} K"),
            HUD_TEMP,
        ));
        if self.camera.near_fade[0] > 0.0 {
            self.hud_lines.push(HudLine::new(
                format!("near fade: {:.2}", self.camera.near_fade[0]),
                HUD_TEXT,
            ));
        }
    }

    fn update_title(&mut self) {
        let Some(window) = &self.window else {
            return;
        };

        let paused = if self.paused { " [paused]" } else { "" };
        let fade = if self.camera.near_fade[0] > 0.0 {
            format!(", near fade {:.2}", self.camera.near_fade[0])
        } else {
            ", near fade off".to_string()
        };
        // Measured kinetic temperature — drifts as the NVE run equilibrates.
        let temperature = self.engine.temperature_kelvin();
        let title = format!(
            "molecular-dynamics — {} particles, {} step(s)/frame, T ≈ {:.0} K{}{}",
            self.paints.len(),
            self.steps_per_frame,
            temperature,
            fade,
            paused,
        );
        if title == self.last_title {
            return;
        }
        window.set_title(&title);
        self.last_title = title;
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
                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                        // Step: pause if still running, then advance exactly
                        // one simulation step and leave it on screen.
                        self.paused = true;
                        self.paints = self.engine.simulation_step();
                    }
                    PhysicalKey::Code(KeyCode::Escape) => {
                        event_loop.exit();
                        update_title = false;
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        self.camera =
                            OrbitCamera::new(Vec3::splat(self.box_length * 0.5), self.box_length * 2.2);
                    }
                    PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.steps_per_frame = (self.steps_per_frame - 1).max(1);
                    }
                    PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.steps_per_frame = (self.steps_per_frame * 2).min(4096);
                    }
                    PhysicalKey::Code(KeyCode::Comma) => {
                        // Shrink the near-fade shell; below ~2% of the box
                        // it turns off entirely.
                        let (start, end) = (self.camera.near_fade[0], self.camera.near_fade[1]);
                        if start > 0.0 {
                            let start = start * 0.8;
                            let end = end * 0.8;
                            self.camera.near_fade = if start < self.box_length * 0.02 {
                                [0.0, 0.0]
                            } else {
                                [start, end]
                            };
                        }
                    }
                    PhysicalKey::Code(KeyCode::Period) => {
                        if self.camera.near_fade[0] <= 0.0 {
                            // Re-enable with the default shell.
                            self.camera.near_fade =
                                [self.box_length * 0.07, self.box_length * 0.22];
                        } else {
                            let start = (self.camera.near_fade[0] * 1.25)
                                .min(self.box_length * 0.6);
                            let end = (self.camera.near_fade[1] * 1.25)
                                .min(self.box_length * 1.9);
                            self.camera.near_fade = [start, end];
                        }
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
