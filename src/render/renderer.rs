//! wgpu renderer: draws the particle instances as shaded billboard spheres
//! plus a wireframe of the periodic box, with depth testing.
//!
//! Both pipelines share a single bind group holding just the camera uniform
//! (updated every frame). Per-particle data is fed through an *instance-rate
//! vertex buffer* instead of a storage buffer: that is core functionality
//! everywhere (including GLES 3.0 / WebGL2-class hardware like Ivy Bridge's
//! HD 4000, which has no storage-buffer support), and it is at least as fast.
//! When the simulation eventually moves to GPU compute, a single
//! `copy_buffer_to_buffer` per frame is all that is needed to feed it.

use std::sync::Arc;

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;
use winit::window::Window;

use super::camera::CameraUniform;
use super::hud::{build_char_instances, CharInstance, Hud, HudLine};
use super::instances::Instance;

/// Depth24Plus is guaranteed renderable on every backend (including
/// GLES 3.0 / WebGL2 class hardware); Metal/Vulkan map it to their native
/// 24/32-bit depth formats automatically.
///
/// Every pipeline used in the render pass must declare this format, even
/// the HUD which never tests or writes depth.
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.03,
    b: 0.05,
    a: 1.0,
};
const BOX_COLOR: [f32; 4] = [0.36, 0.50, 0.72, 1.0];

/// Corner table for the particle billboard (triangle strip, in [-1, 1]).
const CORNERS: [[f32; 2]; 4] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [-1.0, 1.0],
    [1.0, 1.0],
];

/// Printed when adapter selection fails: lists every adapter the system
/// exposes and whether it can present to our window surface. `request_adapter`
/// silently drops adapters that fail the surface check, so without this the
/// reason for a failure would be invisible.
async fn print_adapter_report(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) {
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    if adapters.is_empty() {
        println!("no GPU adapters are visible to wgpu at all");
        return;
    }
    println!("adapters visible to wgpu:");
    for adapter in &adapters {
        let info = adapter.get_info();
        let verdict = if adapter.is_surface_supported(surface) {
            "can present to the window"
        } else {
            "CANNOT present to the window -> rejected"
        };
        println!(
            "  - {} [{:?}, {:?}] — {}",
            info.name, info.backend, info.device_type, verdict
        );
    }
}

/// Printed when no GPU adapter can be created, listing the usual fixes.
const NO_ADAPTER_HINTS: &str = "no suitable GPU adapter was found (all enabled backends were tried: vulkan / metal / dx12 / gles).

If the adapter report above lists an adapter as CANNOT present, that is
exactly why it was rejected. This is usually fixable by installing a
driver, for example on Linux:

  - software Vulkan (lavapipe) — runs on any CPU, can always present, and is
    plenty for this renderer (the simulation runs on the CPU anyway):
      Arch:            sudo pacman -S vulkan-swrast
      Debian/Ubuntu:  sudo apt install mesa-vulkan-drivers libvulkan1
      Fedora:         sudo dnf install mesa-vulkan-drivers vulkan-loader
  - Mesa GLES drivers (hardware OpenGL on Intel / AMD):
      Debian/Ubuntu:  sudo apt install libegl1 libgles2 libgl1-mesa-dri
      Fedora:         sudo dnf install mesa-libGLES2 mesa-dri-drivers
  - or the proprietary driver for your GPU

Also make sure a display is reachable (DISPLAY or WAYLAND_DISPLAY is set).";

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,

    camera_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,

    particle_pipeline: wgpu::RenderPipeline,
    corner_buffer: wgpu::Buffer,

    box_pipeline: wgpu::RenderPipeline,
    box_vertex_buffer: Option<wgpu::Buffer>,
    box_vertex_count: u32,

    hud: Hud,
    /// Scratch buffer for the per-frame HUD character instances.
    hud_chars: Vec<CharInstance>,
}

impl Renderer {
    /// Creates the window surface and all GPU resources.
    ///
    /// `box_length <= 0.0` disables the box wireframe and ghosting.
    pub async fn new(window: Arc<Window>, box_length: f32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let surface = instance
            .create_surface(window.clone())
            .expect(
                "failed to create a window surface — is a display server reachable? \
                 (DISPLAY or WAYLAND_DISPLAY must be set)",
            );

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => {
                print_adapter_report(&instance, &surface).await;
                panic!("\n{NO_ADAPTER_HINTS}");
            }
        };

        let adapter_info = adapter.get_info();
        if adapter_info.device_type == wgpu::DeviceType::Cpu {
            println!(
                "note: no hardware GPU found; rendering with the software adapter \
                 '{}' ({:?}). This works, but installing a real GPU driver \
                 (e.g. mesa-vulkan-drivers) is much faster.",
                adapter_info.name, adapter_info.backend
            );
        } else {
            println!(
                "rendering with {} ({:?}, {:?})",
                adapter_info.name, adapter_info.backend, adapter_info.device_type
            );
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("molecular-dynamics renderer device"),
                ..Default::default()
            })
            .await
            .expect("failed to create GPU device");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by the adapter");
        surface.configure(&device, &config);
        let depth_view = create_depth_view(&device, config.width, config.height);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let instance_buffer = create_instance_buffer(&device, 1024);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let particle_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/particle.wgsl"));
        let box_shader = device.create_shader_module(wgpu::include_wgsl!("shaders/box.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // slot 0: billboard corners (per-vertex)
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    }),
                    // slot 1: per-particle data (instance-rate):
                    // location 1 = pos_radius, location 2 = color
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let corner_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("billboard corner buffer"),
            contents: bytemuck::cast_slice(&CORNERS),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let box_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("box wireframe pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &box_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &box_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state()),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let (box_vertex_buffer, box_vertex_count) = if box_length > 0.0 {
            let vertices = box_wireframe_vertices(box_length);
            let count = vertices.len() as u32;
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("box wireframe vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            (Some(buffer), count)
        } else {
            (None, 0)
        };

        let hud = Hud::new(&device, &queue, config.format);

        Self {
            device,
            queue,
            surface,
            config,
            depth_view,
            camera_buffer,
            bind_group,
            instance_buffer,
            particle_pipeline,
            corner_buffer,
            box_pipeline,
            box_vertex_buffer,
            box_vertex_count,
            hud,
            hud_chars: Vec::new(),
        }
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    pub fn config_height(&self) -> u32 {
        self.config.height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = create_depth_view(&self.device, width, height);
    }

    /// Uploads the camera, particles and HUD, then renders one frame.
    ///
    /// Handles surface loss transparently (reconfigures and skips the frame).
    pub fn render(
        &mut self,
        camera: &CameraUniform,
        instances: &[Instance],
        hud_lines: &[HudLine],
    ) {
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera));

        self.ensure_instance_capacity(instances.len() as u64);
        if !instances.is_empty() {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(instances),
            );
        }

        build_char_instances(
            hud_lines,
            (self.config.width as f32, self.config.height as f32),
            &mut self.hud_chars,
        );
        self.hud
            .prepare(&self.device, &self.queue, &self.hud_chars);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated => {
                // Recreate the swap chain with the last known size and retry
                // on the next redraw.
                let (w, h) = (self.config.width, self.config.height);
                self.resize(w, h);
                return;
            }
            // Occluded / timeout / lost / validation: skip this frame.
            _ => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // Particles (instanced billboards).
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.particle_pipeline);
            pass.set_vertex_buffer(0, self.corner_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.draw(0..4, 0..instances.len() as u32);

            // Box wireframe on top (depth-tested against the particles).
            if let Some(box_vertices) = &self.box_vertex_buffer {
                pass.set_pipeline(&self.box_pipeline);
                pass.set_vertex_buffer(0, box_vertices.slice(..));
                pass.draw(0..self.box_vertex_count, 0..1);
            }

            // HUD last: blends on top of everything, ignores depth.
            self.hud.draw(&mut pass, self.hud_chars.len() as u32);
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
    }

    /// Grows the instance vertex buffer when the number of instances no
    /// longer fits. No bind group rebuild is needed since the buffer is
    /// bound as a vertex buffer, not through the bind group.
    fn ensure_instance_capacity(&mut self, instance_count: u64) {
        let needed = instance_count * Instance::SIZE;
        if needed <= self.instance_buffer.size() {
            return;
        }
        let new_capacity = needed.next_power_of_two();
        self.instance_buffer =
            create_instance_buffer(&self.device, new_capacity / Instance::SIZE);
    }
}

fn depth_stencil_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_instance_buffer(device: &wgpu::Device, capacity_instances: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instance buffer"),
        size: capacity_instances * Instance::SIZE,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// 12 box edges as a line list. Vertex layout: [x, y, z, r, g, b, a, _pad]
/// (32-byte stride to keep the attribute offsets nicely aligned).
fn box_wireframe_vertices(l: f32) -> Vec<[f32; 8]> {
    let corners = [
        [0.0, 0.0, 0.0],
        [l, 0.0, 0.0],
        [l, 0.0, l],
        [0.0, 0.0, l],
        [0.0, l, 0.0],
        [l, l, 0.0],
        [l, l, l],
        [0.0, l, l],
    ];
    let edges = [
        // bottom face
        (0usize, 1usize),
        (1, 2),
        (2, 3),
        (3, 0),
        // top face
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        // verticals
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut vertices = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        for corner in [corners[a], corners[b]] {
            vertices.push([
                corner[0],
                corner[1],
                corner[2],
                BOX_COLOR[0],
                BOX_COLOR[1],
                BOX_COLOR[2],
                BOX_COLOR[3],
                0.0,
            ]);
        }
    }
    vertices
}

#[cfg(test)]
mod tests {
    use super::box_wireframe_vertices;

    /// The WGSL sources must parse *and* pass naga validation — this catches
    /// shader bugs without needing a GPU (wgpu validates again at runtime).
    fn validate(source: &'static str, name: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} failed validation: {e:?}"));
    }

    #[test]
    fn particle_shader_is_valid_wgsl() {
        validate(include_str!("shaders/particle.wgsl"), "shaders/particle.wgsl");
    }

    #[test]
    fn box_shader_is_valid_wgsl() {
        validate(include_str!("shaders/box.wgsl"), "shaders/box.wgsl");
    }

    #[test]
    fn hud_shader_is_valid_wgsl() {
        validate(include_str!("shaders/hud.wgsl"), "shaders/hud.wgsl");
    }

    #[test]
    fn box_wireframe_has_twelve_edges() {
        assert_eq!(box_wireframe_vertices(10.0).len(), 24);
    }
}
