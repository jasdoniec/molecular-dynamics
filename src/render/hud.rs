//! Minimal on-screen HUD: an 8x8 bitmap font rendered as instanced quads.
//!
//! The HUD is data-driven: the caller supplies a list of [`HudLine`]s each
//! frame and the HUD draws whatever it is given — adding readouts later is
//! just pushing another line, no renderer changes needed.
//!
//! Layout (all in pixels, converted to NDC on the CPU):
//!
//! ```text
//!   +----------------------------------------
//!   |  particles: 1250          <- line 0
//!   |  steps/frame: 4           <- line 1
//!   |  T: 175 K                 <- line 2
//!   |  ...more lines           <- line n
//! ```

use font8x8::{BASIC_FONTS, UnicodeFonts};
use wgpu::util::DeviceExt;

/// One line of HUD text, drawn top-left, top-to-bottom, in the line's colour.
pub struct HudLine {
    pub text: String,
    pub color: [f32; 4],
}

impl HudLine {
    pub fn new(text: impl Into<String>, color: [f32; 4]) -> Self {
        Self {
            text: text.into(),
            color,
        }
    }
}

/// One renderable character quad. Instance-rate vertex buffer with two
/// `Float32x4` attributes (32-byte stride, same pattern as the particles).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CharInstance {
    /// NDC rect: x/y of the top-left corner, width/height of the quad
    /// (height is negative — NDC y grows downwards on screen).
    pub rect: [f32; 4],
    /// x: glyph index into the 16x8 font atlas grid,
    /// y: packed sRGB colour (see [`pack_color`]) stored in a f32's bits,
    /// z/w: unused.
    pub glyph_color: [f32; 4],
}

impl CharInstance {
    pub const SIZE: u64 = std::mem::size_of::<CharInstance>() as u64;
}

/// Source glyph resolution, pixels.
const GLYPH_PX: f32 = 8.0;
/// Glyphs per row in the atlas.
const GLYPHS_PER_ROW: u32 = 16;
/// On-screen glyph size multiplier (2 => 16 px characters).
const CHAR_SCALE: f32 = 2.0;
/// Vertical distance between lines, pixels.
const LINE_HEIGHT_PX: f32 = GLYPH_PX * CHAR_SCALE + 2.0;
/// Distance of the HUD block from the window edges, pixels.
const MARGIN_PX: f32 = 10.0;

/// Packs the lines into per-character instances.
///
/// `viewport` is the current render size in pixels (used to convert pixel
/// positions into NDC — no per-frame uniform needed, resizing is free).
pub fn build_char_instances(
    lines: &[HudLine],
    viewport: (f32, f32),
    out: &mut Vec<CharInstance>,
) {
    out.clear();
    let (vw, vh) = (viewport.0.max(1.0), viewport.1.max(1.0));
    let char_px = GLYPH_PX * CHAR_SCALE;
    let ndc_w = 2.0 * (char_px / vw);
    let ndc_h = -2.0 * (char_px / vh);

    for (line_index, line) in lines.iter().enumerate() {
        let y_px = MARGIN_PX + line_index as f32 * LINE_HEIGHT_PX;
        let y0 = 1.0 - 2.0 * (y_px / vh);
        let packed_color = f32::from_bits(pack_color(line.color));

        for (char_index, ch) in line.text.chars().enumerate() {
            let x_px = MARGIN_PX + char_index as f32 * char_px;
            let x0 = -1.0 + 2.0 * (x_px / vw);
            out.push(CharInstance {
                rect: [x0, y0, ndc_w, ndc_h],
                glyph_color: [glyph_index(ch) as f32, packed_color, 0.0, 0.0],
            });
        }
    }
}

/// Maps a character to its cell in the atlas (the atlas is laid out by
/// ASCII code point); anything outside printable ASCII falls back to `?`.
fn glyph_index(ch: char) -> u32 {
    let code = ch as u32;
    if code < 128 {
        code
    } else {
        b'?' as u32
    }
}

/// Packs an RGBA colour into the low 32 bits that the shader unpacks with
/// shifts (LSB -> MSB = R, G, B, A).
pub fn pack_color(color: [f32; 4]) -> u32 {
    let channel = |v: f32| -> u32 {
        // Pass NaN through as 0 rather than 255.
        if v.is_finite() {
            (v.clamp(0.0, 1.0) * 255.0).round() as u32
        } else {
            0
        }
    };
    channel(color[0])
        | (channel(color[1]) << 8)
        | (channel(color[2]) << 16)
        | (channel(color[3]) << 24)
}

/// Builds the 128x64 R8Unorm font atlas: 16 glyphs per row, 8 rows;
/// glyph bits come straight from `font8x8` (each byte one row, LSB =
/// leftmost pixel).
pub fn font_atlas_data() -> Vec<u8> {
    let mut atlas = vec![0u8; 128 * 64];
    for code in 0u32..128 {
        let ch = char::from_u32(code).unwrap_or('?');
        let glyph = BASIC_FONTS.get(ch).unwrap_or([0; 8]);
        let (col, row) = (code % GLYPHS_PER_ROW, code / GLYPHS_PER_ROW);
        for (y, bits) in glyph.iter().enumerate() {
            for x in 0..8usize {
                if (bits >> x) & 1 == 1 {
                    atlas[(row * 8 + y as u32) as usize * 128 + (col * 8 + x as u32) as usize] = 255;
                }
            }
        }
    }
    atlas
}

/// GPU side of the HUD: font atlas texture, pipeline and instance buffer.
pub struct Hud {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    corner_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
}

impl Hud {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
        // Font atlas: single-channel texture, nearest filtering.
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HUD font atlas"),
            size: wgpu::Extent3d {
                width: 128,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_atlas_data(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(128),
                rows_per_image: Some(64),
            },
            wgpu::Extent3d {
                width: 128,
                height: 64,
                depth_or_array_layers: 1,
            },
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HUD font sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HUD bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HUD bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HUD pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/hud.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("HUD pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // slot 0: quad corners (per-vertex), in [0, 1]
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    }),
                    // slot 1: per-character data (instance-rate)
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            // HUD draws last and ignores depth.
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let corner_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HUD corner buffer"),
            contents: bytemuck::cast_slice(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let instance_buffer = create_char_instance_buffer(device, 256);

        Self {
            pipeline,
            bind_group,
            corner_buffer,
            instance_buffer,
        }
    }

    /// Uploads the character instances, growing the buffer when needed.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, instances: &[CharInstance]) {
        let needed = instances.len() as u64 * CharInstance::SIZE;
        if needed > self.instance_buffer.size() {
            let new_capacity = needed.next_power_of_two();
            self.instance_buffer = create_char_instance_buffer(device, new_capacity / CharInstance::SIZE);
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
    }

    /// Records the HUD draw. Must be called last in the render pass
    /// (the HUD blends on top of everything).
    pub fn draw(&self, pass: &mut wgpu::RenderPass, char_count: u32) {
        if char_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.corner_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.draw(0..4, 0..char_count);
    }
}

fn create_char_instance_buffer(device: &wgpu::Device, capacity_chars: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HUD char instance buffer"),
        size: capacity_chars * CharInstance::SIZE,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_packing_matches_shader_layout() {
        // LSB -> MSB = R, G, B, A.
        assert_eq!(pack_color([1.0, 0.0, 0.0, 1.0]), 0xFF0000FF);
        assert_eq!(pack_color([0.0, 1.0, 0.0, 0.5]), 0x8000FF00);
        // Out-of-range clamps, NaN falls back to 0.
        assert_eq!(pack_color([2.0, -1.0, 0.0, 1.0]), 0xFF0000FF);
        assert_eq!(pack_color([f32::NAN, 0.0, 0.0, 1.0]), 0xFF000000);
    }

    #[test]
    fn one_instance_per_character() {
        let lines = vec![
            HudLine::new("particles: 1250", [1.0; 4]),
            HudLine::new("T: 175 K", [1.0; 4]),
        ];
        let mut out = Vec::new();
        build_char_instances(&lines, (800.0, 600.0), &mut out);
        assert_eq!(out.len(), 15 + 8);
    }

    #[test]
    fn instances_carry_ndc_rects_and_glyphs() {
        let lines = vec![HudLine::new("a!", [1.0, 0.0, 0.0, 1.0])];
        let mut out = Vec::new();
        build_char_instances(&lines, (800.0, 600.0), &mut out);

        let first = out[0];
        // First char starts at the margin: x0 = -1 + 2*10/800.
        assert!((first.rect[0] - (-1.0 + 2.0 * 10.0 / 800.0)).abs() < 1e-6);
        // Top of line 0: y0 = 1 - 2*10/600.
        assert!((first.rect[1] - (1.0 - 2.0 * 10.0 / 600.0)).abs() < 1e-6);
        // NDC y grows downwards => negative height.
        assert!(first.rect[3] < 0.0);
        // 'a' is ASCII 97.
        assert_eq!(first.glyph_color[0], 97.0);
        // Packed color round-trips through the f32 bits.
        assert_eq!(first.glyph_color[1].to_bits(), 0xFF0000FF);

        // Second character is one glyph width further right.
        let second = out[1];
        assert!(
            (second.rect[0] - first.rect[0] - first.rect[2]).abs() < 1e-5,
            "characters must advance by one glyph width"
        );
    }

    #[test]
    fn non_ascii_falls_back_to_question_mark() {
        let lines = vec![HudLine::new("λ", [1.0; 4])];
        let mut out = Vec::new();
        build_char_instances(&lines, (800.0, 600.0), &mut out);
        assert_eq!(out[0].glyph_color[0], b'?' as f32);
    }

    #[test]
    fn font_atlas_matches_source_glyphs() {
        let atlas = font_atlas_data();
        assert_eq!(atlas.len(), 128 * 64);

        // Every glyph's set pixels must land in its own atlas cell.
        for code in [b'!' as u32, b'A' as u32, b'0' as u32, b'z' as u32] {
            let ch = char::from_u32(code).unwrap();
            let glyph = BASIC_FONTS.get(ch).unwrap();
            let (col, row) = (code % GLYPHS_PER_ROW, code / GLYPHS_PER_ROW);
            let mut set_pixels = 0;
            for y in 0..8usize {
                for x in 0..8usize {
                    let atlas_px = atlas[(row * 8) as usize * 128 + y * 128 + (col * 8) as usize + x];
                    let glyph_px = if (glyph[y] >> x) & 1 == 1 { 255 } else { 0 };
                    assert_eq!(atlas_px, glyph_px, "glyph {ch} mismatch at {x},{y}");
                    set_pixels += u32::from(atlas_px == 255);
                }
            }
            assert!(set_pixels > 0, "glyph {ch} should not be empty");
        }
    }
}
