// On-screen HUD: one instanced quad per character, textured from the 8x8
// bitmap font atlas (16 glyphs per row => 128x64 texture).
//
// Per-character instances carry an NDC rect (positions were already
// converted from pixels on the CPU, so no per-frame uniform is needed) and
// the packed RGBA8 colour in the low 32 bits of a f32.

@group(0) @binding(0) var font: texture_2d<f32>;
@group(0) @binding(1) var font_sampler: sampler;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
};

@vertex
fn vs_main(
    @location(0) corner: vec2f,      // quad corner in [0, 1]^2
    @location(1) rect: vec4f,        // NDC x/y (top-left), w/h (h negative)
    @location(2) glyph_color: vec4f, // x: glyph index, y: packed RGBA8
) -> VSOut {
    let ndc = rect.xy + corner * rect.zw;

    let glyph = u32(glyph_color.x);
    let cell = vec2f(f32(glyph % 16u), f32(glyph / 16u));
    // Glyph cell size is 8px in a 128x64 atlas.
    let uv = (cell + corner) * vec2f(8.0) / vec2f(128.0, 64.0);

    let packed = bitcast<u32>(glyph_color.y);
    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    let a = f32((packed >> 24u) & 0xFFu) / 255.0;

    var out: VSOut;
    out.pos = vec4f(ndc, 0.0, 1.0);
    out.uv = uv;
    out.color = vec4f(r, g, b, a);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let coverage = textureSample(font, font_sampler, in.uv).r;
    if (coverage < 0.5) {
        discard;
    }
    return in.color;
}
