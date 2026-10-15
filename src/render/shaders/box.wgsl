// Wireframe of the (periodic) simulation box: 12 edges drawn as a line list.

struct Camera {
    view_proj: mat4x4f,
    right: vec4f,
    up: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) color: vec4f,
};

@vertex
fn vs_main(@location(0) pos: vec3f, @location(1) color: vec4f) -> VSOut {
    var out: VSOut;
    out.clip_pos = camera.view_proj * vec4f(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    return in.color;
}
