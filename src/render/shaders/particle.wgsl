// Particle billboards: one camera-facing quad per instance, shaded like a
// sphere. Instance data (position, radius, colour) comes from a storage buffer
// written by the CPU each frame; in the future the same buffer can be filled
// by a GPU compute step without changing this shader.

struct Camera {
    view_proj: mat4x4f,
    right: vec4f,
    up: vec4f,
};

struct Instance {
    pos_radius: vec4f, // xyz = world position, w = radius
    color: vec4f,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;

struct VSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) quad_pos: vec2f, // [-1, 1] within the billboard
    @location(1) color: vec4f,
};

@vertex
fn vs_main(
    @location(0) corner: vec2f,
    @builtin(instance_index) instance_idx: u32,
) -> VSOut {
    let inst = instances[instance_idx];
    let center = inst.pos_radius.xyz;
    let radius = inst.pos_radius.w;

    // Expand the quad in world space along the camera basis => billboard.
    let world = center
        + camera.right.xyz * (corner.x * radius)
        + camera.up.xyz * (corner.y * radius);

    var out: VSOut;
    out.clip_pos = camera.view_proj * vec4f(world, 1.0);
    out.quad_pos = corner;
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4f {
    let d = length(in.quad_pos);
    if (d > 1.0) {
        discard;
    }

    // Reconstruct a sphere normal in view space from the billboard offset.
    let z = sqrt(max(0.0, 1.0 - d * d));
    let n = normalize(vec3f(in.quad_pos.x, in.quad_pos.y, z));

    // Light fixed relative to the viewer: spheres always read as 3D.
    let light_dir = normalize(vec3f(0.35, 0.5, 0.8));
    let diffuse = max(dot(n, light_dir), 0.0);
    let shade = 0.35 + 0.75 * diffuse;

    let rgb = clamp(in.color.rgb * shade, vec3f(0.0), vec3f(1.0));
    return vec4f(rgb, 1.0);
}
