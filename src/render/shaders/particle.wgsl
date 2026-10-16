// Particle billboards: one camera-facing quad per instance, shaded like a
// sphere. Per-particle data (position, radius, colour) arrives as an
// instance-rate *vertex buffer* instead of a storage buffer — core
// functionality everywhere, including GLES 3.0 / WebGL2-class hardware that
// has no storage-buffer support.

struct Camera {
    view_proj: mat4x4f,
    right: vec4f,
    up: vec4f,
    eye: vec4f,  // xyz = world-space eye position
    fade: vec4f, // x = near-fade start (0 = disabled), y = fully-visible distance
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) quad_pos: vec2f, // [-1, 1] within the billboard
    @location(1) color: vec4f,
};

@vertex
fn vs_main(
    @location(0) corner: vec2f,
    @location(1) pos_radius: vec4f, // xyz = world position, w = radius
    @location(2) color: vec4f,
) -> VSOut {
    let center = pos_radius.xyz;

    // Near-camera fade: particles inside the shell shrink smoothly to
    // nothing (degenerate billboards rasterize no pixels), so zooming in
    // lets you look inside the cloud. fade.x <= 0 disables it.
    let d = distance(center, camera.eye.xyz);
    let e0 = camera.fade.x;
    let e1 = max(camera.fade.y, e0 + 0.001);
    let k = select(1.0, smoothstep(e0, e1, d), e0 > 0.0);
    let radius = pos_radius.w * k;

    // Expand the quad in world space along the camera basis => billboard.
    let world = center
        + camera.right.xyz * (corner.x * radius)
        + camera.up.xyz * (corner.y * radius);

    var out: VSOut;
    out.clip_pos = camera.view_proj * vec4f(world, 1.0);
    out.quad_pos = corner;
    out.color = color;
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
