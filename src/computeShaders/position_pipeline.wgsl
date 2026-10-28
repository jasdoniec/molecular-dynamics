struct Environment {
    size: f32,
    delta_time: f32,
}

@group(0) @binding(0)
var<uniform> env: Environment;

@group(0) @binding(1)
var<storage, read> velocity: array<f32>;

@group(0) @binding(2)
var<storage, read> acceleration: array<f32>;

@group(0) @binding(3)
var<storage, read_write> position: array<f32>;

fn scalar_rem_euclid(v: f32, size: f32) -> f32 {
    return v - size * floor(v / size);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x;

    if i >= arrayLength(&position) {
        return;
    }

    position[i] = position[i] + velocity[i] * env.delta_time
        + 0.5 * acceleration[i] * env.delta_time * env.delta_time;

    position[i] = scalar_rem_euclid(position[i], env.size);
}
