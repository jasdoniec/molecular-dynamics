# molecular-dynamics

A molecular dynamics simulation in Rust with a built-in real-time 3D visualizer.

## Features

- Lennard-Jones fluid with multiple particle species (per-species `sigma`,
  `epsilon`, `mass` and render color, with standard mixing rules)
- Velocity Verlet integration on top of a cell list + Verlet (neighbor) list
- Periodic boundary conditions using the minimum-image convention
- CPU implementation today, GPU (compute shader) implementation in progress
- Real-time 3D renderer built on [wgpu](https://wgpu.rs):
  - orbit camera (rotate / zoom / pan)
  - per-particle colors and true, sigma-based radii
  - periodic "ghost" copies at the box edges, so particles that cross a
    wall visibly re-enter on the opposite side
  - near-camera fade: zoom in and the particles in front of you dissolve,
    so you can look inside the cloud
  - step-by-step mode for inspecting individual timesteps

Prebuilt binaries: see the [releases page](https://github.com/jasdoniec/molecular-dynamics/releases).

## Running

```bash
cargo run --release
```

Requirements:

- A display session (X11 or Wayland). Over SSH, run from a terminal inside
  the desktop session or set `WAYLAND_DISPLAY` (or `DISPLAY`) to point at it.
- Any Vulkan-capable GPU, or — on machines without one — software Vulkan
  (`lavapipe`): `sudo apt install mesa-vulkan-drivers libvulkan1`
  (Debian/Ubuntu), `sudo dnf install mesa-vulkan-drivers` (Fedora) or
  `sudo pacman -S vulkan-swrast` (Arch). GLES 3.0-class drivers also work.

### Controls

| input | action |
|---|---|
| left-drag | rotate |
| right/mid-drag | pan |
| scroll | zoom |
| `space` | pause / resume stepping |
| `→` (right arrow) | advance one step (pauses first) |
| `[` / `]` | steps per frame − / + |
| `,` / `.` | near fade − / + |
| `R` | reset camera |
| `Esc` | quit |

### Configuration

The demo setup lives in [`src/main.rs`](src/main.rs): species (count,
`sigma`, `epsilon`, `mass`, `color`), temperature, box size and timestep.
Tweak the species list to build your own mixtures — rendered sizes and
colors follow automatically.

## Tests

```bash
cargo test
```

Includes WGSL shader validation (via `naga`) and a 10,000-step stability
regression test that catches periodic-boundary integration bugs.

## Project layout

| path | contents |
|---|---|
| `src/cpu_simulation.rs` | CPU simulation: cell lists, Verlet lists, velocity Verlet |
| `src/gpu_simulation.rs` | GPU simulation (work in progress) |
| `src/particle.rs` | particle state and Lennard-Jones force math |
| `src/simulation_state.rs` | setup types (`SimulationState`, `ParticleInit`) |
| `src/md_simulation.rs` | the `MdSimulation` trait and `ParticlePaint` |
| `src/render/` | wgpu renderer, orbit camera, visualizer and shaders |
