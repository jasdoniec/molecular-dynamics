# molecular-dynamics

A molecular dynamics simulation in Rust with a built-in real-time 3D visualizer.

## Features

- Lennard-Jones fluid with multiple particle species (per-species `sigma`,
  `epsilon`, `mass` and render color, with standard mixing rules)
- physics lives in one swappable force kernel (the `Potential` enum in
  `src/potential.rs`) — ion (Coulomb) and neural-network potentials can be
  added as new variants without touching the engine
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
  - on-screen HUD with live readouts (particle count, steps/frame, measured
    temperature in Kelvin) — the HUD is a plain list of text lines, so
    adding more readouts later is trivial

## Running

Either grab a prebuilt binary from the
[releases page](https://github.com/jasdoniec/molecular-dynamics/releases)
and run it directly, or build from source:

```bash
cargo build --release
./target/release/molecular-dynamics      # Windows: target\release\molecular-dynamics.exe
# or build-and-run in one step:
cargo run --release
```

The app runs on Linux, macOS and Windows — it just needs a desktop session
to open a window in.

- **macOS** — renders with Metal; any recent Mac works, nothing extra to
  install.
- **Windows** — renders with DirectX 12; just run the binary.
- **Linux** — renders with Vulkan, with an automatic GLES fallback:
  - any Vulkan-capable GPU driver, or on machines without a GPU software
    Vulkan (`lavapipe`): `sudo apt install mesa-vulkan-drivers libvulkan1`
    (Debian/Ubuntu), `sudo dnf install mesa-vulkan-drivers` (Fedora) or
    `sudo pacman -S vulkan-swrast` (Arch). GLES 3.0-class drivers also work.
  - the window connects to Wayland or X11; over SSH, run from a terminal
    inside the desktop session, or set `WAYLAND_DISPLAY` (or `DISPLAY`).

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
`sigma`, `epsilon` as ε/k_B in Kelvin, `mass`, `color`), temperature in
Kelvin, box size and timestep. Tweak the species list to build your own
mixtures — rendered sizes and colors follow automatically, and each species
sits at its own reduced temperature T* = T / (ε/k_B) while sharing one
physical temperature.

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
| `src/particle.rs` | particle state: positions/velocities, Verlet lists, minimum-image geometry |
| `src/potential.rs` | the force kernel: the `Potential` enum (Lennard-Jones today; ion and neural-network variants later) |
| `src/simulation_state.rs` | setup types (`SimulationState`, `ParticleInit`) |
| `src/md_simulation.rs` | the `MdSimulation` trait and `ParticlePaint` |
| `src/render/` | wgpu renderer, orbit camera, HUD, visualizer and shaders |

### Troubleshooting

If the app dies with a driver-level fault right after startup (e.g. a
segfault with Intel iGPUs via Vulkan on Windows), note that DX12 is already
the preferred backend there — the startup line tells you what was picked.
You can force a specific backend with the `WGPU_BACKEND` environment
variable:

```bash
WGPU_BACKEND=vulkan ./molecular-dynamics.exe    # force Vulkan on Windows
WGPU_BACKEND=dx12 ./molecular-dynamics.exe      # force DX12
WGPU_BACKEND=gl ./molecular-dynamics            # force GLES on Linux
```

The startup log prints which adapter and backend was picked, so you can
tell at a glance which one you are running on. Keeping the GPU driver up
to date also fixes a large share of startup crashes.
