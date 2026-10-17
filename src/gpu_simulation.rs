pub struct GpuSimulation {
    particles: Vec<crate::Particle>,

    device: wgpu::Device,
    queue: wgpu::Queue,

    position_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    forces_buffer: wgpu::Buffer,

    position_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    velocity_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl crate::MdSimulation for GpuSimulation {
    fn simulation_step(&mut self) -> Vec<crate::md_simulation::ParticlePaint> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let particle_count = self.particles.len() as u32;

        for _ in 0..1000 {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MD Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            compute_pass.set_pipeline(&self.position_pipeline);
            compute_pass.dispatch_workgroups((particle_count + 63) / 64, 1, 1);

            compute_pass.set_pipeline(&self.force_pipeline);
            compute_pass.dispatch_workgroups((particle_count + 63) / 64, 1, 1);

            compute_pass.set_pipeline(&self.force_pipeline);
            compute_pass.dispatch_workgroups((particle_count + 63) / 64, 1, 1);
        }

        vec![crate::md_simulation::ParticlePaint {
            position: [0.0, 0.0, 0.0],
            radius: 0.5,
            color: [1.0, 1.0, 1.0, 1.0],
        }]
    }

    // TODO: measure from the actual GPU velocity buffer once implemented.
    fn temperature_kelvin(&self) -> f32 {
        0.0
    }
}
