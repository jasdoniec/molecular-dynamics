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
    fn simulation_step(&mut self) -> Vec<[f32; 3]> {
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

        vec![[0 as f32, 0 as f32, 0 as f32]]
    }
}
