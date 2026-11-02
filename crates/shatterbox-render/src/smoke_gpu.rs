//! GPU smoke volume: a local 3D flow simulation plus a half-res raymarch.
//!
//! The simulation runs in one box around an active source. State lives in flat
//! storage buffers (`vec4` per cell: density, temperature, age, unused); the
//! obstacle mask is one `u32` per cell built by the game from voxel occupancy.
//! Every step is a separate entry point in `smoke_sim.wgsl`, dispatched inside a
//! single compute pass so one timestamp pair measures the whole simulation.
//!
//! The render pass (`smoke_volume.wgsl`) raymarches the box at half internal
//! resolution into `smoke_vol`, and `smoke_composite.wgsl` upsamples it over the
//! lit HDR image with a depth-aware 2x2 filter.

use std::borrow::Cow;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Grid of the local smoke box. 0.25 m x 0.3125 m cells give a 16 x 10 x 16 m
/// volume: enough for the 8 x 6 x 3.5 m test room and the 10 m shaft.
pub const SMOKE_GRID: (u32, u32, u32) = (48, 24, 48);
pub const SMOKE_SPACING: [f32; 3] = [1.0 / 3.0, 10.0 / 24.0, 1.0 / 3.0];
pub const SMOKE_HALF_EXTENTS: [f32; 3] = [
    SMOKE_GRID.0 as f32 * SMOKE_SPACING[0] * 0.5,
    SMOKE_GRID.1 as f32 * SMOKE_SPACING[1] * 0.5,
    SMOKE_GRID.2 as f32 * SMOKE_SPACING[2] * 0.5,
];
const CELLS: usize = (SMOKE_GRID.0 * SMOKE_GRID.1 * SMOKE_GRID.2) as usize;
const MAX_SOURCES: usize = 4;

/// One active emitter fed to the simulation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmokeSource {
    pub position: Vec3,
    pub radius: f32,
    /// Density injected per second at the centre.
    pub density: f32,
    /// Normalised temperature 0..1.5; drives buoyancy and the orange emission.
    pub temperature: f32,
    /// Upward velocity injected at the centre, m/s².
    pub updraft: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SimUniform {
    origin: [f32; 4],
    spacing: [f32; 4],
    grid: [u32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    source: [[f32; 4]; MAX_SOURCES],
    source2: [[f32; 4]; MAX_SOURCES],
}

struct SimPasses {
    advect_velocity: wgpu::ComputePipeline,
    divergence: wgpu::ComputePipeline,
    clear_pressure: wgpu::ComputePipeline,
    clear_all: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    advect_state: wgpu::ComputePipeline,
}

pub(crate) struct SmokeGpu {
    volume_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    sim: SimPasses,
    volume_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,

    params: wgpu::Buffer,
    state: [wgpu::Buffer; 2],
    solid: wgpu::Buffer,

    adv: [wgpu::BindGroup; 2],
    div: [wgpu::BindGroup; 2],
    clear: [wgpu::BindGroup; 2],
    jacobi: [[wgpu::BindGroup; 2]; 2],
    project: wgpu::BindGroup,
    advect: [wgpu::BindGroup; 2],

    /// Volume bind groups, indexed by the state parity written last.
    volume: Option<[wgpu::BindGroup; 2]>,
    composite: Option<wgpu::BindGroup>,

    /// Box min corner in world space.
    origin: Vec3,
    /// Parity of the state buffer that holds the current density.
    pub(crate) state_parity: usize,
    pub(crate) reset: bool,
    pub(crate) iterations: u32,
    pub(crate) active: bool,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_binding(
    binding: u32,
    buffer: &wgpu::Buffer,
    _read_only: bool,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: NonZeroU64::new(buffer.size()),
        }),
    }
}

impl SmokeGpu {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let iterations = std::env::var("SHATTERBOX_SMOKE_ITERATIONS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(6)
            .clamp(2, 16);

        let sim_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shatterbox smoke simulation"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/smoke_sim.wgsl"
            ))),
        });
        let volume_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shatterbox smoke volume"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/smoke_volume.wgsl"
            ))),
        });
        let composite_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shatterbox smoke composite"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../assets/shaders/smoke_composite.wgsl"
            ))),
        });

        let mut sim_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(size_of::<SimUniform>() as u64),
            },
            count: None,
        }];
        for binding in 1..=4 {
            sim_entries.push(storage_entry(binding, true));
        }
        for binding in 5..=8 {
            sim_entries.push(storage_entry(binding, false));
        }
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shatterbox smoke sim layout"),
            entries: &sim_entries,
        });
        let sim_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shatterbox smoke sim pipeline layout"),
            bind_group_layouts: &[Some(&sim_layout)],
            immediate_size: 0,
        });

        let compute = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&sim_pipeline_layout),
                module: &sim_module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let sim = SimPasses {
            advect_velocity: compute("Shatterbox smoke advect velocity", "advect_velocity"),
            divergence: compute("Shatterbox smoke divergence", "divergence_pass"),
            clear_pressure: compute("Shatterbox smoke clear pressure", "clear_pressure"),
            clear_all: compute("Shatterbox smoke clear all", "clear_all"),
            jacobi: compute("Shatterbox smoke jacobi", "jacobi"),
            project: compute("Shatterbox smoke project", "project"),
            advect_state: compute("Shatterbox smoke advect state", "advect_state"),
        };

        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shatterbox smoke volume layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size_of::<SimUniform>() as u64),
                    },
                    count: None,
                },
                storage_entry(3, true),
                storage_entry(4, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let volume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shatterbox smoke volume pipeline layout"),
                bind_group_layouts: &[Some(&volume_layout)],
                immediate_size: 0,
            });
        let volume_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shatterbox smoke volume pipeline"),
            layout: Some(&volume_pipeline_layout),
            module: &volume_module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shatterbox smoke composite layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shatterbox smoke composite pipeline layout"),
                bind_group_layouts: &[Some(&composite_layout)],
                immediate_size: 0,
            });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shatterbox smoke composite pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_module,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &composite_module,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let create_buffer = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let vec_bytes = (CELLS * 16) as u64;
        let scalar_bytes = (CELLS * 4) as u64;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shatterbox smoke params"),
            size: size_of::<SimUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let velocity = [
            create_buffer("Shatterbox smoke velocity a", vec_bytes),
            create_buffer("Shatterbox smoke velocity b", vec_bytes),
        ];
        let state = [
            create_buffer("Shatterbox smoke state a", vec_bytes),
            create_buffer("Shatterbox smoke state b", vec_bytes),
        ];
        let pressure = [
            create_buffer("Shatterbox smoke pressure a", scalar_bytes),
            create_buffer("Shatterbox smoke pressure b", scalar_bytes),
        ];
        let divergence = create_buffer("Shatterbox smoke divergence", scalar_bytes);
        let solid = create_buffer("Shatterbox smoke obstacle mask", scalar_bytes);

        let sim_group = |vel_in: &wgpu::Buffer,
                         vel_out: &wgpu::Buffer,
                         state_in: &wgpu::Buffer,
                         state_out: &wgpu::Buffer,
                         pressure_in: &wgpu::Buffer,
                         pressure_out: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox smoke sim bind group"),
                layout: &sim_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    buffer_binding(1, vel_in, true),
                    buffer_binding(2, state_in, true),
                    buffer_binding(3, &solid, true),
                    buffer_binding(4, pressure_in, true),
                    buffer_binding(5, vel_out, false),
                    buffer_binding(6, state_out, false),
                    buffer_binding(7, pressure_out, false),
                    buffer_binding(8, &divergence, false),
                ],
            })
        };

        let adv = [
            sim_group(
                &velocity[0],
                &velocity[1],
                &state[0],
                &state[1],
                &pressure[0],
                &pressure[1],
            ),
            sim_group(
                &velocity[0],
                &velocity[1],
                &state[1],
                &state[0],
                &pressure[0],
                &pressure[1],
            ),
        ];
        let div = [
            sim_group(
                &velocity[1],
                &velocity[0],
                &state[0],
                &state[1],
                &pressure[0],
                &pressure[1],
            ),
            sim_group(
                &velocity[1],
                &velocity[0],
                &state[1],
                &state[0],
                &pressure[0],
                &pressure[1],
            ),
        ];
        // Clear writes every ping-pong A buffer: velocity[0], state[0], pressure[0].
        let clear = sim_group(
            &velocity[1],
            &velocity[0],
            &state[1],
            &state[0],
            &pressure[1],
            &pressure[0],
        );
        let jacobi_group = |sp: usize, phase: usize| {
            let (p_in, p_out) = if phase == 0 {
                (&pressure[0], &pressure[1])
            } else {
                (&pressure[1], &pressure[0])
            };
            sim_group(
                &velocity[0],
                &velocity[1],
                &state[sp],
                &state[1 - sp],
                p_in,
                p_out,
            )
        };
        let jacobi = [
            [jacobi_group(0, 0), jacobi_group(0, 1)],
            [jacobi_group(1, 0), jacobi_group(1, 1)],
        ];
        let project = sim_group(
            &velocity[1],
            &velocity[0],
            &state[0],
            &state[1],
            &pressure[0],
            &pressure[1],
        );
        let advect = [
            sim_group(
                &velocity[0],
                &velocity[1],
                &state[0],
                &state[1],
                &pressure[0],
                &pressure[1],
            ),
            sim_group(
                &velocity[0],
                &velocity[1],
                &state[1],
                &state[0],
                &pressure[0],
                &pressure[1],
            ),
        ];

        Self {
            volume_layout,
            composite_layout,
            sim,
            volume_pipeline,
            composite_pipeline,
            params,
            state,
            solid,
            adv,
            div,
            clear: [clear.clone(), clear],
            jacobi,
            project,
            advect,
            volume: None,
            composite: None,
            origin: Vec3::ZERO,
            state_parity: 0,
            reset: true,
            iterations,
            active: false,
        }
    }

    /// Builds the volume and composite bind groups; called after a resize.
    pub(crate) fn bind_targets(
        &mut self,
        device: &wgpu::Device,
        _frame_uniform: &wgpu::Buffer,
        depth_view: &wgpu::TextureView,
        probe_view: &wgpu::TextureView,
        smoke_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) {
        let make_volume = |state: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shatterbox smoke volume bind group"),
                layout: &self.volume_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: _frame_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params.as_entire_binding(),
                    },
                    buffer_binding(3, state, true),
                    buffer_binding(4, &self.solid, true),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(probe_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(smoke_view),
                    },
                ],
            })
        };
        self.volume = Some([make_volume(&self.state[0]), make_volume(&self.state[1])]);
        self.composite = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shatterbox smoke composite bind group"),
            layout: &self.composite_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(smoke_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        }));
    }

    /// Uploads the obstacle mask (0/1 per cell, already dilated by the caller).
    pub(crate) fn upload_obstacles(&self, queue: &wgpu::Queue, mask: &[u32]) {
        let count = mask.len().min(CELLS);
        if count > 0 {
            queue.write_buffer(&self.solid, 0, bytemuck::cast_slice(&mask[..count]));
        }
    }

    pub(crate) fn write_params(
        &mut self,
        queue: &wgpu::Queue,
        dt: f32,
        time: f32,
        sources: &[SmokeSource],
    ) {
        let mut uniform = SimUniform {
            origin: [self.origin.x, self.origin.y, self.origin.z, 0.0],
            spacing: [SMOKE_SPACING[0], SMOKE_SPACING[1], SMOKE_SPACING[2], 0.0],
            grid: [SMOKE_GRID.0, SMOKE_GRID.1, SMOKE_GRID.2, 0],
            params: [dt.clamp(0.0, 1.0 / 30.0), 1.35, 1.0, 0.55],
            params2: [0.10, time, 0.015, 0.008],
            params3: [1.0, 0.0, 0.0, 0.0],
            source: [[0.0; 4]; MAX_SOURCES],
            source2: [[0.0; 4]; MAX_SOURCES],
        };
        let count = sources.len().min(MAX_SOURCES);
        for (index, source) in sources.iter().take(count).enumerate() {
            uniform.source[index] = [
                source.position.x,
                source.position.y,
                source.position.z,
                source.radius.max(0.25),
            ];
            uniform.source2[index] = [
                source.density.max(0.0),
                source.temperature.clamp(0.0, 1.5),
                source.updraft,
                0.0,
            ];
        }
        uniform.grid[3] = count as u32;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&uniform));
    }

    /// Anchors the box at `center`; the mask must be uploaded separately.
    pub(crate) fn set_center(&mut self, center: Vec3) {
        self.origin = center - Vec3::from_array(SMOKE_HALF_EXTENTS);
        self.reset = true;
        self.active = true;
    }

    pub(crate) fn encode_sim(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<&wgpu::QuerySet>,
        pair: Option<u32>,
    ) {
        let timestamp_writes = pair.and_then(|pair| {
            timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(pair * 2),
                end_of_pass_write_index: Some(pair * 2 + 1),
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shatterbox smoke simulation"),
            timestamp_writes,
        });
        let groups = CELLS as u32;
        if self.reset {
            pass.set_pipeline(&self.sim.clear_all);
            pass.set_bind_group(0, &self.clear[0], &[]);
            pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
            self.state_parity = 0;
            self.reset = false;
        }
        let sp = self.state_parity;
        pass.set_pipeline(&self.sim.advect_velocity);
        pass.set_bind_group(0, &self.adv[sp], &[]);
        pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.sim.divergence);
        pass.set_bind_group(0, &self.div[sp], &[]);
        pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.sim.clear_pressure);
        pass.set_bind_group(0, &self.clear[sp], &[]);
        pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        for iteration in 0..self.iterations {
            pass.set_pipeline(&self.sim.jacobi);
            pass.set_bind_group(0, &self.jacobi[sp][(iteration % 2) as usize], &[]);
            pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        }
        pass.set_pipeline(&self.sim.project);
        pass.set_bind_group(0, &self.project, &[]);
        pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.sim.advect_state);
        pass.set_bind_group(0, &self.advect[sp], &[]);
        pass.dispatch_workgroups(groups.div_ceil(64), 1, 1);
        drop(pass);
        self.state_parity = 1 - sp;
    }

    pub(crate) fn encode_volume(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pair: Option<u32>,
        size: (u32, u32),
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        let Some(groups) = &self.volume else {
            return;
        };
        let timestamp_writes = pair.and_then(|pair| {
            timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(pair * 2),
                end_of_pass_write_index: Some(pair * 2 + 1),
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Shatterbox smoke volume"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.volume_pipeline);
        pass.set_bind_group(0, &groups[self.state_parity], &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
    }

    pub(crate) fn encode_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let Some(group) = &self.composite else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shatterbox smoke composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: timestamps,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}
