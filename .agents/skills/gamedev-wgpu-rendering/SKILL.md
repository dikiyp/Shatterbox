---
name: gamedev-wgpu-rendering
description: >
  Use when working on Rust rendering code that touches wgpu - including render
  pipelines, render passes, bind groups, vertex/index buffers, textures, depth
  buffers, MSAA, instancing, swapchain/surface handling, or renderer architecture.
  Also use when deciding how game world state should flow to the GPU, when handling
  window resize in the renderer, or when setting up the wgpu device/adapter/queue.
  Trigger this skill whenever the user mentions wgpu, render pass, pipeline, bind
  group, surface, texture sampler, depth buffer, or GPU buffer in a Rust context.
---

# Game Dev: wgpu Rendering

## Core principle

Keep renderer state and game world state separated. Game code submits renderable
data; it does not own wgpu resources. This is the most important structural rule
in a custom Rust engine - violating it creates tight coupling that makes the
renderer hard to change, test, or replace.

The data flow is:

```
World state -> extracted render data -> prepared GPU resources -> render pass
```

## Renderer architecture

### Device setup

Adapter, device, queue, and surface belong together in a single `Renderer` or
`GpuContext` struct. They are not global state. Pass them explicitly.

```rust
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub adapter: wgpu::Adapter,
}
```

Surface format, limits, and features should be selected at startup and stored.
Do not re-query them per frame.

### Resize handling

Surface-dependent resources must be recreated on resize: the surface config,
any textures sized to the framebuffer (depth buffer, MSAA resolve targets,
post-process intermediates), and the surface itself via `surface.configure`.

A resize should not rebuild pipelines. Pipelines are not surface-size dependent.

```rust
pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    if new_size.width == 0 || new_size.height == 0 {
        return;
    }
    self.surface_config.width = new_size.width;
    self.surface_config.height = new_size.height;
    self.surface.configure(&self.device, &self.surface_config);
    self.depth_texture = create_depth_texture(&self.device, &self.surface_config);
}
```

### Surface lost / outdated

`wgpu::SurfaceError::Lost` and `wgpu::SurfaceError::Outdated` both mean the
surface needs reconfiguring. Treat them identically: reconfigure and skip the
frame. `OutOfMemory` should propagate as a fatal error.

```rust
match surface.get_current_texture() {
    Ok(frame) => frame,
    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
        surface.configure(&device, &config);
        return;
    }
    Err(wgpu::SurfaceError::OutOfMemory) => panic!("GPU out of memory"),
    Err(e) => {
        eprintln!("surface error: {e:?}");
        return;
    }
}
```

## Pipelines

Create pipelines once at startup or on demand when a new material type is first
needed. Do not create pipelines per frame.

Group related pipelines in a struct. Name them after what they render, not after
implementation details.

```rust
pub struct Pipelines {
    pub opaque: wgpu::RenderPipeline,
    pub transparent: wgpu::RenderPipeline,
    pub shadow: wgpu::RenderPipeline,
    pub ui: wgpu::RenderPipeline,
}
```

### Bind group layouts

Bind group layouts must be created before pipelines that use them. Store layouts
separately from bind groups so you can create new bind groups (e.g., per texture)
without rebuilding pipelines.

```rust
pub struct Layouts {
    pub global: wgpu::BindGroupLayout,   // camera, time, lights
    pub material: wgpu::BindGroupLayout, // texture, sampler, material params
}
```

### Bind group numbers

Assign bind group slots by update frequency, not arbitrarily:

| Slot | Content | Update rate |
|------|---------|-------------|
| 0 | Global (camera, time, lights) | Once per frame |
| 1 | Material (texture, sampler) | Per material change |
| 2 | Object (transform, skinning) | Per draw call |

Document slot assignments in one place. Never scatter magic binding numbers
across Rust and WGSL files - they will drift.

## Buffers

### Uniform buffers

Use `wgpu::util::DeviceExt::create_buffer_init` for static data. Use
`queue.write_buffer` for data updated every frame.

```rust
queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
```

Align structs to `wgpu::COPY_BUFFER_ALIGNMENT` (16 bytes). Add explicit
`_padding` fields if needed. Mark structs `#[repr(C)]` and derive
`bytemuck::Pod + bytemuck::Zeroable`.

### Vertex buffers

Prefer interleaved vertex data (position + normal + uv in one buffer) for
static meshes. Use separate streams only when the use case warrants it
(e.g., skinning weights as a separate buffer updated by compute).

Avoid per-frame vertex buffer creation. Upload to a pre-allocated staging buffer
or use `queue.write_buffer` into a persistent vertex buffer.

### Storage buffers

Prefer storage buffers over large uniform arrays for instancing data. Uniform
buffers have a minimum guaranteed size limit (often 65536 bytes); storage buffers
do not.

## Depth buffer

Always create a depth texture to match the surface dimensions. Format is
typically `wgpu::TextureFormat::Depth32Float`. Recreate it on resize alongside
the surface.

Depth write and depth compare should both be set on the pipeline:

```rust
depth_stencil: Some(wgpu::DepthStencilState {
    format: wgpu::TextureFormat::Depth32Float,
    depth_write_enabled: true,
    depth_compare: wgpu::CompareFunction::Less,
    stencil: wgpu::StencilState::default(),
    bias: wgpu::DepthBiasState::default(),
}),
```

Transparent geometry should disable depth write but keep depth test enabled.

## Render passes

Structure passes by purpose, not by mesh type. Common order:

1. Shadow pass (if shadows are needed)
2. Opaque geometry pass
3. Transparent geometry pass (sorted back to front)
4. Post-processing passes
5. UI pass

Clear color and depth at the start of the opaque pass. Do not clear mid-frame
unless you have explicit reason.

Name your `RenderPassDescriptor` labels - they appear in GPU debuggers (RenderDoc,
PIX, Xcode GPU frame capture).

```rust
let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    label: Some("opaque pass"),
    ...
});
```

## Render data extraction

Do not read `wgpu::Buffer` back to CPU for gameplay decisions. Extract renderable
data from the game world into CPU-side structs, then write those structs to GPU
buffers.

```rust
pub struct RenderMesh {
    pub transform: Mat4,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub material_bind_group: wgpu::BindGroup,
}
```

The game world submits `RenderMesh` lists to the renderer. It never hands over
raw `wgpu::Device` or pipeline handles.

## What NOT to do

- Do not store `wgpu::Device`, `wgpu::Queue`, or pipelines inside gameplay components or ECS resources that gameplay systems touch.
- Do not create pipelines inside render functions that run every frame.
- Do not allocate new GPU buffers every frame unless you are intentionally streaming.
- Do not ignore `SurfaceError::Lost` or `SurfaceError::Outdated`.
- Do not read GPU buffers back to CPU on the hot path.
- Do not put magic binding numbers inline - centralize them.

## Common commands

```sh
cargo fmt
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

## Review checklist

Before submitting renderer changes:

- [ ] Resize path recreates depth texture and surface-dependent resources
- [ ] Surface errors are handled (Lost, Outdated, OutOfMemory)
- [ ] No pipelines created per frame
- [ ] No wgpu types in gameplay components
- [ ] Bind group slot assignments match WGSL `@group(N) @binding(M)`
- [ ] Uniform structs are `#[repr(C)]`, `Pod`, `Zeroable`, and aligned
- [ ] Render pass labels are set
- [ ] Transparent pass is sorted back to front if order matters
