---
name: gamedev-wgsl-shaders
description: Use when writing or modifying WGSL shaders, debugging bind group mismatches, checking vertex attribute layouts, aligning uniform buffer structs, setting up shader hot reload, or ensuring Rust-side wgpu layouts match WGSL declarations.
---

# gamedev-wgsl-shaders

Skill for WGSL shader authoring and Rust/wgpu integration. Covers file layout, uniform alignment, bind group discipline, vertex formats, coordinate systems, and hot reload.

## Shader File Organization

Prefer external files over inline strings for anything longer than a trivial utility shader.

```
assets/
  shaders/
    sprite.wgsl
    terrain.wgsl
    post_process.wgsl
    common.wgsl        # shared structs/functions, included via naga-oil
```

Load at runtime for hot reload:

```rust
let source = std::fs::read_to_string("assets/shaders/sprite.wgsl").unwrap();
device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("sprite"),
    source: wgpu::ShaderSource::Wgsl(source.into()),
});
```

Use `include_str!` only for shaders that never need hot reload (e.g. internal blit passes).

## Uniform Buffer Alignment

Every WGSL struct used as a uniform buffer must be mirrored by a `#[repr(C)]` Rust struct with identical layout. Mismatches cause silent garbage data.

WGSL alignment rules (std140-like):
- `f32` / `i32` / `u32`: 4 bytes
- `vec2<f32>`: 8 bytes
- `vec3<f32>`: 16 bytes (same as vec4 - always pad to 16)
- `vec4<f32>` / `mat4x4<f32>` column: 16 bytes
- struct: rounded up to largest member alignment

```wgsl
// WGSL
struct CameraUniforms {
    view_proj: mat4x4<f32>,  // 64 bytes
    position:  vec3<f32>,    // 12 bytes
    _pad:      f32,          // 4 bytes padding to reach 16-byte boundary
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
```

```rust
// Rust mirror - must match exactly
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    view_proj: [[f32; 4]; 4],  // 64 bytes
    position:  [f32; 3],       // 12 bytes
    _pad:      f32,            // 4 bytes
}
```

Use `bytemuck::Pod` - it will refuse to compile if the struct has implicit padding, catching alignment bugs at compile time.

## Storage Buffers vs Uniform Buffers

Uniform buffers are capped at **65536 bytes** by the WebGPU spec. Exceeding this crashes on some backends.

```wgsl
// For large arrays: use storage
@group(1) @binding(0) var<storage, read> instances: array<InstanceData>;

// For small per-draw constants: use uniform
@group(0) @binding(1) var<uniform> time: TimeUniforms;
```

Prefer storage buffers for:
- Instance arrays (sprites, particles, tiles)
- Light lists
- Any array whose size is not known at shader-compile time

## Bind Group Layout: The Central Rule

Never scatter magic `@group(N) @binding(M)` literals across multiple files. Define one binding map in Rust and reference it everywhere.

```rust
// src/renderer/bindings.rs
pub mod group {
    pub const GLOBAL: u32 = 0;
    pub const MATERIAL: u32 = 1;
    pub const OBJECT: u32 = 2;
}

pub mod binding {
    // group 0
    pub const CAMERA: u32 = 0;
    pub const TIME: u32 = 1;
    pub const LIGHTS: u32 = 2;
    // group 1
    pub const ALBEDO_TEXTURE: u32 = 0;
    pub const ALBEDO_SAMPLER: u32 = 1;
    pub const NORMAL_TEXTURE: u32 = 2;
    // group 2
    pub const INSTANCES: u32 = 0;
}
```

The `BindGroupLayout` constructed in Rust must use the same indices. The WGSL `@group`/`@binding` attributes must match. Any mismatch is a runtime error caught by naga at pipeline creation.

## Vertex Format

`@location` indices in WGSL must match the `shader_location` fields in `wgpu::VertexAttribute`.

```wgsl
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
}
```

```rust
const VERTEX_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
    0 => Float32x3,  // position
    1 => Float32x2,  // uv
    2 => Float32x3,  // normal
];
```

For instanced rendering, instance attributes go in a second `VertexBufferLayout` with `step_mode: wgpu::VertexStepMode::Instance` and `@location` indices continuing from where the vertex buffer left off.

## Coordinate Systems

| Space | Convention |
|---|---|
| NDC | X: -1 left, +1 right. Y: -1 bottom, +1 top (Y-up). Z: 0..1 depth |
| Clip space | Right-handed, same Y-up as NDC |
| Texture UV | (0,0) top-left, Y-down (opposite of NDC Y) |
| wgpu depth | 0.0 = near, 1.0 = far |

When sampling a render target as a texture, flip UV.y if it was rendered with a standard camera matrix.

## Time Uniforms

WGSL has no built-in clock. Pass elapsed time from Rust.

```wgsl
struct TimeUniforms {
    elapsed:   f32,  // total seconds since start
    delta:     f32,  // seconds since last frame
    _pad:      vec2<f32>,
}
@group(0) @binding(1) var<uniform> time: TimeUniforms;
```

Write this uniform once per frame before any draw calls that consume it.

## Naga Validation

wgpu runs naga at `create_render_pipeline` time. Errors surface as panics or log messages - not at shader load.

Validation catches:
- Type mismatches between stages (vertex output != fragment input)
- Missing `@builtin(position)` in vertex output
- Unresolved bindings
- Struct size mismatches

Enable `wgpu` tracing in dev builds to see full naga errors:

```
RUST_LOG=wgpu_core=warn,naga=info cargo run
```

## Hot Reload Pattern

```rust
fn rebuild_pipeline(device: &wgpu::Device, path: &str) -> wgpu::RenderPipeline {
    let src = std::fs::read_to_string(path).expect("shader not found");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(path),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    build_pipeline(device, &module)
}
```

Watch shader files with `notify` crate. On change event, call `rebuild_pipeline` and swap the stored `RenderPipeline`. The old pipeline stays alive until all in-flight frames complete (wgpu handles this via `Arc` internally).

For larger projects, use `naga-oil` for `#import` preprocessing and shader composition before passing source to wgpu.

## Rules

- Never scatter raw `@group`/`@binding` numbers - keep one binding map in Rust.
- Every WGSL uniform struct must have a `#[repr(C)]` Rust mirror checked with `bytemuck::Pod`.
- Use storage buffers for arrays that may exceed 65536 bytes.
- `@location` indices in WGSL vertex structs must match `shader_location` in `VertexBufferLayout`.
- Pass time via a uniform - do not use WGSL clock functions (none exist in standard WGSL).
- Check naga validation output when a pipeline creation panics - the error is almost always there.
