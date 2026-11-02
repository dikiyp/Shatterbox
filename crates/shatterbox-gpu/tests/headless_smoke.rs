//! Headless GPU smoke test.
//!
//! This is intentionally outside the normal unit-test modules: it creates a
//! display-less wgpu instance, requests the best available adapter and builds a
//! trivial compute pipeline. Machines without a GPU or a working backend skip
//! instead of failing, because CI runners often have no graphics stack.

use std::borrow::Cow;

#[test]
fn headless_device_and_compute_pipeline() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(error) => {
            eprintln!("skipping headless GPU smoke test: no adapter ({error})");
            return;
        }
    };

    let info = adapter.get_info();
    let (device, _queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Shatterbox headless smoke device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping headless GPU smoke test: no device ({error})");
                return;
            }
        };

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shatterbox headless smoke shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
            r#"
            @compute @workgroup_size(1)
            fn main() {}
            "#,
        )),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Shatterbox headless smoke layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Shatterbox headless smoke pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    eprintln!(
        "headless GPU smoke test passed on {} ({:?})",
        info.name, info.backend
    );
}
