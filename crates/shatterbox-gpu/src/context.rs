//! GPU bootstrap and surface lifetime management.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use winit::window::Window;

/// Preferred backend order for each supported desktop platform.
fn backend_order() -> &'static [wgpu::Backends] {
    #[cfg(target_os = "windows")]
    {
        // DX12 is the primary Windows backend; Vulkan is the documented fallback.
        &[wgpu::Backends::DX12, wgpu::Backends::VULKAN]
    }
    #[cfg(target_os = "linux")]
    {
        &[wgpu::Backends::VULKAN]
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        &[wgpu::Backends::all()]
    }
}

/// All GPU objects needed to render to one window surface.
pub struct GpuContext {
    // Kept alive because the surface was created from this instance.
    _instance: wgpu::Instance,
    pub surface: wgpu::Surface<'static>,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
    /// Whether timestamp queries were enabled on the device.
    pub timestamp_supported: bool,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
}

impl GpuContext {
    /// Creates a context, trying DX12 then Vulkan on Windows and Vulkan on Linux.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let mut errors = Vec::new();

        for &backends in backend_order() {
            match Self::new_with_backends(window.clone(), backends).await {
                Ok(context) => {
                    tracing::info!(
                        backend = ?context.adapter_info.backend,
                        adapter = %context.adapter_info.name,
                        "wgpu context created"
                    );
                    return Ok(context);
                }
                Err(error) => {
                    tracing::warn!(?backends, %error, "GPU backend unavailable, trying next");
                    errors.push(format!("{backends:?}: {error:#}"));
                }
            }
        }

        Err(anyhow!(
            "no usable GPU backend found:\n{}",
            errors.join("\n")
        ))
    }

    async fn new_with_backends(window: Arc<Window>, backends: wgpu::Backends) -> Result<Self> {
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = backends;
        instance_descriptor.flags = if cfg!(debug_assertions) {
            wgpu::InstanceFlags::DEBUG | wgpu::InstanceFlags::VALIDATION
        } else {
            wgpu::InstanceFlags::default()
        };

        let instance = wgpu::Instance::new(instance_descriptor);
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create wgpu surface from window")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .context("no compatible high-performance adapter")?;

        let adapter_info = adapter.get_info();
        let available_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        let timestamp_supported = available_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        if timestamp_supported {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Shatterbox device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to request wgpu device")?;
        // Capture validation/device errors so the autotest report can fail the
        // run instead of quietly rendering with a fallback pipeline.
        crate::diagnostics::install(&device);
        let timestamp_period = queue.get_timestamp_period();

        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            bail!("window has a zero-sized drawable");
        }

        let capabilities = surface.get_capabilities(&adapter);
        if capabilities.formats.is_empty() {
            bail!("surface reports no compatible texture formats");
        }

        // egui's shader outputs display-referred color, so a non-sRGB swapchain
        // is preferred. Fall back to whatever the surface offers (e.g. sRGB-only).
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or(capabilities.formats[0]);

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow!("surface has no default configuration"))?;
        config.format = format;
        // `SHATTERBOX_NO_VSYNC=1` uses Immediate for performance measurements.
        // The shipping default is `FifoRelaxed`: it is still tear-free, but a
        // late frame is presented right away instead of waiting for the next
        // vblank, which is what made 80-100 fps on a 144 Hz panel feel choppy.
        config.present_mode = if std::env::var_os("SHATTERBOX_NO_VSYNC").is_some() {
            wgpu::PresentMode::Immediate
        } else if capabilities
            .present_modes
            .contains(&wgpu::PresentMode::FifoRelaxed)
        {
            wgpu::PresentMode::FifoRelaxed
        } else {
            wgpu::PresentMode::AutoVsync
        };
        config.desired_maximum_frame_latency = 2;
        config.view_formats = vec![format];

        surface.configure(&device, &config);

        Ok(Self {
            _instance: instance,
            surface,
            adapter,
            device,
            queue,
            config,
            adapter_info,
            timestamp_supported,
            timestamp_period,
        })
    }

    /// Reconfigures the swapchain after a window resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn reconfigure(&self) {
        self.surface.configure(&self.device, &self.config);
    }

    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    #[must_use]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_info.name
    }

    #[must_use]
    pub fn backend_name(&self) -> String {
        format!("{:?}", self.adapter_info.backend)
    }
}
