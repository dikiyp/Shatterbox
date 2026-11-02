//! Minimal winit 0.30 application shell.
//!
//! The platform crate owns the event loop and window; game code implements
//! [`App`] and receives already-created windows. This keeps winit types out of
//! the renderer's public surface.

use std::sync::Arc;

use anyhow::Result;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowId};

/// Window configuration used when the first `resumed` callback fires.
#[derive(Clone, Debug)]
pub struct PlatformConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Start borderless-fullscreen on the primary monitor.
    pub fullscreen: bool,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            title: String::from("Shatterbox"),
            width: 1600,
            height: 900,
            fullscreen: false,
        }
    }
}

/// Returned by app callbacks to request event-loop shutdown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppEvent {
    Continue,
    Exit,
}

/// Game-side interface to the platform event loop.
pub trait App {
    /// Called when the window is ready. Create/recreate GPU resources here.
    fn resumed(&mut self, window: Arc<Window>) -> AppEvent;

    /// Called for every window event. The platform handles `CloseRequested`.
    fn window_event(&mut self, window: &Window, event: &WindowEvent) -> AppEvent;

    /// Raw device events, used for relative mouse motion while captured.
    fn device_event(&mut self, _event: &DeviceEvent) {}

    /// Called once per event-loop iteration after pending events are drained.
    fn about_to_wait(&mut self, window: &Window) {
        window.request_redraw();
    }
}

/// Creates the event loop and runs the app until it requests exit.
pub fn run<A: App + 'static>(config: PlatformConfig, app: A) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut handler = Handler {
        app,
        config,
        window: None,
    };
    event_loop.run_app(&mut handler)?;
    Ok(())
}

struct Handler<A> {
    app: A,
    config: PlatformConfig,
    window: Option<Arc<Window>>,
}

impl<A: App + 'static> ApplicationHandler for Handler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attributes = Window::default_attributes()
                .with_title(&self.config.title)
                .with_inner_size(LogicalSize::new(
                    f64::from(self.config.width),
                    f64::from(self.config.height),
                ))
                .with_min_inner_size(LogicalSize::new(960.0, 540.0));

            let attributes = if self.config.fullscreen {
                attributes.with_fullscreen(Some(Fullscreen::Borderless(None)))
            } else {
                attributes
            };

            match event_loop.create_window(attributes) {
                Ok(window) => self.window = Some(Arc::new(window)),
                Err(error) => {
                    tracing::error!(%error, "failed to create window");
                    event_loop.exit();
                    return;
                }
            }
        }

        if let Some(window) = &self.window
            && self.app.resumed(window.clone()) == AppEvent::Exit
        {
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
            return;
        }

        let Some(window) = &self.window else {
            return;
        };

        if self.app.window_event(window, &event) == AppEvent::Exit {
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.app.device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            self.app.about_to_wait(window);
        }
    }
}
