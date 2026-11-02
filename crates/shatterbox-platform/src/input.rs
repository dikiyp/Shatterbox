//! winit event -> logical action mapping and cursor capture.

use glam::Vec2;
use shatterbox_core::{Action, InputState};
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

/// Owns input state and translates raw winit events into engine actions.
pub struct Input {
    state: InputState,
    last_cursor: Option<Vec2>,
    capture_active: bool,
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Input {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: InputState::default(),
            last_cursor: None,
            capture_active: false,
        }
    }

    #[must_use]
    pub fn state(&self) -> &InputState {
        &self.state
    }

    pub fn set_gameplay_enabled(&mut self, enabled: bool) {
        self.state.set_gameplay_enabled(enabled);
    }

    #[must_use]
    pub fn cursor_captured(&self) -> bool {
        self.capture_active
    }

    /// Locks the cursor so raw mouse motion always turns the view. Safe to call
    /// when the window has not received focus yet; the grab simply retries.
    pub fn capture(&mut self, window: &Window) {
        self.set_capture(window, true);
    }

    pub fn take_mouse_delta(&mut self) -> Vec2 {
        self.state.take_mouse_delta()
    }

    pub fn take_scroll(&mut self) -> f32 {
        self.state.take_scroll()
    }

    pub fn end_frame(&mut self) {
        self.state.end_frame();
    }

    /// Releases the cursor and clears held actions; used by Escape and focus loss.
    pub fn release_cursor(&mut self, window: &Window) {
        self.set_capture(window, false);
        self.state.clear_actions();
        self.last_cursor = None;
    }

    pub fn handle_window_event(&mut self, window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                if let Some(action) = map_key(code) {
                    self.state
                        .set_action(action, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                // First click only captures the cursor: otherwise the click
                // that gives the window focus would also fire a shot.
                if pressed && !self.capture_active {
                    self.set_capture(window, true);
                    self.last_cursor = None;
                    return;
                }
                match button {
                    MouseButton::Left => self.state.set_action(Action::Fire, pressed),
                    MouseButton::Right => self.state.set_action(Action::AltFire, pressed),
                    MouseButton::Middle => {
                        self.state.set_action(Action::Look, pressed);
                        if pressed {
                            self.set_capture(window, true);
                            self.last_cursor = None;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                if self.state.down(Action::Look) && !self.capture_active {
                    if let Some(previous) = self.last_cursor {
                        self.state.add_mouse_delta(position - previous);
                    }
                }
                self.last_cursor = Some(position);
            }
            WindowEvent::CursorLeft { .. } => {
                self.last_cursor = None;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32 / 120.0,
                };
                self.state.add_scroll(scroll);
            }
            WindowEvent::Focused(focused) => {
                self.state.set_focused(*focused);
                if *focused {
                    self.set_capture(window, true);
                } else {
                    self.release_cursor(window);
                }
            }
            _ => {}
        }
    }

    pub fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.capture_active
        {
            self.state
                .add_mouse_delta(Vec2::new(delta.0 as f32, delta.1 as f32));
        }
    }

    fn set_capture(&mut self, window: &Window, enabled: bool) {
        if enabled == self.capture_active {
            return;
        }

        if enabled {
            let result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));

            match result {
                Ok(()) => {
                    self.capture_active = true;
                    window.set_cursor_visible(false);
                }
                Err(error) => {
                    // Keep Look active: CursorMoved fallback still works without a grab.
                    self.capture_active = false;
                    tracing::warn!(%error, "cursor grab unavailable; using window-relative look");
                }
            }
        } else {
            if let Err(error) = window.set_cursor_grab(CursorGrabMode::None) {
                tracing::warn!(%error, "failed to release cursor grab");
            }
            window.set_cursor_visible(true);
            self.capture_active = false;
            self.last_cursor = None;
        }

        self.state.set_cursor_captured(self.capture_active);
    }
}

fn map_key(code: KeyCode) -> Option<Action> {
    match code {
        KeyCode::KeyW | KeyCode::ArrowUp => Some(Action::MoveForward),
        KeyCode::KeyS | KeyCode::ArrowDown => Some(Action::MoveBackward),
        KeyCode::KeyD | KeyCode::ArrowRight => Some(Action::MoveRight),
        KeyCode::KeyA | KeyCode::ArrowLeft => Some(Action::MoveLeft),
        KeyCode::Space => Some(Action::MoveUp),
        KeyCode::ControlLeft | KeyCode::ControlRight => Some(Action::MoveDown),
        KeyCode::ShiftLeft | KeyCode::ShiftRight => Some(Action::Sprint),
        KeyCode::F1 => Some(Action::ToggleOverlay),
        KeyCode::F2 => Some(Action::ToggleGovernor),
        KeyCode::F3 => Some(Action::CycleRenderScale),
        KeyCode::KeyU => Some(Action::CycleUpscaler),
        KeyCode::BracketLeft => Some(Action::TimeBackward),
        KeyCode::BracketRight => Some(Action::TimeForward),
        KeyCode::F11 => Some(Action::ToggleFullscreen),
        KeyCode::KeyR => Some(Action::ResetCamera),
        KeyCode::Digit1 => Some(Action::Weapon1),
        KeyCode::Digit2 => Some(Action::Weapon2),
        KeyCode::Digit3 => Some(Action::Weapon3),
        KeyCode::Escape => Some(Action::Cancel),
        _ => None,
    }
}
