//! Platform-independent input snapshot and action mapping state.

use glam::Vec2;

/// Logical actions understood by the game layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Action {
    MoveForward = 0,
    MoveBackward,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Sprint,
    Look,
    ToggleOverlay,
    ResetCamera,
    ToggleGovernor,
    CycleRenderScale,
    CycleUpscaler,
    Cancel,
    Fire,
    AltFire,
    TimeBackward,
    TimeForward,
    ToggleFullscreen,
    Weapon1,
    Weapon2,
    Weapon3,
}

impl Action {
    pub const COUNT: usize = 22;

    const fn index(self) -> usize {
        self as usize
    }
}

/// Edge-sensitive keyboard/mouse state for one frame.
#[derive(Clone, Debug)]
pub struct InputState {
    down: [bool; Action::COUNT],
    pressed: [bool; Action::COUNT],
    released: [bool; Action::COUNT],
    mouse_delta: Vec2,
    scroll_delta: f32,
    focused: bool,
    cursor_captured: bool,
    gameplay_enabled: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            down: [false; Action::COUNT],
            pressed: [false; Action::COUNT],
            released: [false; Action::COUNT],
            mouse_delta: Vec2::ZERO,
            scroll_delta: 0.0,
            focused: true,
            cursor_captured: false,
            gameplay_enabled: true,
        }
    }
}

impl InputState {
    #[must_use]
    pub fn down(&self, action: Action) -> bool {
        self.down[action.index()]
    }

    #[must_use]
    pub fn pressed(&self, action: Action) -> bool {
        self.pressed[action.index()]
    }

    #[must_use]
    pub fn released(&self, action: Action) -> bool {
        self.released[action.index()]
    }

    /// Updates the held state and records press/release edges.
    pub fn set_action(&mut self, action: Action, is_down: bool) {
        let index = action.index();
        if self.down[index] == is_down {
            return;
        }
        self.down[index] = is_down;
        if is_down {
            self.pressed[index] = true;
        } else {
            self.released[index] = true;
        }
    }

    pub fn add_mouse_delta(&mut self, delta: Vec2) {
        self.mouse_delta += delta;
    }

    #[must_use]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// Returns the accumulated mouse delta and resets it.
    pub fn take_mouse_delta(&mut self) -> Vec2 {
        std::mem::take(&mut self.mouse_delta)
    }

    pub fn add_scroll(&mut self, delta: f32) {
        self.scroll_delta += delta;
    }

    pub fn take_scroll(&mut self) -> f32 {
        std::mem::take(&mut self.scroll_delta)
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
        if !focused {
            self.clear_actions();
        }
    }

    #[must_use]
    pub fn focused(&self) -> bool {
        self.focused
    }

    pub fn set_cursor_captured(&mut self, captured: bool) {
        self.cursor_captured = captured;
    }

    #[must_use]
    pub fn cursor_captured(&self) -> bool {
        self.cursor_captured
    }

    pub fn set_gameplay_enabled(&mut self, enabled: bool) {
        self.gameplay_enabled = enabled;
    }

    #[must_use]
    pub fn gameplay_enabled(&self) -> bool {
        self.gameplay_enabled
    }

    /// Clears held and edge state, e.g. when the window loses focus.
    pub fn clear_actions(&mut self) {
        self.down.fill(false);
        self.pressed.fill(false);
        self.released.fill(false);
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
    }

    /// Clears per-frame data. Call after gameplay and rendering have read it.
    pub fn end_frame(&mut self) {
        self.pressed.fill(false);
        self.released.fill(false);
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_count_covers_every_variant() {
        // Guards against adding a variant and forgetting to bump `COUNT`, which
        // used to panic with an index-out-of-bounds on the first key press.
        assert_eq!(Action::COUNT, Action::Weapon3 as usize + 1);
    }

    #[test]
    fn press_and_release_produce_edges() {
        let mut input = InputState::default();
        input.set_action(Action::MoveForward, true);
        assert!(input.down(Action::MoveForward));
        assert!(input.pressed(Action::MoveForward));
        assert!(!input.released(Action::MoveForward));

        input.end_frame();
        assert!(input.down(Action::MoveForward));
        assert!(!input.pressed(Action::MoveForward));

        input.set_action(Action::MoveForward, false);
        assert!(!input.down(Action::MoveForward));
        assert!(input.released(Action::MoveForward));
    }

    #[test]
    fn losing_focus_clears_everything() {
        let mut input = InputState::default();
        input.set_action(Action::Look, true);
        input.add_mouse_delta(Vec2::new(4.0, 2.0));
        input.set_focused(false);
        assert!(!input.down(Action::Look));
        assert_eq!(input.mouse_delta(), Vec2::ZERO);
    }
}
