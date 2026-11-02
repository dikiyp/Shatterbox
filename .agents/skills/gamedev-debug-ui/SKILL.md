---
name: gamedev-debug-ui
description: Use when adding egui debug overlays, frame timing panels, entity inspectors, renderer stat displays, in-game feature toggles, or puffin profiler integration to a Rust game. Also use when the user asks how to add a debug view or troubleshoot via in-game tooling.
---

# In-Game Debug UI for Rust Games

## Philosophy

Add debug views early, not after everything hurts. A frame time graph and entity inspector added at week 2 will catch performance regressions and data bugs before they compound. Debug UI is infrastructure, not a luxury.

---

## egui Integration with wgpu

Use the `egui-wgpu` crate for the wgpu backend. egui runs its own render pass after all game render passes.

```toml
[dependencies]
egui = "0.27"
egui-wgpu = "0.27"
egui-winit = "0.27"
```

The egui context lives in the renderer. Game code does not touch egui directly. Instead it fills a `DebugData` struct that the renderer reads.

```rust
pub struct DebugData {
    pub frame_time_ms: f32,
    pub draw_calls: u32,
    pub entity_count: usize,
    pub triangle_count: u32,
    pub texture_binds: u32,
    pub pipeline_switches: u32,
    pub custom_panels: Vec<Box<dyn DebugPanel>>,
}

pub trait DebugPanel {
    fn title(&self) -> &str;
    fn draw(&self, ui: &mut egui::Ui);
}
```

The renderer owns `egui::Context` and calls `render_debug_panel` at end of frame.

---

## Feature Flag Pattern

Guard all debug tooling behind a feature flag. In release builds, debug UI compiles out entirely.

```toml
[features]
dev-tools = ["egui", "egui-wgpu", "egui-winit", "puffin", "puffin_egui"]
```

```rust
#[cfg(feature = "dev-tools")]
mod debug_ui {
    use egui::Context;
    use crate::debug::DebugData;

    pub fn render_debug_panel(ctx: &Context, data: &DebugData) {
        egui::Window::new("Debug").show(ctx, |ui| {
            ui.label(format!("Frame: {:.2} ms", data.frame_time_ms));
            ui.label(format!("Draw calls: {}", data.draw_calls));
            ui.label(format!("Entities: {}", data.entity_count));
        });
    }
}
```

Toggle visibility with F1 or a console command. Never hardcode it always-visible.

```rust
#[cfg(feature = "dev-tools")]
if input.just_pressed(KeyCode::F1) {
    debug_state.visible = !debug_state.visible;
}
```

---

## Standard Debug Panels

Add these in roughly this order as the project grows.

**Frame timing graph** - rolling window of the last 120 frame times. Shows spikes immediately.

```rust
pub struct FrameTimeHistory {
    samples: VecDeque<f32>,
    capacity: usize,
}

impl FrameTimeHistory {
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(ms);
    }

    pub fn draw(&self, ui: &mut egui::Ui) {
        let values: Vec<f32> = self.samples.iter().copied().collect();
        egui::plot::Plot::new("frame_times")
            .height(60.0)
            .show(ui, |plot_ui| {
                plot_ui.line(egui::plot::Line::new(
                    values.iter().enumerate()
                        .map(|(i, &v)| [i as f64, v as f64])
                        .collect::<Vec<_>>(),
                ));
            });
    }
}
```

**Draw call and triangle counters** - increment during the render loop, reset at frame start.

**Entity count** - query your ECS for total live entities each frame.

**Renderer stats** - texture binds, pipeline switches, buffer uploads.

**Input state viewer** - show current key/button/axis states. Invaluable for diagnosing input bugs.

**Feature toggles** - `bool` flags gated behind `dev-tools`:

```rust
#[cfg(feature = "dev-tools")]
pub struct DevToggles {
    pub disable_shadows: bool,
    pub wireframe: bool,
    pub show_collision_aabbs: bool,
    pub freeze_physics: bool,
}
```

Renderer and physics systems read these flags; they do not query egui.

---

## Entity Inspector

Allow selecting an entity from a world-space click or a list. Show all components and their current values.

```rust
pub trait Inspect {
    fn inspect(&self, ui: &mut egui::Ui);
}

// Implement for each component type
impl Inspect for Position {
    fn inspect(&self, ui: &mut egui::Ui) {
        ui.label(format!("Position  x: {:.2}  y: {:.2}", self.x, self.y));
    }
}
```

Register inspectable component types in a table. On selection, iterate the table for the selected entity and call `inspect`. This is the fastest way to diagnose transform, velocity, and state machine bugs.

---

## Puffin Profiler

Add `puffin::profile_scope!` to each major system. `puffin_egui` renders the flamegraph inside an egui window.

```rust
#[cfg(feature = "dev-tools")]
puffin::set_scopes_on(true);

fn physics_system(...) {
    puffin::profile_scope!("physics_system");
    // ...
}

fn animation_system(...) {
    puffin::profile_scope!("animation_system");
    // ...
}
```

In the debug UI render pass:

```rust
#[cfg(feature = "dev-tools")]
puffin_egui::profiler_window(ctx);

#[cfg(feature = "dev-tools")]
puffin::GlobalProfiler::lock().new_frame();
```

Call `new_frame` once per game frame, outside any scope.

---

## Rules

- Debug UI must not affect gameplay logic. It is a read-only view of game state.
- Game code fills `DebugData`; it never calls into egui directly.
- All debug tools are guarded behind `#[cfg(feature = "dev-tools")]`.
- Release builds must not compile or link any debug UI code.
- Feature toggles live in a `DevToggles` resource; systems read the toggles, not egui widgets.
- The egui render pass runs after all game render passes; never interleave it.
