---
name: gamedev-winit-loop
description: Use when creating or modifying the winit event loop, app lifecycle (ApplicationHandler), frame timing, fixed timestep logic, window creation, resize handling, or input event collection in a Rust game.
---

# gamedev-winit-loop

Skill for working with the winit event loop, window lifecycle, frame timing, and input collection in a Rust game. Targets winit 0.30+ (the `ApplicationHandler` API).

## Core API: ApplicationHandler

Always use `ApplicationHandler`, not the deprecated `EventLoop::run` closure pattern.

```rust
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Window>,
    renderer: Option<Renderer>,
    game: GameState,
    input: InputState,
    timer: FrameTimer,
    accumulator: f64,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window here, not in main()
        let attrs = Window::default_attributes().with_title("My Game");
        let window = event_loop.create_window(attrs).unwrap();
        self.renderer = Some(Renderer::new(&window));
        self.window = Some(window);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Drop GPU resources if targeting mobile/web
        self.renderer = None;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { .. }
            | WindowEvent::MouseInput { .. }
            | WindowEvent::CursorMoved { .. } => {
                self.input.record(&event);
            }
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size); // renderer owns surface recreation
                }
            }
            WindowEvent::RedrawRequested => self.on_redraw(),
            _ => {}
        }
    }
}
```

## Window Creation

- Always create the window inside `resumed()`, not in `main()` or `new()`.
- On Android and web, `resumed()` may be called multiple times - guard with `if self.window.is_none()`.
- Request redraws from within the event loop, not from threads: `window.request_redraw()`.

## RedrawRequested: Driving the Game Loop

`RedrawRequested` is the canonical place to run one full game tick. Call `request_redraw()` at the end of each frame to keep it looping.

```rust
fn on_redraw(&mut self) {
    const FIXED_STEP: f64 = 1.0 / 60.0; // 60 Hz physics

    let dt = self.timer.tick(); // seconds since last frame, capped
    self.accumulator += dt;

    while self.accumulator >= FIXED_STEP {
        let snapshot = self.input.snapshot();
        self.game.fixed_update(&snapshot, FIXED_STEP);
        self.accumulator -= FIXED_STEP;
    }

    let alpha = self.accumulator / FIXED_STEP; // interpolation factor [0,1)
    if let Some(r) = &mut self.renderer {
        r.render(self.game.extract_render_data(alpha));
    }

    self.input.end_frame();

    if let Some(w) = &self.window {
        w.request_redraw();
    }
}
```

## Frame Timing

Cap `dt` to prevent the "spiral of death" when the window is moved or the system stalls.

```rust
struct FrameTimer {
    last: std::time::Instant,
}

impl FrameTimer {
    fn new() -> Self { Self { last: std::time::Instant::now() } }

    fn tick(&mut self) -> f64 {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last).as_secs_f64();
        self.last = now;
        dt.min(0.25) // cap at 250ms
    }
}
```

## Fixed vs Variable Timestep

| Concern | Timestep |
|---|---|
| Physics, collisions, game logic | Fixed (e.g. 60 Hz) |
| Rendering, interpolation | Variable (render every frame) |
| Animation blending | Variable or fixed, consistent with physics |

Never run physics at a variable rate. It produces non-deterministic results and tunneling at low framerates.

## Input Collection Rule

**Do not process gameplay logic inside raw winit event handlers.** Event handlers are interrupt-like callbacks - they must only record state.

```
event_handler  ->  InputState.record(event)   // collect only
update()       ->  InputState.snapshot()      // read and act
```

`InputState` holds keys-held, keys-just-pressed, mouse delta, etc. `snapshot()` returns an immutable copy consumed by `fixed_update`. `end_frame()` clears per-frame fields (just-pressed, just-released).

## Resize Handling

Delegate surface recreation entirely to the renderer. The event loop only forwards the new size.

```rust
WindowEvent::Resized(size) => {
    if size.width > 0 && size.height > 0 {
        if let Some(r) = &mut self.renderer { r.resize(size); }
    }
}
```

Guard against zero-size (minimized window) to avoid wgpu surface errors.

## Pause / Resume

```rust
fn suspended(&mut self, _: &ActiveEventLoop) {
    self.renderer = None; // release surface
}

fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
        // first launch
        let w = event_loop.create_window(Window::default_attributes()).unwrap();
        self.window = Some(w);
    }
    let window = self.window.as_ref().unwrap();
    self.renderer = Some(Renderer::new(window));
}
```

## Entry Point

```rust
fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
```

Use `ControlFlow::Poll` for games (busy loop). Use `ControlFlow::Wait` only for editor-style tools where frames should not render unless something changes.

## Rules

- Never call `std::thread::sleep` inside the event loop to cap framerate - use `ControlFlow::WaitUntil` or a platform vsync swap chain instead.
- Never store `ActiveEventLoop` outside the callback - it is not `'static`.
- Do not put gameplay logic inside event handlers.
- Resize events must not crash on zero-size windows (minimized).
- Surface recreation belongs in the renderer, not in the event loop.
