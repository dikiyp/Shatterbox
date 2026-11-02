---
name: gamedev-profiling
description: Use when the user wants to profile a Rust game, add puffin or tracing instrumentation, find per-frame allocations, measure GPU timing, or decide what to optimize. Trigger when user says "too slow", "frame drops", "profiling", or asks how to find performance bottlenecks.
---

# Rust Game Profiling

**Core rule: do not guess. Measure first, then optimize.**

Every optimization must be preceded by a measurement that shows the problem.
No measurement, no change.

---

## CPU Profiling Tools

**puffin** - preferred for in-game profiling
- Lightweight flamegraph rendered inside your egui panel
- Minimal overhead, can be left in dev builds
- Spans are added with macros; no sampling required

**cargo flamegraph** - for cold-path and startup analysis
- Full call graph from a real run
- Requires `perf` on Linux or `dtrace` on macOS
- Run with `cargo flamegraph --bin my-game`
- Best for: startup time, load spikes, one-shot expensive operations

**tracy** - for deep frame analysis
- External profiler with timeline, memory, and zone views
- Higher integration cost (tracy client crate + tracy app)
- Use when puffin doesn't give enough resolution

---

## Adding puffin Spans

Add to `Cargo.toml`:

```toml
[dependencies]
puffin = { version = "0.19", optional = true }

[features]
profiling = ["puffin/puffin"]
```

Add spans to systems:

```rust
fn update_physics(&mut self) {
    puffin::profile_function!();

    {
        puffin::profile_scope!("broad phase");
        // ...
    }

    {
        puffin::profile_scope!("narrow phase");
        // ...
    }
}
```

Enable in your main loop (dev builds only):

```rust
#[cfg(feature = "profiling")]
puffin::GlobalProfiler::lock().new_frame();
```

Show the puffin panel in egui:

```rust
#[cfg(feature = "profiling")]
puffin_egui::profiler_window(ctx);
```

---

## tracing Spans (Always-on Structured Timing)

Use `tracing` when you want timing that works in release builds and integrates
with log subscribers.

```rust
use tracing::{instrument, info_span};

#[instrument]
fn load_assets(&self) {
    // entire function is a span
}

fn render_frame(&self) {
    let _span = info_span!("render_frame").entered();
    // span ends when _span drops
}
```

Wire up a subscriber in main:

```rust
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

For performance-sensitive code, prefer puffin. Use tracing for load paths,
background tasks, and anything that runs outside the hot loop.

---

## GPU Timing with wgpu

Use timestamp queries to measure GPU pass durations.

Requirements:
- `wgpu::Features::TIMESTAMP_QUERY` must be enabled on the device
- Results are async: read back on the next frame

Minimal pattern:

```rust
// Create a query set with 2 slots per pass (begin + end)
let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
    ty: wgpu::QueryType::Timestamp,
    count: 2,
    ..Default::default()
});

// In your render pass:
encoder.write_timestamp(&query_set, 0);
// ... render commands ...
encoder.write_timestamp(&query_set, 1);

// Resolve to a buffer, then map and read back
encoder.resolve_query_set(&query_set, 0..2, &resolve_buf, 0);
```

Divide the timestamp delta by `queue.get_timestamp_period()` to get nanoseconds.

**Interpreting results:**
- CPU submit time much shorter than GPU time: GPU-bound
- GPU idle while CPU submits: CPU-bound or poor parallelism
- Compare submit time vs frame time to find which is the bottleneck

---

## Allocations Per Frame

Zero-allocation per frame is the goal for the hot loop.

**Find allocations:**
- `cargo-heaptrack` (Linux): `heaptrack cargo run --release`
- `dhat` crate: add as a dev dependency, runs inline, outputs JSON

**Common causes:**
- `Vec::push` into a vec that wasn't pre-allocated
- `String::from` / `format!` in hot paths
- `Box::new` or `Arc::new` mid-frame
- Collecting iterators into new allocations unnecessarily

**Fix pattern:**
Pre-allocate scratch buffers at startup. Clear and reuse each frame.

```rust
struct FrameScratch {
    visible_entities: Vec<Entity>,
    draw_calls: Vec<DrawCall>,
}

impl FrameScratch {
    fn reset(&mut self) {
        self.visible_entities.clear();
        self.draw_calls.clear();
    }
}
```

---

## Asset Loading Stalls

Never load from disk on the main thread during gameplay.

Profile blocking loads with tracing:

```rust
let _span = info_span!("load_texture", path = %path.display()).entered();
let data = std::fs::read(&path)?;
```

If spans show load calls inside the frame loop, move them to a background thread
or an async loader and stream results back via a channel.

---

## Profiling Workflow

1. Add `puffin::profile_function!()` to every major system (physics, rendering,
   input, audio, AI).
2. Run the game, open the puffin panel, identify the slowest system.
3. Add `puffin::profile_scope!("...")` inside the slow system to isolate the
   bottleneck to a specific block.
4. Measure the baseline time. Make one change. Measure again.
5. Only keep the change if the measurement improves.

---

## Rules

- No optimization without a measurement that shows the problem.
- Keep all profiling instrumentation behind `feature = "profiling"` or
  `feature = "dev-tools"`. Never enable in default release builds.
- Release builds: puffin disabled, tracing subscriber set to warn or error only.
- Do not profile debug builds. Always profile with `--release`.
