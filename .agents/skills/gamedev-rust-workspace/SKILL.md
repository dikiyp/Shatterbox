---
name: gamedev-rust-workspace
description: >
  Use when setting up CI, configuring a Cargo workspace, choosing error
  handling patterns, adding clippy lints, or enforcing project-wide Rust code
  quality standards in a game project. Covers workspace dependency management,
  lint configuration, test runners, feature flags for dev tools, error type
  strategy, and minimum CI requirements.
---

# Rust Game Workspace Hygiene

## Formatting

Run `cargo fmt --all` before every commit. Enforce in CI with:

```bash
cargo fmt --all -- --check
```

Do not configure custom `rustfmt.toml` settings unless the team has explicitly
agreed on them. Default rustfmt style is the baseline. Custom settings create
friction for contributors and tooling.

## Clippy

The project must pass clippy clean with no warnings treated as allowed noise:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

- `--workspace` catches all crates, not just the current one.
- `--all-targets` includes examples, tests, and benchmarks.
- `--all-features` ensures feature-gated code is linted.
- `-D warnings` treats all warnings as errors.

Configure project-wide lints in the root `Cargo.toml`:

```toml
[workspace.lints.rust]
unsafe_code = "forbid"
unused_imports = "warn"

[workspace.lints.clippy]
pedantic = "warn"
cast_possible_truncation = "allow"   # common in graphics math
cast_sign_loss = "allow"             # same
```

Inherit workspace lints in each crate:

```toml
# crates/game-core/Cargo.toml
[lints]
workspace = true
```

## Tests

Prefer `cargo nextest` for faster parallel test execution:

```bash
cargo nextest run --workspace
```

Fall back to standard test if nextest is not available:

```bash
cargo test --workspace
```

Both commands must pass before merge. Do not skip tests in CI with
`-- --ignored` unless specific tests are explicitly tagged `#[ignore]`
with a documented reason.

## Workspace Dependencies

Declare all shared dependency versions in the root `Cargo.toml`. This is the
single source of truth for library versions across all crates.

```toml
# Root Cargo.toml
[workspace.dependencies]
wgpu = "0.20"
winit = "0.30"
glam = "0.27"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
```

Reference workspace versions in each crate without repeating the version:

```toml
# crates/game-render/Cargo.toml
[dependencies]
wgpu.workspace = true
glam.workspace = true
tracing.workspace = true
```

Never specify a version number in a crate-level `[dependencies]` table for any
dependency declared in `[workspace.dependencies]`. Inconsistent versions cause
subtle breakage and confusing lock file churn.

## Crate Layering - No Circular Dependencies

Enforce a strict dependency direction. The allowed graph is:

```
game-app
  -> game-platform  (winit, OS)
  -> game-render    (wgpu)
  -> game-audio     (rodio/cpal)
  -> game-assets    (file I/O, asset handles)
  -> game-core      (ECS, logic - no engine deps)
```

`game-core` depends on nothing in the workspace. Any crate may depend on
`game-core`. No crate except `game-app` may depend on `game-platform`.

If you feel the urge to add a backward dependency, extract the shared type into
`game-core` or a new utility crate instead.

Cargo will catch actual cycles at compile time, but layering violations that
do not form cycles (e.g. `game-render` importing `game-platform`) are
architectural debt - catch them in code review.

## Examples Must Compile

All examples in `examples/` are part of the build contract:

```bash
cargo build --examples --workspace
```

Add this to CI. A broken example is a broken API. If an example is a
work-in-progress, either keep it out of the tree or gate it behind a feature
flag and exclude it from the default CI run.

## Feature Flags for Dev Tools

Define a `dev-tools` feature that enables optional debug and profiling
dependencies. This feature must never be enabled in release builds shipped
to players.

```toml
# crates/game-core/Cargo.toml
[features]
default = []
dev-tools = ["dep:egui", "dep:puffin"]
profiling = ["dep:puffin"]

[dependencies]
egui = { workspace = true, optional = true }
puffin = { workspace = true, optional = true }
```

Gate the code:

```rust
#[cfg(feature = "dev-tools")]
fn draw_debug_overlay(ctx: &egui::Context) { ... }
```

The root feature in `game-app` re-exports:

```toml
[features]
default = []
dev-tools = [
  "game-core/dev-tools",
  "game-render/dev-tools",
]
```

Run CI without dev-tools to verify the default build is clean. Run a second
pass with `--features dev-tools` to verify the overlay code compiles.

## Error Handling Strategy

Apply the right tool at the right boundary:

**`game-app` (binary crate):** Use `anyhow::Result`. Top-level main and event
loop code is the correct place for context-enriched error propagation. It is
not appropriate in library crates.

```rust
// game-app/src/main.rs
fn main() -> anyhow::Result<()> { ... }
```

**Library crates (`game-core`, `game-render`, etc.):** Define explicit typed
errors using `thiserror`. Callers need to match on variants; `anyhow` erases
that information.

```rust
// game-assets/src/lib.rs
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("asset not found: {0}")]
    NotFound(String),
    #[error("decode failed: {0}")]
    Decode(#[from] std::io::Error),
}
```

**Renderer errors:** Keep `wgpu::Error` and surface errors inside
`game-render`. Do not propagate raw wgpu types into `game-core` or `game-app`.
Wrap them:

```rust
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("device lost")]
    DeviceLost,
    #[error("surface error: {0:?}")]
    Surface(wgpu::SurfaceError),
}
```

## Avoid Global Mutable State

Do not use `static Mutex<T>` or `lazy_static` to share game state. Instead:

- Pass resources explicitly through function parameters or ECS resource types.
- In a Bevy project, use `Res<T>` / `ResMut<T>`.
- In a custom ECS, store world resources in the `World` struct.

Acceptable globals: compile-time constants, read-only lookup tables, and
logging/tracing subscribers (initialized once in `main`).

## Platform-Specific Code

Use `#[cfg(target_os = ...)]` for small isolated cases:

```rust
#[cfg(target_os = "macos")]
fn configure_metal_layer(window: &winit::window::Window) { ... }
```

For anything larger than a few lines, move the code into a platform subcrate
or use a `cfg`-gated module file:

```
crates/game-platform/src/
  lib.rs
  windows.rs   // #[cfg(target_os = "windows")]
  macos.rs     // #[cfg(target_os = "macos")]
  linux.rs     // #[cfg(target_os = "linux")]
```

Never spread `#[cfg(target_os = ...)]` throughout `game-core`. Core logic must
be platform-agnostic.

## Minimum CI Pipeline

A CI configuration must include at least these steps, run on every PR and push
to main:

```yaml
steps:
  - name: fmt
    run: cargo fmt --all -- --check

  - name: clippy
    run: cargo clippy --workspace --all-targets --all-features -- -D warnings

  - name: test
    run: cargo nextest run --workspace

  - name: build examples
    run: cargo build --examples --workspace

  - name: clippy dev-tools
    run: cargo clippy --workspace --all-targets --features dev-tools -- -D warnings
```

Run on at least one Linux target. Add macOS and Windows targets when the
project targets those platforms. Use the same Rust toolchain version (pin with
`rust-toolchain.toml`) to prevent CI surprises from toolchain updates.
