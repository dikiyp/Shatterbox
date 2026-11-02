// Phase 2 bloom: quarter-resolution prefilter + separable Gaussian blur.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba16float, write>;

// Auto exposure state. `sum`/`count` are cleared by `cs_exposure` after the
// prefilter has accumulated one quarter-res log-luminance histogram for the
// frame. `adapted` is the only value that persists between frames; the rest is
// written by the CPU every frame and is deliberately kept out of FrameUniform
// so the large shared struct does not change layout.
struct ExposureState {
    sum: atomic<u32>,
    count: atomic<u32>,
    adapted: f32,
    manual: f32,
    auto_on: f32,
    speed: f32,
    bias: f32,
    dt: f32,
    target_luma: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(2) var<storage, read_write> exposure_state: ExposureState;

// Plain read-only view for post.wgsl (fragment shaders cannot bind atomics).
struct ExposureValue {
    adapted: f32,
    auto_on: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(3) var<storage, read_write> exposure_value: ExposureValue;

const THRESHOLD: f32 = 1.0;
const KNEE: f32 = 0.55;
const BLOOM_DOWNSCALE: i32 = 4;
const LOG_LUMA_MIN: f32 = -12.0;
const LOG_LUMA_RANGE: f32 = 24.0;
// 8 bits of fractional EV. The quarter-res sample count at 1440p is ~230k, so
// 24 EV * 256 * 230k ≈ 1.4e9: safely inside u32.
const LOG_LUMA_FIXED: f32 = 256.0;

fn prefilter(color: vec3<f32>) -> vec3<f32> {
    let brightness = max(color.r, max(color.g, color.b));
    let soft = clamp(brightness - THRESHOLD + KNEE, 0.0, 2.0 * KNEE);
    let contribution = soft * soft / (4.0 * KNEE + 1e-4);
    let weight = max(soft, contribution) / max(brightness, 1e-4);
    return color * weight * step(THRESHOLD - KNEE, brightness);
}

@compute @workgroup_size(8, 8, 1)
fn cs_prefilter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(destination);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let source_dims = vec2<i32>(textureDimensions(source));
    let base = vec2<i32>(gid.xy) * BLOOM_DOWNSCALE;
    var color = vec3<f32>(0.0);
    for (var y = 0; y < 2; y = y + 1) {
        for (var x = 0; x < 2; x = x + 1) {
            let p = clamp(base + vec2<i32>(x * 2, y * 2), vec2<i32>(0), source_dims - 1);
            color = color + textureLoad(source, p, 0).rgb;
        }
    }
    color = color * 0.25;
    // One log-luminance sample per quarter-res texel. Fixed point keeps the
    // accumulation exact and lets the resolve pass do the expensive exp2.
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let log_luma = clamp(log2(max(luma, 1.0e-4)) - LOG_LUMA_MIN, 0.0, LOG_LUMA_RANGE);
    atomicAdd(&exposure_state.sum, u32(log_luma * LOG_LUMA_FIXED));
    atomicAdd(&exposure_state.count, 1u);
    textureStore(destination, vec2<i32>(gid.xy), vec4<f32>(prefilter(color), 1.0));
}

/// Resolve the frame average and adapt the exposure. Runs once per frame after
/// the prefilter. The adaptation is frame-rate independent: `dt` comes from the
/// CPU, and the exponential curve is the same at 30, 60 or 144 fps.
@compute @workgroup_size(1, 1, 1)
fn cs_exposure() {
    let count = atomicLoad(&exposure_state.count);
    let sum = atomicLoad(&exposure_state.sum);
    atomicStore(&exposure_state.count, 0u);
    atomicStore(&exposure_state.sum, 0u);

    var adapted = exposure_state.adapted;
    if (count > 0u) {
        let average_log = f32(sum) / f32(count) / LOG_LUMA_FIXED + LOG_LUMA_MIN;
        let average = exp2(average_log);
        let target_luma = max(exposure_state.target_luma, 1.0e-3);
        // Cap the adapted exposure. A dark interior asks for a 3-4x lift, but
        // letting the meter have it turns a lamp-lit room into a flat beige
        // daylight box. 1.25 keeps night dark while still lifting the shadows.
        let desired = clamp(
            target_luma / max(average, 1.0e-4) * exp2(exposure_state.bias),
            0.2,
            1.25,
        );
        let dt = clamp(exposure_state.dt, 0.0, 0.25);
        let blend = 1.0 - exp(-max(exposure_state.speed, 0.01) * dt);
        adapted = mix(adapted, desired, blend);
    }
    exposure_state.adapted = adapted;
    exposure_value.adapted = adapted;
    exposure_value.auto_on = exposure_state.auto_on;
}

const WEIGHTS: array<f32, 9> = array<f32, 9>(
    0.016216, 0.054054, 0.121621, 0.194594, 0.227027,
    0.194594, 0.121621, 0.054054, 0.016216,
);

fn blur(gid: vec3<u32>, axis: vec2<i32>) {
    let dims = textureDimensions(destination);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    var color = vec3<f32>(0.0);
    for (var i = -4; i <= 4; i = i + 1) {
        let p = clamp(pixel + axis * i, vec2<i32>(0), vec2<i32>(dims) - 1);
        color = color + textureLoad(source, p, 0).rgb * WEIGHTS[i + 4];
    }
    textureStore(destination, pixel, vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur(gid, vec2<i32>(1, 0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    blur(gid, vec2<i32>(0, 1));
}
