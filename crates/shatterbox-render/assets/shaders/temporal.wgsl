// Phase 2 temporal upscale (TAAU).
//
// The scene is rendered at `internal` resolution with a Halton sub-pixel
// jitter, lit into an HDR target, then reprojected and accumulated at output
// resolution. Depth+world-space reprojection makes it work for a free camera
// without motion vectors, and the 3x3 neighbourhood clamp removes ghosting
// after destruction (reactive masking arrives with Phase 3).

struct Frame {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    // x = effect bitmask (1 shadows, 2 ssao, 4 ssgi, 8 vct, 16 taa).
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var current_color: texture_2d<f32>;
@group(0) @binding(2) var current_depth: texture_2d<f32>;
@group(0) @binding(3) var history_color: texture_2d<f32>;
@group(0) @binding(4) var history_depth: texture_2d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_depth: texture_storage_2d<r32float, write>;

// Fraction of the new sample kept per frame. Lower = more temporal averaging of
// the shadow grain and voxel edges; the reactive rejection in `cs_main` drops
// history where the scene actually changed, so this can be aggressive.
const HISTORY_BLEND: f32 = 0.07;

fn world_from_pixel(pixel: vec2<i32>, depth: f32) -> vec3<f32> {
    let internal = frame.internal_size.xy;
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / internal;
    var ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    ndc = ndc + frame.params.zw * 2.0 / internal;
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let rd = normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
    return frame.camera_pos.xyz + rd * depth;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_color);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let internal = frame.internal_size.xy;
    let internal_ivec = vec2<i32>(internal);

    // The G-buffer was rendered with a jitter, so its content is shifted by
    // `jitter` internal pixels relative to the nominal grid.
    let sample_uv = uv - frame.params.zw / internal;
    let clamped_uv = clamp(sample_uv, vec2<f32>(0.5) / internal, vec2<f32>(1.0) - vec2<f32>(0.5) / internal);
    let current = textureSampleLevel(current_color, linear_sampler, clamped_uv, 0.0).rgb;

    let internal_pixel = clamp(vec2<i32>(sample_uv * internal), vec2<i32>(0), internal_ivec - 1);
    let depth = textureLoad(current_depth, internal_pixel, 0).r;

    if (depth < 0.0) {
        // Sky: smooth gradient, no history needed.
        textureStore(out_color, pixel, vec4<f32>(current, 1.0));
        textureStore(out_depth, pixel, vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    // 3x3 neighbourhood in the current frame: mean/variance clipping is what
    // keeps jittered samples from smearing while still rejecting ghosting.
    var sum = vec3<f32>(0.0);
    var sum_sq = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let p = clamp(internal_pixel + vec2<i32>(dx, dy), vec2<i32>(0), internal_ivec - 1);
            let c = textureLoad(current_color, p, 0).rgb;
            sum = sum + c;
            sum_sq = sum_sq + c * c;
        }
    }
    let mean = sum / 9.0;
    let variance = max(sum_sq / 9.0 - mean * mean, vec3<f32>(0.0));
    let sigma = sqrt(variance);
    let neighbourhood_min = max(mean - sigma * 1.35, vec3<f32>(0.0));
    let neighbourhood_max = mean + sigma * 1.35;

    let world = world_from_pixel(internal_pixel, depth);
    var prev_uv = uv;
    var history_valid = frame.params2.y < 0.5;
    if (history_valid) {
        let clip = frame.prev_view_proj * vec4<f32>(world, 1.0);
        if (clip.w > 1e-4) {
            prev_uv = vec2<f32>(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        } else {
            history_valid = false;
        }
    }
    if (history_valid && (any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)))) {
        history_valid = false;
    }

    if (history_valid) {
        let history_pixel = clamp(
            vec2<i32>(prev_uv * vec2<f32>(dims)),
            vec2<i32>(0),
            vec2<i32>(dims) - 1,
        );
        let history_depth_value = textureLoad(history_depth, history_pixel, 0).r;
        let expected = length(world - frame.prev_camera_pos.xyz);
        if (abs(history_depth_value - expected) > 0.12 + expected * 0.02) {
            history_valid = false;
        }
    }

    var color = current;
    let taa_enabled = (u32(frame.params3.x) & 16u) != 0u;
    if (history_valid && taa_enabled) {
        var history = textureSampleLevel(history_color, linear_sampler, prev_uv, 0.0).rgb;
        let raw_history = history;
        history = clamp(history, neighbourhood_min, neighbourhood_max);
        // Reactive rejection: destroyed geometry, falling chips and smoke are
        // not in the depth/normal history, so their colour lands far outside the
        // current neighbourhood. Clamping alone leaves a faint ghost; when the
        // history had to move that much, drop it entirely for this pixel.
        let rejected = length(raw_history - history) > 0.12 + length(sigma) * 2.0;
        // More history when the pixel is stable (low variance), less on edges
        // and after destruction, which is where TAA normally ghosts.
        let stability = 1.0 - clamp(length(sigma) * 2.0, 0.0, 0.55);
        var blend = mix(0.22, HISTORY_BLEND, stability);
        if (rejected) {
            blend = 1.0;
        }
        color = mix(history, current, blend);
    }

    textureStore(out_color, pixel, vec4<f32>(max(color, vec3<f32>(0.0)), 1.0));
    textureStore(out_depth, pixel, vec4<f32>(depth, 0.0, 0.0, 0.0));
}
