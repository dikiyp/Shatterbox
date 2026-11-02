// FSR 1 style upscale: FidelityFX Super Resolution 1 is a purely spatial
// (EASU upscale + RCAS sharpen) filter. We ship our own WGSL port of that
// algorithm family instead of linking the AMD runtime: the shader is
// edge-adaptive on the luma channel, keeps the TAA history blend that the
// renderer already had, and finishes with a contrast-adaptive sharpen pass.
//
// It replaces `temporal.wgsl` 1:1 - same bindings, same outputs - so the rest
// of the frame graph (bloom, post, UI) is untouched. Select it with
// `SHATTERBOX_UPSCALER=fsr` or the overlay toggle.
//
// Why not bit-exact FSR 1:
//   * AMD's EASU is a 12-tap quartic fit; we use a separable Catmull-Rom base
//     with an edge-direction bias, which is 8 taps and measurably cheaper at
//     1440p while keeping the same sharpness on voxel silhouettes.
//   * RCAS is reproduced in its contrast-adaptive form: sharpen only where the
//     local contrast has headroom, never ringing past the 4-neighbour range.

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
    light_axis: array<vec4<f32>, 12>,
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
    debug_hit: vec4<f32>,
    // x = 1 enables the debris reactive mask.
    params4: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var current_color: texture_2d<f32>;
@group(0) @binding(2) var current_depth: texture_2d<f32>;
@group(0) @binding(3) var history_color: texture_2d<f32>;
@group(0) @binding(4) var history_depth: texture_2d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(8) var material_tex: texture_2d<f32>;

const HISTORY_BLEND: f32 = 0.11;
// 0.18 keeps voxel edges crisp without haloing the material detail texture.
const RCAS_STRENGTH: f32 = 0.18;

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

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

// Catmull-Rom basis: w.x is applied to tap -1, w.w to tap +2.
fn catmull_rom(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

// Samples the low-resolution colour with two taps per row plus an edge bias:
// the horizontal pair is collapsed with bilinear hardware taps, so the whole
// kernel costs eight fetches instead of sixteen.
fn easu_upscale(uv: vec2<f32>, internal_pixel: vec2<f32>) -> vec3<f32> {
    let internal = frame.internal_size.xy;
    let half_texel = vec2<f32>(0.5) / internal;
    let pos = internal_pixel - vec2<f32>(0.5);
    let base = floor(pos);
    let frac = pos - base;
    let w = catmull_rom(frac.x);

    // Edge direction: a strong luma gradient across the edge means the lobes
    // along the edge carry the detail, so bias the kernel towards them.
    let left = luma(textureSampleLevel(current_color, linear_sampler, clamp(uv - vec2<f32>(half_texel.x, 0.0), half_texel, vec2<f32>(1.0) - half_texel), 0.0).rgb);
    let right = luma(textureSampleLevel(current_color, linear_sampler, clamp(uv + vec2<f32>(half_texel.x, 0.0), half_texel, vec2<f32>(1.0) - half_texel), 0.0).rgb);
    let down = luma(textureSampleLevel(current_color, linear_sampler, clamp(uv - vec2<f32>(0.0, half_texel.y), half_texel, vec2<f32>(1.0) - half_texel), 0.0).rgb);
    let up = luma(textureSampleLevel(current_color, linear_sampler, clamp(uv + vec2<f32>(0.0, half_texel.y), half_texel, vec2<f32>(1.0) - half_texel), 0.0).rgb);
    let gx = right - left;
    let gy = up - down;
    let edge = clamp(sqrt(gx * gx + gy * gy) * 4.0, 0.0, 1.0);

    var total = vec3<f32>(0.0);
    var weight_sum = 0.0;
    // Colour range of the eight taps the kernel actually reads. Catmull-Rom
    // lobes can overshoot; on a silhouette that shows up as a bright fringe on
    // the lit side and a dark seam on the shadowed side, which is the "black
    // staircase" the eye locks onto. Clamping to the fetched range keeps the
    // edge crisp without inventing colours that are not in the source.
    var tap_min = vec3<f32>(1.0e20);
    var tap_max = vec3<f32>(-1.0e20);
    let pair_a = w.x + w.y;
    let pair_b = w.z + w.w;
    // A Catmull-Rom pair can cancel out at frac ~= 1; drop that pair instead of
    // dividing by ~0. One of the two pairs always carries most of the weight.
    let pair_a_valid = select(pair_a, 0.0, abs(pair_a) < 1e-3);
    let pair_b_valid = select(pair_b, 0.0, abs(pair_b) < 1e-3);
    let x_a = base.x - 1.0 + w.y * select(1.0 / pair_a, 0.0, abs(pair_a) < 1e-3);
    let x_b = base.x + 1.0 + w.w * select(1.0 / pair_b, 0.0, abs(pair_b) < 1e-3);
    // Both sample columns map back to normalised uv once, then the four rows
    // only shift `y`.
    let uv_a = vec2<f32>((x_a + 0.5) / internal.x, 0.0);
    let uv_b = vec2<f32>((x_b + 0.5) / internal.x, 0.0);
    for (var row = 0; row < 4; row = row + 1) {
        let ry = base.y + f32(row) - 1.0;
        let wy = catmull_rom(frac.y)[row];
        let row_y = (ry + 0.5) / internal.y;
        let a = textureSampleLevel(
            current_color,
            linear_sampler,
            clamp(vec2<f32>(uv_a.x, row_y), half_texel, vec2<f32>(1.0) - half_texel),
            0.0,
        ).rgb;
        let b = textureSampleLevel(
            current_color,
            linear_sampler,
            clamp(vec2<f32>(uv_b.x, row_y), half_texel, vec2<f32>(1.0) - half_texel),
            0.0,
        ).rgb;
        // Horizontal pair sign is preserved by the pair sums, so the kernel can
        // still overshoot (that is what keeps a hard voxel edge hard).
        let row_color = a * pair_a_valid + b * pair_b_valid;
        total = total + row_color * wy;
        weight_sum = weight_sum + wy;
        tap_min = min(tap_min, min(a, b));
        tap_max = max(tap_max, max(a, b));
    }
    var filtered = total / max(weight_sum, 1e-5);

    // Bilinear fallback blended in when the local neighbourhood is flat: a
    // Catmull-Rom lobe on flat colour only adds noise.
    let bilinear = textureSampleLevel(
        current_color,
        linear_sampler,
        clamp(uv, half_texel, vec2<f32>(1.0) - half_texel),
        0.0,
    ).rgb;
    filtered = mix(bilinear, filtered, 0.35 + 0.65 * edge);
    // Light anti-aliasing before the upscale (`SHATTERBOX_LIGHT_AA=1`, effect bit
    // 128). Voxel silhouettes are rasterised at the internal resolution without
    // any sub-pixel jitter, so a near-horizontal edge comes out as a staircase
    // of 1-pixel steps that the upscale then magnifies. Averaging the two taps
    // along the detected edge direction rounds those steps off without touching
    // the edge itself: the gradient direction is preserved, only the run along
    // it is blurred.
    if ((u32(frame.params3.x) & 128u) != 0u && edge > 0.15) {
        let along = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), abs(gx) >= abs(gy));
        let offset = along * half_texel * 1.5;
        let side_a = textureSampleLevel(
            current_color,
            linear_sampler,
            clamp(uv - offset, half_texel, vec2<f32>(1.0) - half_texel),
            0.0,
        ).rgb;
        let side_b = textureSampleLevel(
            current_color,
            linear_sampler,
            clamp(uv + offset, half_texel, vec2<f32>(1.0) - half_texel),
            0.0,
        ).rgb;
        let smoothed = (side_a + side_b + bilinear * 2.0) * 0.25;
        filtered = mix(filtered, smoothed, 0.5);
    }
    // `SHATTERBOX_UPSCALE_CLAMP=0` restores the unclamped (overshooting) filter
    // for A/B; `params3.w` carries the switch.
    if (frame.params3.w > 0.5) {
        filtered = clamp(filtered, min(tap_min, bilinear), max(tap_max, bilinear));
    }
    return max(filtered, vec3<f32>(0.0));
}

// Contrast-adaptive sharpen in the spirit of RCAS: boost the 4-neighbour
// Laplacian, then clamp into the local range so edges get crisp without rings.
fn easu_guide(uv: vec2<f32>) -> vec3<f32> {
    let half_texel = vec2<f32>(0.5) / frame.internal_size.xy;
    return textureSampleLevel(
        current_color,
        linear_sampler,
        clamp(uv, half_texel, vec2<f32>(1.0) - half_texel),
        0.0,
    ).rgb;
}

fn rcas(color: vec3<f32>, north: vec3<f32>, south: vec3<f32>, east: vec3<f32>, west: vec3<f32>) -> vec3<f32> {
    let local_min = min(color, min(min(north, south), min(east, west)));
    let local_max = max(color, max(max(north, south), max(east, west)));
    let average = (north + south + east + west) * 0.25;
    let sharpened = color + (color - average) * RCAS_STRENGTH;
    return clamp(sharpened, local_min, local_max);
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

    // The G-buffer was rendered with a sub-pixel jitter, so its content is
    // shifted by `jitter` internal pixels relative to the nominal grid.
    let sample_uv = uv - frame.params.zw / internal;
    let source = clamp(sample_uv * internal, vec2<f32>(0.0), internal - vec2<f32>(1.0));

    let internal_pixel = clamp(vec2<i32>(source), vec2<i32>(0), internal_ivec - 1);
    let depth = textureLoad(current_depth, internal_pixel, 0).r;
    let current = easu_upscale(clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0)), source);

    if (depth < 0.0) {
        textureStore(out_color, pixel, vec4<f32>(current, 1.0));
        textureStore(out_depth, pixel, vec4<f32>(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    // 3x3 neighbourhood in the current frame (internal grid) for the history
    // clamp - same rejection the TAAU path uses, so ghosting behaves the same.
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
        if (abs(history_depth_value - expected) > 0.2 + expected * 0.03) {
            history_valid = false;
        }
    }

    var color = current;
    let taa_enabled = (u32(frame.params3.x) & 16u) != 0u;
    // Reactive mask: `gbuffer.wgsl` writes 1 in the fourth material channel for
    // pixels covered by debris. Those move several internal pixels per frame, so
    // reprojected history is stale and would leave a ghost trail behind the
    // piece; the current frame wins outright there.
    let reactive = textureLoad(material_tex, internal_pixel, 0).w;
    if (frame.params4.x > 0.5 && reactive > 0.5) {
        history_valid = false;
    }
    if (history_valid && taa_enabled) {
        var history = textureSampleLevel(history_color, linear_sampler, prev_uv, 0.0).rgb;
        history = clamp(history, neighbourhood_min, neighbourhood_max);
        let stability = 1.0 - clamp(length(sigma) * 2.0, 0.0, 0.55);
        let blend = mix(0.22, HISTORY_BLEND, stability);
        color = mix(history, current, blend);
    }

    // RCAS runs on the upscaled result. The cross taps come from the cheap
    // bilinear read of the same source: reading `out_color` back is illegal
    // (storage textures are write-only), and the bilinear guide is what the
    // sharpen only needs to reject ringing.
    let source_texel = vec2<f32>(1.0) / internal;
    let guide = clamp(sample_uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let north = easu_guide(guide + vec2<f32>(0.0, source_texel.y));
    let south = easu_guide(guide - vec2<f32>(0.0, source_texel.y));
    let east = easu_guide(guide + vec2<f32>(source_texel.x, 0.0));
    let west = easu_guide(guide - vec2<f32>(source_texel.x, 0.0));
    color = rcas(color, north, south, east, west);

    textureStore(out_color, pixel, vec4<f32>(max(color, vec3<f32>(0.0)), 1.0));
    textureStore(out_depth, pixel, vec4<f32>(depth, 0.0, 0.0, 0.0));
}
