// FXAA 3.11 (simplified) at *internal* resolution, before the upscaler.
//
// The internal image has no MSAA and the upscaler only resamples it, so a one
// internal-pixel staircase becomes a 1.5 output-pixel staircase. This pass
// detects luma edges and blurs *along* them (never across), which removes the
// staircase while keeping a brick edge a brick edge instead of smearing the
// whole wall.
//
// This is the core of Timothy Lottes' FXAA 3.11 without the SearchTex/AreaTex
// lookup tables:
//   1. contrast test on the 4-neighbourhood (cheap early-out on flat sky/wall),
//   2. edge-orientation test choosing horizontal or vertical blur,
//   3. a 2-texel blur along the edge with a distance falloff, scaled by how
//      *isolated* the edge is: a smooth silhouette (a wall against the sky)
//      gets the full weight, high-frequency surface detail (the brick pattern)
//      gets almost none, so the wall keeps its texture.
// It is a purely spatial filter: no history, no jitter, so it composes with
// TAAU/FSR and with the debris reactive mask without adding ghosting.

const CONTRAST_THRESHOLD: f32 = 0.020;
const CONTRAST_MIN: f32 = 0.005;
/// Weight of the blurred result against the untouched centre.
const SUBPIXEL_BLEND: f32 = 0.80;
const BLUR_TAPS: i32 = 3;

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba16float, write>;

/// Perceptual luma. HDR values go through log2 so a 40.0 specular highlight and
/// a 1.0 lit brick get comparable edge weights (a linear test ignored edges in
/// the bright half of the frame and over-sharpened the dark half).
fn luma(color: vec3<f32>) -> f32 {
    let y = dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.299, 0.587, 0.114));
    return log2(1.0 + y);
}

fn fetch(coord: vec2<i32>, limit: vec2<i32>) -> vec3<f32> {
    return textureLoad(src_tex, clamp(coord, vec2<i32>(0), limit), 0).rgb;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(src_tex));
    let pixel = vec2<i32>(gid.xy);
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    let limit = dims - vec2<i32>(1);
    let centre = fetch(pixel, limit);
    let west = fetch(pixel + vec2<i32>(-1, 0), limit);
    let east = fetch(pixel + vec2<i32>(1, 0), limit);
    let north = fetch(pixel + vec2<i32>(0, -1), limit);
    let south = fetch(pixel + vec2<i32>(0, 1), limit);

    let l_m = luma(centre);
    let l_w = luma(west);
    let l_e = luma(east);
    let l_n = luma(north);
    let l_s = luma(south);
    let l_min = min(min(min(l_w, l_e), min(l_n, l_s)), l_m);
    let l_max = max(max(max(l_w, l_e), max(l_n, l_s)), l_m);
    let contrast = l_max - l_min;
    // Flat region (sky, a single lit brick face): copy through untouched.
    if (contrast < max(CONTRAST_MIN, l_max * CONTRAST_THRESHOLD)) {
        textureStore(dst_tex, pixel, vec4<f32>(centre, 1.0));
        return;
    }

    // Which way does the edge run? Compare how much the luma changes across the
    // vertical axis versus across the horizontal one.
    let gradient_h = abs(l_n + l_s - 2.0 * l_m) + 0.5 * abs(l_w - l_e);
    let gradient_v = abs(l_w + l_e - 2.0 * l_m) + 0.5 * abs(l_n - l_s);
    let horizontal_edge = gradient_h >= gradient_v;
    // An edge that runs horizontally (a wall/floor boundary) must be blurred
    // left-right, so the tap axis is X for a horizontal edge.
    let step = select(vec2<i32>(0, 1), vec2<i32>(1, 0), horizontal_edge);

    // Isolation: for a staircase silhouette the luma is almost constant *along*
    // the edge, so the smaller gradient stays far below the larger one. A brick
    // boundary has structure in both directions, the ratio approaches 1 and the
    // blur is almost switched off - that is what keeps the wall from smearing.
    let detail_along = min(gradient_h, gradient_v);
    let isolation = clamp(1.0 - detail_along / max(max(gradient_h, gradient_v), 1e-4), 0.0, 1.0);
    let blend = SUBPIXEL_BLEND * pow(isolation, 0.7);

    var sum = centre;
    var weight_sum = 1.0;
    for (var i = 1; i <= BLUR_TAPS; i = i + 1) {
        let offset = step * i;
        let side_a = fetch(pixel + offset, limit);
        let side_b = fetch(pixel - offset, limit);
        // Falloff keeps the outer taps from softening a two-pixel-wide detail.
        let weight = 1.0 / (1.0 + f32(i) * f32(i) * 0.9);
        sum = sum + (side_a + side_b) * weight;
        weight_sum = weight_sum + 2.0 * weight;
    }
    let blurred = sum / weight_sum;
    let result = mix(centre, blurred, blend);
    textureStore(dst_tex, pixel, vec4<f32>(result, 1.0));
}
