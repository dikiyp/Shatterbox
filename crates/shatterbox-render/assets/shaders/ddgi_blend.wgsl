// DDGI blend pass: ray radiance -> irradiance, with temporal hysteresis.
//
// For every texel of the refreshed slice this pass convolves the 8x8 ray cloud
// written by `ddgi_update.wgsl` with a cosine lobe around the texel direction:
//
//   E(d0) = sum_i L_i * max(dot(d_i, d0), 0) * (4*pi / 64)
//
// That is the irradiance a surface whose normal points along `d0` receives.
// `lighting.wgsl` multiplies it by albedo / pi, so the units match the old
// colourless `probe.wgsl` + cone trace path.
//
// Depth moments store the cosine-weighted mean and mean-square of the ray hit
// distances (normalised by MAX_DISTANCE). `gi.wgsl` uses them for a Chebyshev
// visibility test, which is what stops light from bleeding through a wall into
// the room behind it.
//
// Every texel keeps `params.hysteresis` of the previous value and replaces
// `1 - hysteresis` with the fresh convolution. `params.reset != 0` skips the
// history for one full update cycle after an allocation or when the effect is
// toggled on, so the atlas does not fade up from black.

struct DdgiParams {
    slice: u32,
    hysteresis: f32,
    spacing: f32,
    reset: u32,
    tile: u32,
    samples: u32,
    slices: u32,
    priority: u32,
    // Priority centre in probe-grid coordinates (probe index space).
    priority_center: vec3<f32>,
    priority_radius: f32,
    priority_hysteresis: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> params: DdgiParams;
@group(0) @binding(1) var trace_tex: texture_2d<f32>;
@group(0) @binding(2) var prev_irradiance: texture_2d<f32>;
@group(0) @binding(3) var prev_moments: texture_2d<f32>;
@group(0) @binding(4) var out_irradiance: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var out_moments: texture_storage_2d<rgba16float, write>;

// Must match DDGI_GRID / DDGI_TILE in `renderer.rs` and `ddgi_update.wgsl`.
const GRID_X: u32 = 64u;
const GRID_Y: u32 = 16u;
const GRID_Z: u32 = 64u;
const TILE: u32 = 8u;
const PI: f32 = 3.14159265;
const MAX_DISTANCE: f32 = 24.0;

fn octahedral_direction(uv: vec2<f32>) -> vec3<f32> {
    var d = vec3<f32>(uv.x, uv.y, 1.0 - abs(uv.x) - abs(uv.y));
    if (d.z < 0.0) {
        d = vec3<f32>((1.0 - abs(d.yx)) * sign(d.xy), d.z);
    }
    return normalize(d);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_extent = vec2<u32>(GRID_X * TILE, GRID_Y * TILE);
    if (gid.x >= tile_extent.x || gid.y >= tile_extent.y) {
        return;
    }
    let tile = clamp(params.tile, 1u, TILE);
    let z = min(params.slice, GRID_Z - 1u);
    let probe = vec3<u32>(gid.x / TILE, gid.y / TILE, z);
    let texel = vec2<u32>(gid.x % TILE, gid.y % TILE);
    if (texel.x >= tile || texel.y >= tile) {
        return;
    }
    let atlas = vec2<i32>(i32(gid.x), i32(z * tile_extent.y + gid.y));

    let center_uv = (vec2<f32>(texel) + vec2<f32>(0.5)) / f32(tile) * 2.0 - vec2<f32>(1.0);
    let center = octahedral_direction(center_uv);

    var irradiance = vec3<f32>(0.0);
    var mean_distance = 0.0;
    var mean_square = 0.0;
    var cosine_sum = 0.0;
    let solid_angle = 4.0 * PI / f32(tile * tile);
    for (var y = 0u; y < tile; y = y + 1u) {
        for (var x = 0u; x < tile; x = x + 1u) {
            let ray_atlas = vec2<i32>(
                i32(probe.x * TILE + x),
                i32(probe.z * GRID_Y * TILE + probe.y * TILE + y),
            );
            let sample = textureLoad(trace_tex, ray_atlas, 0);
            let uv = (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) / f32(tile) * 2.0
                - vec2<f32>(1.0);
            let direction = octahedral_direction(uv);
            let cosine = max(dot(direction, center), 0.0);
            if (cosine <= 0.0) {
                continue;
            }
            irradiance = irradiance + sample.rgb * cosine * solid_angle;
            let distance = clamp(sample.a, 0.0, 1.0) * MAX_DISTANCE;
            mean_distance = mean_distance + distance * cosine;
            mean_square = mean_square + distance * distance * cosine;
            cosine_sum = cosine_sum + cosine;
        }
    }
    let inverse = 1.0 / max(cosine_sum, 1e-4);
    mean_distance = mean_distance * inverse;
    mean_square = mean_square * inverse;
    let normalized_mean = mean_distance / MAX_DISTANCE;
    let variance = max(mean_square / (MAX_DISTANCE * MAX_DISTANCE) - normalized_mean * normalized_mean, 0.0);

    // A single probe can catch a firefly (a ray that slipped through a crack or
    // hit an emissive texel); clamping the convolution and smoothing it with the
    // four probe neighbours at the same texel removes the blocky bright/dark
    // cells that made the room look tiled. Only at full tiles: a low preset has
    // too few texels for a spatial pass to average honestly.
    var smoothed = irradiance;
    if (params.tile >= 8u) {
        let step = array<vec2<i32>, 4>(
            vec2<i32>(1, 0),
            vec2<i32>(-1, 0),
            vec2<i32>(0, 1),
            vec2<i32>(0, -1),
        );
        var sum = irradiance;
        var count = 1.0;
        for (var i = 0; i < 4; i = i + 1) {
            let nx = i32(probe.x) + step[i].x;
            let ny = i32(probe.y) + step[i].y;
            if (nx < 0 || nx >= i32(GRID_X) || ny < 0 || ny >= i32(GRID_Y)) {
                continue;
            }
            let neighbour_atlas = vec2<i32>(
                nx * i32(TILE) + i32(texel.x),
                i32(z) * i32(tile_extent.y) + ny * i32(TILE) + i32(texel.y),
            );
            sum = sum + textureLoad(prev_irradiance, neighbour_atlas, 0).rgb;
            count = count + 1.0;
        }
        smoothed = mix(irradiance, sum / count, 0.35);
    }
    let fresh_irradiance = vec4<f32>(clamp(smoothed, vec3<f32>(0.0), vec3<f32>(16.0)), 1.0);
    let fresh_moments = vec4<f32>(normalized_mean, variance, 0.0, normalized_mean);

    // Per-probe hysteresis: probes around fresh destruction fade in faster.
    var hysteresis = params.hysteresis;
    if (params.priority != 0u) {
        let distance = length(vec3<f32>(probe) - params.priority_center) * params.spacing;
        if (distance < params.priority_radius) {
            hysteresis = params.priority_hysteresis;
        }
    }
    let blend = select(
        clamp(1.0 - hysteresis, 0.0, 1.0),
        1.0,
        params.reset != 0u,
    );
    let previous_irradiance = textureLoad(prev_irradiance, atlas, 0);
    let previous_moments = textureLoad(prev_moments, atlas, 0);
    textureStore(out_irradiance, atlas, mix(previous_irradiance, fresh_irradiance, blend));
    textureStore(out_moments, atlas, mix(previous_moments, fresh_moments, blend));
}
