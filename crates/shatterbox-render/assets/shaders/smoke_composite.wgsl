// Depth-aware upsample of the half-res ray marched smoke volume.
//
// Four half-resolution taps are weighted by how closely their scene depth
// matches the full-resolution pixel, so smoke behind a wall cannot bleed over
// its silhouette. The output is premultiplied colour with alpha = 1 - T and is
// blended with One / OneMinusSrcAlpha over the lit HDR image.

@group(0) @binding(0) var depth_tex: texture_2d<f32>;
@group(0) @binding(1) var smoke_tex: texture_2d<f32>;
@group(0) @binding(2) var smoke_sampler: sampler;

struct CompositeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_composite(@builtin(vertex_index) index: u32) -> CompositeOut {
    var out: CompositeOut;
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = positions[index];
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

@fragment
fn fs_composite(in: CompositeOut) -> @location(0) vec4<f32> {
    let depth_dims = vec2<i32>(textureDimensions(depth_tex));
    let smoke_dims = vec2<i32>(textureDimensions(smoke_tex));
    let full = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), depth_dims - 1);
    let half = clamp(full / 2, vec2<i32>(0), smoke_dims - 1);
    let center_depth = textureLoad(depth_tex, full, 0).r;

    var colour = vec3<f32>(0.0);
    var alpha = 0.0;
    var weight_sum = 0.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let tap = clamp(half + vec2<i32>(dx, dy) - vec2<i32>(1, 1), vec2<i32>(0), smoke_dims - 1);
            let tap_full = clamp(tap * 2, vec2<i32>(0), depth_dims - 1);
            let tap_depth = textureLoad(depth_tex, tap_full, 0).r;
            var weight = 1.0;
            if (center_depth > 0.0 && tap_depth > 0.0) {
                weight = 1.0 / (0.05 + abs(tap_depth - center_depth) * 0.35);
            } else if ((center_depth > 0.0) != (tap_depth > 0.0)) {
                weight = 0.05;
            }
            let sample_value = textureLoad(smoke_tex, tap, 0);
            colour += sample_value.rgb * weight;
            alpha += (1.0 - sample_value.a) * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum < 1.0e-4) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(colour / weight_sum, clamp(alpha / weight_sum, 0.0, 1.0));
}
