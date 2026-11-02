// Phase 2 post: bloom composite, ACES tonemap, grade, vignette, dither.
//
// The swapchain format is non-sRGB, so gamma encoding happens here.

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
    // Declared even though post does not light anything: the offsets of
    // params4 must match the Rust FrameUniform exactly.
    light_axis: array<vec4<f32>, 12>,
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
    debug_hit: vec4<f32>,
    // x = debris reactive, y = fog height, z = post debug bits, w = look.
    params4: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;

struct ExposureValue {
    adapted: f32,
    auto_on: f32,
    _pad0: f32,
    _pad1: f32,
};
@group(0) @binding(3) var<storage, read> exposure_value: ExposureValue;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = positions[index];
    var out: VertexOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn aces_fitted(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(hdr_texture));
    let pixel = clamp(vec2<i32>(in.position.xy), vec2<i32>(0), dims - 1);
    var color = textureLoad(hdr_texture, pixel, 0).rgb;

    let bloom_dims = vec2<i32>(textureDimensions(bloom_texture));
    let bloom_pixel = clamp(pixel / 4, vec2<i32>(0), bloom_dims - 1);
    let bloom = textureLoad(bloom_texture, bloom_pixel, 0).rgb;

    // Exposure is either the manual slider (params2.w) or the adapted value
    // from the luminance pass. The latter is measured in linear HDR before
    // tonemapping, so changing the look does not change what the meter sees.
    var exposure = frame.params2.w;
    if (exposure_value.auto_on > 0.5) {
        exposure = exposure_value.adapted;
    }
    // Diagnostic bitmask (SHATTERBOX_POST_DEBUG / scenario `Post`):
    //   1 = no saturation+grade, 2 = no black lift, 4 = no pre-tonemap gain,
    //   8 = no ACES, 16 = no bloom, 32 = no vignette.
    let debug = u32(frame.params4.z);
    let look = u32(frame.params4.w);

    // The three looks share one display transform. They differ only in how much
    // colour they add *after* the tonemapper and how they split the warm/cool
    // ends; none of them adds an S-curve on top of ACES.
    var bloom_strength = 0.035;
    var saturation = 1.02;
    var shadow_tint = vec3<f32>(-0.018, -0.008, 0.026);
    var highlight_tint = vec3<f32>(0.030, 0.012, -0.022);
    var vignette_strength = 0.06;
    var gain = 1.0;
    if (look == 1u) {
        bloom_strength = 0.065;
        saturation = 1.10;
        shadow_tint = vec3<f32>(-0.038, -0.018, 0.062);
        highlight_tint = vec3<f32>(0.055, 0.025, -0.038);
        vignette_strength = 0.12;
    } else if (look >= 2u) {
        bloom_strength = 0.095;
        saturation = 1.16;
        shadow_tint = vec3<f32>(-0.016, 0.000, 0.030);
        highlight_tint = vec3<f32>(0.060, 0.025, -0.048);
        vignette_strength = 0.17;
        gain = 1.06;
    }

    if ((debug & 16u) == 0u) {
        color = color + bloom * bloom_strength;
    }
    color = color * exposure;
    if ((debug & 4u) == 0u) {
        color = color * gain;
    }

    // ACES is the single display transform. Its toe already crushes blacks
    // gently; the old stack then added gain, saturation *and* an S-curve on top,
    // which is what turned dim interiors into olive mush.
    if ((debug & 8u) == 0u) {
        color = aces_fitted(color);
    } else {
        color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    }

    // Grade after the tonemapper, weighted by luminance. Saturation is zero in
    // the shadows: dark pixels keep the material colour the lighting produced
    // instead of being pushed into neon by a global 1.22 multiplier.
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if ((debug & 1u) == 0u) {
        let saturation_weight = smoothstep(0.025, 0.45, luma);
        color = mix(vec3<f32>(luma), color, mix(1.0, saturation, saturation_weight));
        let shadow_weight = 1.0 - smoothstep(0.04, 0.30, luma);
        let highlight_weight = smoothstep(0.45, 0.85, luma);
        color = color * (vec3<f32>(1.0) + shadow_tint * shadow_weight + highlight_tint * highlight_weight);
    }
    if ((debug & 2u) == 0u) {
        // A tiny lift, not a contrast curve: it keeps the darkest readable
        // material from collapsing to the same black as the background.
        color = color * 0.998 + vec3<f32>(0.002);
    }

    if ((debug & 32u) == 0u) {
        let uv = in.uv;
        let centered = uv * 2.0 - 1.0;
        let vignette = 1.0 - dot(centered, centered) * vignette_strength;
        color = color * clamp(vignette, 0.0, 1.0);
    }
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Gamma encode + ordered-ish noise dither to hide sky banding.
    let dither = (hash12(vec2<f32>(in.position.xy)) - 0.5) / 255.0;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)) + dither;

    return vec4<f32>(color, 1.0);
}
