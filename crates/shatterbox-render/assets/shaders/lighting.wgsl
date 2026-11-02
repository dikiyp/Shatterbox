// Phase 2 deferred lighting: PBR sun + sky ambient + point lights + fog.
//
// Shadow visibility arrives in the G-buffer alpha, so this pass never touches
// voxel data. All reads use textureLoad because every pass runs at exactly the
// internal resolution.

struct Frame {
    inv_view_proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    prev_camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    // rgb = sun/moon radiance, w = normalized daylight factor 0..1
    sun_color: vec4<f32>,
    world_min: vec4<f32>,
    world_max: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    // x = effect bitmask (1 shadows, 2 ssao, 4 ssgi, 8 vct, 16 taa, 32 fog,
    // 64 ssr).
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
    // xyz = emitter axis, w = half extent. w > 0 is a capsule of length 2w,
    // w < 0 is a one-sided rectangle with half size -w and normal xyz, w == 0
    // is a plain point light.
    light_axis: array<vec4<f32>, 12>,
    // World-space AABB of the uploaded debris set (unused here, declared so the
    // field order stays identical to the Rust uniform).
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
    // xyz = CPU aim point, w = marker radius in metres (0 disables).
    debug_hit: vec4<f32>,
    // x = debris reactive mask, y = fog height, z = post/lighting debug bits,
    // w = look variant. Declared so the field offset matches the Rust uniform.
    params4: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var albedo_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var material_tex: texture_2d<f32>;
@group(0) @binding(4) var depth_tex: texture_2d<f32>;
@group(0) @binding(5) var out_hdr: texture_storage_2d<rgba16float, write>;
// Half-resolution GI from `gi.wgsl`: rgb = indirect diffuse, a = screen AO.
@group(0) @binding(6) var gi_tex: texture_2d<f32>;
// Half-resolution volumetric fog: rgb = inscatter, a = transmittance.
@group(0) @binding(7) var fog_tex: texture_2d<f32>;
// Half-resolution sun visibility from `shadow.wgsl` (r = visibility).
@group(0) @binding(8) var shadow_tex: texture_2d<f32>;
@group(0) @binding(9) var linear_sampler: sampler;
// Half-resolution local-light visibility from `light_shadow.wgsl`: bit i is set
// when light i reaches this texel. Uint textures cannot be filtered, so the 2x2
// gather below does the bilinear blend by hand.
@group(0) @binding(10) var light_shadow_tex: texture_2d<u32>;
// Half-resolution specular reflection traced in screen space by `gi.wgsl`.
@group(0) @binding(11) var reflection_tex: texture_2d<f32>;

const PI: f32 = 3.14159265;

/// Depth/normal-aware 2x2 upsample of a half-resolution buffer.
///
/// `pixel / 2` nearest reads turned AO, GI, fog and reflections into visible
/// 2x2 blocks; plain bilinear bleeds light across silhouettes. Every tap gets
/// its bilinear footprint weight times depth and normal agreement, sampled from
/// the full-resolution G-buffer at the tap's centre. If every tap disagrees
/// (thin geometry, grazing angles) it falls back to bilinear so silhouettes do
/// not show the half-resolution grid as dark stair steps.
fn upsample_half(
    tex: texture_2d<f32>,
    pixel: vec2<i32>,
    full_dims: vec2<i32>,
    center_depth: f32,
    center_normal: vec3<f32>,
    center_world: vec3<f32>,
) -> vec4<f32> {
    let texel_count = textureDimensions(tex);
    let half_dims = vec2<i32>(texel_count);
    let hi = half_dims - vec2<i32>(1);
    let pos = (vec2<f32>(pixel) + vec2<f32>(0.5)) * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(pos));
    let frac = pos - floor(pos);
    let scale = vec2<f32>(full_dims) / vec2<f32>(texel_count);
    let full_hi = full_dims - vec2<i32>(1);

    // Depth changes quickly across a surface seen at a grazing angle. A fixed
    // relative tolerance rejected those taps and fell back to a nearest lookup,
    // which is what made AO look like black 2x2 stairs on walls and edges.
    let view_dir = normalize(center_world - frame.camera_pos.xyz);
    let grazing = max(abs(dot(view_dir, center_normal)), 0.25);
    let tolerance = max(0.08, center_depth * 0.04) / grazing;

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    var bilinear_sum = vec4<f32>(0.0);
    var bilinear_sum_weight = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let ox = i & 1;
        let oy = i >> 1;
        let tap = clamp(base + vec2<i32>(ox, oy), vec2<i32>(0), hi);
        let wx = select(1.0 - frac.x, frac.x, ox == 1);
        let wy = select(1.0 - frac.y, frac.y, oy == 1);
        let w_bilinear = wx * wy;
        // Centre of this half-resolution texel in full-resolution pixels.
        let full_tap = clamp(
            vec2<i32>(floor((vec2<f32>(tap) + vec2<f32>(0.5)) * scale)),
            vec2<i32>(0),
            full_hi,
        );
        let tap_depth = textureLoad(depth_tex, full_tap, 0).r;
        let sample = textureLoad(tex, tap, 0);
        var w = 0.0;
        if (tap_depth >= 0.0) {
            let relative = abs(tap_depth - center_depth) / tolerance;
            let tap_normal = normalize(textureLoad(normal_tex, full_tap, 0).xyz);
            let w_normal = clamp(dot(tap_normal, center_normal) * 0.5 + 0.5, 0.0, 1.0);
            w = w_bilinear * clamp(1.0 - relative, 0.0, 1.0) * w_normal * w_normal;
        }
        sum = sum + sample * w;
        weight_sum = weight_sum + w;
        bilinear_sum = bilinear_sum + sample * w_bilinear;
        bilinear_sum_weight = bilinear_sum_weight + w_bilinear;
    }
    if (weight_sum < 1e-4) {
        // Bilinear rather than nearest: a hard nearest fallback at silhouettes
        // showed the half-resolution grid as dark stair steps.
        return bilinear_sum / max(bilinear_sum_weight, 1e-4);
    }
    return sum / weight_sum;
}

fn sky_hash21(p: vec2<f32>) -> f32 {
    let ip = vec2<i32>(floor(p));
    var h = u32(ip.x) * 1597334677u ^ u32(ip.y) * 3812015801u;
    h = h ^ (h >> 15u);
    h = h * 2246822519u;
    h = h ^ (h >> 13u);
    return f32(h & 0x00ffffffu) / 16777215.0;
}

fn sky_value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = sky_hash21(i);
    let b = sky_hash21(i + vec2<f32>(1.0, 0.0));
    let c = sky_hash21(i + vec2<f32>(0.0, 1.0));
    let d = sky_hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn sky_fbm(p: vec2<f32>) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var q = p;
    for (var octave = 0; octave < 3; octave = octave + 1) {
        sum = sum + amplitude * sky_value_noise(q);
        q = q * 2.07 + vec2<f32>(13.7, 7.3);
        amplitude = amplitude * 0.5;
    }
    return sum / 0.875;
}

fn sky_color(rd: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_dir.xyz);
    let day = clamp(frame.sun_color.w, 0.0, 1.0);
    let up = clamp(rd.y, 0.0, 1.0);
    // Night keeps a faint blue zenith instead of turning pure black, so the
    // arena is still readable by moon plus point lights.
    let zenith = mix(vec3<f32>(0.004, 0.008, 0.022), vec3<f32>(0.018, 0.065, 0.30), day);
    let horizon = mix(vec3<f32>(0.020, 0.032, 0.060), vec3<f32>(0.34, 0.52, 0.78), day);
    var color = mix(horizon, zenith, pow(up, 0.55));
    let sun_amount = max(dot(rd, sun), 0.0);
    // The sun/moon disk and halo take their radiance from the time-of-day sun,
    // so a low warm sun also tints the horizon without a second uniform.
    color = color + frame.sun_color.rgb * pow(sun_amount, 512.0) * 5.0;
    color = color + frame.sun_color.rgb * pow(sun_amount, 8.0) * 0.05;

    // Cloud layer: project the view ray onto a flat deck and drift it with
    // time. Two scales keep the shapes soft up close and busy near the horizon
    // where the projection stretches.
    let deck = rd.xz / max(rd.y, 0.045);
    let drift = vec2<f32>(frame.params.x * 0.0035, frame.params.x * 0.0012);
    var density = sky_fbm(deck * 0.55 + drift);
    density = density + 0.35 * sky_fbm(deck * 1.9 - drift * 1.7);
    let cover = smoothstep(0.64, 1.10, density) * smoothstep(0.02, 0.16, rd.y);
    let night_cloud = vec3<f32>(0.025, 0.030, 0.045);
    let lit = mix(vec3<f32>(0.52, 0.56, 0.66), vec3<f32>(1.35, 1.24, 1.08),
                  clamp(0.35 + 0.65 * sun_amount, 0.0, 1.0));
    color = mix(color, mix(night_cloud, lit, day), cover * 0.92);
    return color;
}

fn distribution_ggx(ndh: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-5);
}

fn geometry_smith(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let gv = ndv / (ndv * (1.0 - k) + k);
    let gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Representative point of an area emitter (Karis 2013). The returned `w` is 0
// when the shading point sits behind a one-sided rectangle, so the caller can
// skip the light entirely.
fn emitter_point(base: vec3<f32>, axis: vec3<f32>, extent: f32, world: vec3<f32>) -> vec4<f32> {
    if (extent > 0.0) {
        let dir = normalize(axis + vec3<f32>(0.0, 1e-6, 0.0));
        let along = clamp(dot(world - base, dir), -extent, extent);
        return vec4<f32>(base + dir * along, 1.0);
    }
    if (extent < 0.0) {
        let n = normalize(axis + vec3<f32>(0.0, 0.0, 1e-6));
        let half_size = -extent;
        let offset = world - base;
        let facing = dot(offset, n);
        if (facing <= 0.0) {
            return vec4<f32>(base, 0.0);
        }
        let planar = clamp(offset - n * facing, vec3<f32>(-half_size), vec3<f32>(half_size));
        return vec4<f32>(base + planar, 1.0);
    }
    return vec4<f32>(base, 1.0);
}

// --- Stylised material detail (moved out of the G-buffer) -------------------
// The primary pass only writes the palette colour; the per-voxel brick pattern,
// the 1 m block variation and the 2-3 m staining are cheap ALU work that fits
// into the deferred pass without the DDA kernel paying for the extra registers.
const MATERIAL_CONCRETE: u32 = 1u;
const MATERIAL_STONE: u32 = 2u;
const MATERIAL_BRICK: u32 = 3u;
const MATERIAL_WOOD: u32 = 4u;
const MATERIAL_METAL: u32 = 5u;
const MATERIAL_GLASS: u32 = 6u;
const MATERIAL_SAND: u32 = 7u;
const MATERIAL_GRASS: u32 = 8u;
const MATERIAL_EMISSIVE: u32 = 10u;
const MATERIAL_HAZARD: u32 = 11u;
const MATERIAL_WET_CONCRETE: u32 = 13u;
const MATERIAL_MIRROR: u32 = 22u;
const MATERIAL_GLASS_CLEAR: u32 = 23u;

fn detail_hash_u32(value: u32) -> u32 {
    var x = value;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn detail_hash01(a: i32, b: i32, seed: u32) -> f32 {
    var h = detail_hash_u32(bitcast<u32>(a) ^ 0x9e3779b9u);
    h = detail_hash_u32(h ^ bitcast<u32>(b));
    h = detail_hash_u32(h ^ (seed * 0x85ebca6bu));
    return f32(h & 0xffffu) / 65535.0;
}

fn detail_uv(cell: vec3<i32>, normal: vec3<f32>) -> vec2<i32> {
    if (abs(normal.y) > 0.5) {
        return vec2<i32>(cell.x, cell.z);
    }
    if (abs(normal.x) > 0.5) {
        return vec2<i32>(cell.z, cell.y);
    }
    return vec2<i32>(cell.x, cell.y);
}

fn material_detail(material: u32, cell: vec3<i32>, normal: vec3<f32>) -> f32 {
    let uv = detail_uv(cell, normal);
    let coarse = detail_hash01(uv.x / 3, uv.y / 3, material);

    // Neon tubes must stay uniform: any per-voxel brightness variation reads as
    // dirt on a light source. Palette indices 14-17 are the coloured neon set.
    if (material >= 14u && material <= 17u) {
        return 1.0;
    }
    if (material == MATERIAL_BRICK) {
        let row = uv.y >> 2;
        let shifted = uv.x + (row & 1) * 4;
        if ((uv.y & 3) == 0 || (shifted % 8) == 0) {
            return 0.76;
        }
        return 0.90 + 0.16 * coarse;
    }
    if (material == MATERIAL_CONCRETE || material == MATERIAL_STONE) {
        var value = 0.90 + 0.14 * coarse;
        if ((uv.x & 31) == 0 || (uv.y & 31) == 0) {
            value = value * 0.87;
        }
        value = value * (0.94 + 0.12 * detail_hash01(uv.x >> 5, uv.y >> 5, material + 31u));
        if (detail_hash01(uv.x, uv.y, material + 5u) < 0.08) {
            value = value * 0.86;
        }
        return value;
    }
    if (material == MATERIAL_WET_CONCRETE) {
        var wet = 0.88 + 0.10 * coarse;
        if ((uv.x & 31) == 0 || (uv.y & 31) == 0) {
            wet = wet * 0.82;
        }
        wet = wet * (0.92 + 0.10 * detail_hash01(uv.x >> 5, uv.y >> 5, 83u));
        return wet;
    }
    if (material == MATERIAL_WOOD) {
        let plank = f32((uv.y >> 2) & 1);
        let grain = f32((uv.x >> 1u) & 3i) / 3.0;
        return 0.85 + 0.07 * plank + 0.08 * grain + 0.05 * coarse;
    }
    if (material == MATERIAL_METAL) {
        return 0.94 + 0.08 * detail_hash01(uv.x / 2, uv.y / 4, 17u);
    }
    if (material == MATERIAL_SAND) {
        return 0.89 + 0.15 * detail_hash01(uv.x, uv.y, 23u);
    }
    if (material == MATERIAL_GRASS) {
        return 0.85 + 0.18 * detail_hash01(uv.x / 2, uv.y / 2, 29u);
    }
    if (material == MATERIAL_HAZARD) {
        let stripe = (uv.x + uv.y) & 7;
        return select(0.32, 1.0, stripe < 4);
    }
    if (material == MATERIAL_EMISSIVE) {
        return 1.0;
    }
    if (material == MATERIAL_GLASS) {
        return 0.97 + 0.04 * coarse;
    }
    if (material == MATERIAL_MIRROR || material == MATERIAL_GLASS_CLEAR) {
        // Polished surfaces must stay flat: a per-voxel brightness field reads
        // as dirt on a mirror.
        return 1.0;
    }
    return 0.92 + 0.12 * detail_hash01(uv.x, uv.y, 41u);
}

fn macro_variation(cell: vec3<i32>, material: u32) -> f32 {
    let block = cell / 8;
    let stain = detail_hash01(cell.x / 24 + cell.z / 16, cell.y / 12, material + 57u);
    return 0.88 + 0.10 * detail_hash01(block.x + block.y * 37, block.z, material + 101u) + 0.07 * stain;
}

// +/-4 % brightness keyed by world position: 25 cm grain plus 2 m patches.
fn micro_jitter(world: vec3<f32>, material: u32) -> f32 {
    let fine = floor(world / 0.25);
    let coarse = floor(world / 2.0);
    let a = detail_hash01(i32(fine.x) + i32(fine.z) * 31, i32(fine.y), material + 211u);
    let b = detail_hash01(i32(coarse.x), i32(coarse.z) + i32(coarse.y) * 17, material + 307u);
    return 0.96 + 0.055 * a + 0.025 * b;
}

// 3x3 depth/normal bilateral average of the full-resolution shadow buffer.
//
// `shadow.wgsl` traces exactly one ray per texel with a screen-space dither, so
// the penumbra is a random mix of lit and shadowed texels. At full resolution
// no bilinear filtering is left to hide that, and the isolated texels read as
// "pepper" along a long cast shadow. The filter averages the dither but keeps
// every edge that also exists in the G-buffer: taps with a different depth or
// normal (another object, the sky) get a weight near zero, so the shadow
// silhouette stays as sharp as the geometry.
fn filter_shadow_bilateral(pixel: vec2<i32>, center_depth: f32, center_normal: vec3<f32>) -> f32 {
    let last = vec2<i32>(textureDimensions(shadow_tex)) - vec2<i32>(1);
    let depth_scale = max(0.02, center_depth * 0.01);
    var sum = 0.0;
    var weight = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let tap = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), last);
            let tap_depth = textureLoad(depth_tex, tap, 0).r;
            let tap_normal = textureLoad(normal_tex, tap, 0).xyz;
            let w_depth = exp(-abs(tap_depth - center_depth) / depth_scale);
            let w_normal = pow(max(dot(normalize(tap_normal), center_normal), 0.0), 8.0);
            let w = w_depth * w_normal;
            sum = sum + textureLoad(shadow_tex, tap, 0).r * w;
            weight = weight + w;
        }
    }
    if (weight < 1.0e-4) {
        return textureLoad(shadow_tex, pixel, 0).r;
    }
    return clamp(sum / weight, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_hdr);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixel = vec2<i32>(gid.xy);
    let size = vec2<f32>(dims);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / size;
    var ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    ndc = ndc + frame.params.zw * 2.0 / size;
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let rd = normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);

    let depth = textureLoad(depth_tex, pixel, 0).r;
    if (depth < 0.0) {
        textureStore(out_hdr, pixel, vec4<f32>(sky_color(rd), 1.0));
        return;
    }

    let albedo_sample = textureLoad(albedo_tex, pixel, 0);
    let normal_sample = textureLoad(normal_tex, pixel, 0);
    let material = textureLoad(material_tex, pixel, 0);

    let base_albedo = albedo_sample.rgb;
    let ao = clamp(albedo_sample.a, 0.0, 1.0);
    let normal = normalize(normal_sample.xyz);
    let roughness = clamp(normal_sample.w, 0.035, 1.0);
    let metalness = clamp(material.r, 0.0, 1.0);
    let emissive = material.g * 8.0;
    let world = frame.camera_pos.xyz + rd * depth;
    // Sun visibility now lives in a half- or full-resolution texture traced by
    // `shadow.wgsl`. Half-res bilinear turns each 2x2 texel block into a soft
    // penumbra but smears the block across shadow edges (grain larger than one
    // internal pixel); `SHATTERBOX_SHADOW_MODE=bilateral` keeps the half-res
    // trace and filters it depth/normal-aware instead, `=full` traces one ray
    // per internal pixel. Mode arrives in `params3.z`.
    let shadow_mode = u32(round(frame.params3.z));
    let shadows_on = (u32(frame.params3.x) & 1u) != 0u;
    var visibility = 1.0;
    if (shadows_on) {
        if (shadow_mode == 2u) {
            let half_dims = vec2<i32>(textureDimensions(shadow_tex));
            if (half_dims.x < vec2<i32>(dims).x) {
                visibility = clamp(
                    upsample_half(shadow_tex, pixel, vec2<i32>(dims), depth, normal, world).r,
                    0.0,
                    1.0,
                );
            } else {
                // A half-res filter on a full-res buffer would only blur.
                visibility = clamp(
                    textureSampleLevel(shadow_tex, linear_sampler, uv, 0.0).r,
                    0.0,
                    1.0,
                );
            }
        } else if (shadow_mode == 3u) {
            let shadow_dims = vec2<i32>(textureDimensions(shadow_tex));
            if (all(shadow_dims == vec2<i32>(dims))) {
                visibility = filter_shadow_bilateral(pixel, depth, normal);
            } else {
                visibility = clamp(
                    textureSampleLevel(shadow_tex, linear_sampler, uv, 0.0).r,
                    0.0,
                    1.0,
                );
            }
        } else {
            visibility = clamp(textureSampleLevel(shadow_tex, linear_sampler, uv, 0.0).r, 0.0, 1.0);
        }
    }
    // Per-voxel AO is carried in the G-buffer alpha and folded into `ao` below.
    // Stylised per-voxel detail, moved out of the G-buffer: brick patterns,
    // 1 m block variation and 2-3 m staining are cheap ALU here, while the
    // primary pass stays a lean "find the voxel and write it" kernel.
    let material_index = u32(round(clamp(material.b, 0.0, 1.0) * 255.0));
    let detail_cell = vec3<i32>(floor(world / 0.125));
    let detail = material_detail(material_index, detail_cell, normal);
    let blocks = macro_variation(detail_cell, material_index);
    let jitter = micro_jitter(world, material_index);
    let albedo = clamp(base_albedo * detail * blocks * jitter, vec3<f32>(0.0), vec3<f32>(1.0));
    let view = -rd;
    let sun = normalize(frame.sun_dir.xyz);
    let day = clamp(frame.sun_color.w, 0.0, 1.0);
    let sun_radiance = frame.sun_color.rgb;

    let f0 = mix(vec3<f32>(0.04), albedo, metalness);
    // Glass is not a diffuse surface: almost all of its energy is transmitted
    // (approximated with the reflections below) and specular.
    let is_glass = material_index == MATERIAL_GLASS;
    let is_clear_glass = material_index == MATERIAL_GLASS_CLEAR;
    let is_glass_like = is_glass || is_clear_glass;
    let diffuse_albedo = albedo * (1.0 - metalness) * select(1.0, 0.10, is_glass_like);
    let ndv = max(dot(normal, view), 1e-4);
    let ndl = max(dot(normal, sun), 0.0);

    var color = vec3<f32>(0.0);
    if (ndl > 0.0) {
        let half_dir = normalize(sun + view);
        let ndh = max(dot(normal, half_dir), 0.0);
        let vdh = max(dot(view, half_dir), 0.0);
        let d = distribution_ggx(ndh, roughness);
        let g = geometry_smith(ndv, ndl, roughness);
        let f = fresnel_schlick(vdh, f0);
        let specular = d * g * f / max(4.0 * ndv * ndl, 1e-4);
        let sun_vis = ndl * visibility;
        color = color + (diffuse_albedo / PI + specular) * sun_radiance * sun_vis;
    }

    // Sky/ground hemisphere ambient with baked voxel AO. The tints are
    // deliberately separated: a cool blue sky fills upward-facing surfaces and
    // a warm ground bounce fills the underside, so shadows have a colour
    // gradient instead of one flat grey ambient term.
    let sky_tint = mix(vec3<f32>(0.18, 0.28, 0.52), vec3<f32>(0.52, 0.66, 0.86), day);
    let ground_tint = mix(vec3<f32>(0.16, 0.16, 0.20), vec3<f32>(0.52, 0.44, 0.34), day);
    let sky_ambient = sky_tint * mix(0.055, 0.32, day);
    let ground_ambient = ground_tint * mix(0.030, 0.16, day);
    // Half-resolution screen-space GI: `a` is SSAO, `rgb` is one-bounce
    // indirect (SSGI + voxel cone trace). Baked voxel AO still multiplies in,
    // because it knows about topology the screen cannot see.
    var gi_sample = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let gi_on = (u32(frame.params3.x) & 14u) != 0u;
    if (gi_on) {
        gi_sample = upsample_half(gi_tex, pixel, vec2<i32>(dims), depth, normal, world);
    }
    let gi_ao = clamp(gi_sample.a, 0.0, 1.0);
    let occlusion = ao * mix(1.0, gi_ao, 0.75);
    // VCT and DDGI both carry the sky/ground hemisphere with voxel visibility.
    // DDGI is a replacement, not an addition: with the fill left at 1.0 the
    // measured frame was +39% brighter than the no-DDGI reference, which is
    // exactly the double count. 0.6 was the best balance found: dropping the
    // fill to 0 made the 07:00 interior 26% darker than the reference because
    // 4 m probes under-sample a shadowed room, while 0.6 keeps the day within
    // +18% / night +10% and the shadowed interior within -9%.
    let vct_on = (u32(frame.params3.x) & 8u) != 0u;
    let ddgi_on = (u32(frame.params3.x) & 2048u) != 0u;
    // Diagnostic bit 6 (64) keeps the full analytic hemisphere even with VCT
    // on, so the lighting_set A/B can tell whether the 0.25 fill is flattening
    // the shadows or whether VCT itself is the problem.
    let full_ambient = (u32(frame.params4.z) & 64u) != 0u;
    let volume_gi = vct_on || ddgi_on;
    let ambient_scale = select(
        select(1.0, 0.55, volume_gi && !full_ambient),
        1.0,
        ddgi_on && !full_ambient,
    );
    // A little warm light from the sunlit floor, strongest on downward faces.
    let floor_bounce = clamp(0.5 - normal.y * 0.5, 0.0, 1.0) * day;
    let ambient = mix(ground_ambient, sky_ambient, normal.y * 0.5 + 0.5)
        + vec3<f32>(0.075, 0.052, 0.030) * floor_bounce;
    color = color + diffuse_albedo * ambient * occlusion * ambient_scale;
    // AO multiplies the indirect term as well: without it a 4 m probe grid
    // paints the same bounce into every crease and reads flat.
    color = color + diffuse_albedo * gi_sample.rgb * mix(0.10, 0.35, occlusion);
    // Cheapest believable bounce: a surface facing the sun but occluded from it
    // still catches light thrown by the sunlit floor and walls. The old term
    // only fired on back-facing normals, so an interior wall in shadow stayed
    // flat; this one uses the shadow visibility to add the warm fill exactly
    // where direct sun is missing.
    let backlit = clamp(-dot(normal, sun), 0.0, 1.0);
    let shadowed_sun = (1.0 - visibility) * clamp(dot(normal, sun), 0.0, 1.0);
    color = color + diffuse_albedo * vec3<f32>(0.13, 0.10, 0.075) * backlit * day;
    color = color + diffuse_albedo * vec3<f32>(0.085, 0.060, 0.040) * shadowed_sun * day;
    // Specular: screen-space reflections (traced in `gi.wgsl`) plus a sky
    // fallback for the part of the ray the screen never sees. Glass reflects at
    // normal incidence, metals and glossy surfaces fade in with fresnel.
    // The reflection texture is only meaningful when the SSR bit is on: the GI
    // pass is what fills it, and when it is skipped the texture still holds the
    // last frame it ran. Reading it unconditionally left frozen reflections.
    // Its alpha is the screen-space hit confidence written by `gi.wgsl`; a
    // mirror whose ray left the screen now falls back to the sky instead of the
    // old 3 % sliver, which is what made the pane read as a black slab.
    let ssr_on = (u32(frame.params3.x) & 64u) != 0u;
    var ssr = vec4<f32>(0.0);
    if (ssr_on) {
        ssr = upsample_half(reflection_tex, pixel, vec2<i32>(dims), depth, normal, world);
    }
    let ssr_hit = clamp(ssr.a, 0.0, 1.0);
    // Directional sky reflection instead of a constant vertical gradient: a
    // window, a glass pane and a metal edge now pick up the actual horizon,
    // sun disk and cloud colour, which is what makes the day read as lit from
    // somewhere rather than wrapped in a flat studio dome.
    var sky_reflection = sky_color(normalize(reflect(rd, normal)));
    sky_reflection = min(sky_reflection, vec3<f32>(8.0));
    sky_reflection = sky_reflection * mix(0.35, 1.0, day) + vec3<f32>(0.012, 0.016, 0.025);
    let refl_fade = clamp(1.0 - roughness * 2.2, 0.0, 1.0);
    let refl_gain = select(mix(0.10, 0.90, day), 1.00, is_glass_like);
    let is_mirror = material_index == MATERIAL_MIRROR;
    // Glass is reflective even head-on (a 4% fresnel term would read as a grey
    // hole), so it gets a flat 0.45 baseline that climbs to a mirror at grazing
    // angles. Dielectrics and metals keep the physical Schlick curve.
    let schlick = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - ndv, 5.0);
    let glass_weight = vec3<f32>(mix(0.45, 1.0, pow(1.0 - ndv, 3.0)));
    let fresnel = select(schlick, glass_weight, is_glass_like);
    // A mirror reflects the room, not the sky dome: when its screen-space ray is
    // lost, give the analytic sky full weight so the surface still reads as a
    // mirror. Dielectrics keep the conservative 0.35 fill.
    let sky_fill = select(0.35, 1.0, is_mirror);
    let specular_env =
        (ssr.rgb + sky_reflection * (1.0 - ssr_hit) * sky_fill) * fresnel;
    color = color + specular_env * refl_gain * refl_fade * mix(0.35, 1.0, occlusion);
    // Glass transmission approximation: deferred shading cannot trace a second
    // ray behind the pane yet (WBOIT is still open), so give clear glass a
    // share of the surrounding ambient tinted by its own albedo. It is not a
    // refraction, but it stops the pane from reading as an opaque mirror.
    if (is_glass_like) {
        color = color + albedo * (sky_ambient + ground_ambient) * 0.45 * (1.0 - fresnel);
    }

    // Local lights used to be added unconditionally, so a neon tube lit the far
    // side of a wall. `light_shadow.wgsl` traces one ray per half-resolution
    // texel per light and stores a bitmask; here the 2x2 block around this pixel
    // is blended by hand (uint textures cannot be filtered) which turns the
    // 0/1 bits into a 0.0–1.0 soft penumbra.
    let light_dims = vec2<i32>(textureDimensions(light_shadow_tex));
    var light_masks = vec4<u32>(0u);
    var light_weights = vec4<f32>(0.0);
    if (shadows_on) {
        let half = (vec2<f32>(pixel) + vec2<f32>(0.5)) * 0.5 - vec2<f32>(0.5);
        let base = vec2<i32>(floor(half));
        let frac = half - floor(half);
        let hi = light_dims - vec2<i32>(1);
        light_masks = vec4<u32>(
            textureLoad(light_shadow_tex, clamp(base, vec2<i32>(0), hi), 0).r,
            textureLoad(light_shadow_tex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), hi), 0).r,
            textureLoad(light_shadow_tex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), hi), 0).r,
            textureLoad(light_shadow_tex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), hi), 0).r,
        );
        light_weights = vec4<f32>(
            (1.0 - frac.x) * (1.0 - frac.y),
            frac.x * (1.0 - frac.y),
            (1.0 - frac.x) * frac.y,
            frac.x * frac.y,
        );
    }

    // Point lights keep the first slots, the rest are Phase 4 area emitters
    // (neon tubes and glowing panels), which are evaluated at the representative
    // point clamped onto the emitter.
    let light_count = i32(frame.params2.z);
    for (var i = 0; i < 12; i = i + 1) {
        if (i >= light_count) {
            break;
        }
        let axis = frame.light_axis[i];
        let emitter = emitter_point(frame.light_pos_radius[i].xyz, axis.xyz, axis.w, world);
        if (emitter.w < 0.5) {
            continue;
        }
        let light_pos = emitter.xyz;
        let radius = max(frame.light_pos_radius[i].w, 0.1);
        let to_light = light_pos - world;
        let distance_sq = dot(to_light, to_light);
        if (distance_sq > radius * radius) {
            continue;
        }
        var light_visibility = 1.0;
        if (shadows_on) {
            let bit = u32(i);
            let taps = vec4<f32>(
                f32((light_masks.x >> bit) & 1u),
                f32((light_masks.y >> bit) & 1u),
                f32((light_masks.z >> bit) & 1u),
                f32((light_masks.w >> bit) & 1u),
            );
            light_visibility = dot(taps, light_weights);
        }
        if (light_visibility <= 0.0) {
            continue;
        }
        let distance = sqrt(distance_sq);
        let l = to_light / max(distance, 1e-4);
        // Plateau-free radius kernel. The previous window was 1.0 out to 75 %
        // of the radius and only then rolled off, which read as a lit sphere
        // with a visible outer ring. `(1 - d²/r²)²` falls from the centre and
        // reaches zero with zero slope, so the outer boundary is invisible.
        let normalized = clamp(distance / radius, 0.0, 1.0);
        let cutoff = 1.0 - normalized * normalized;
        // Inverse-square falloff, floored so a shading point centimetres away
        // from the emitter cannot explode.
        let falloff = cutoff * cutoff / max(distance_sq, 0.25);
        let nld = max(dot(normal, l), 0.0);
        let half_dir = normalize(l + view);
        // Extended emitters spread their highlight: widen the lobe by the
        // emitter size, which reads as a soft neon reflection instead of a dot.
        let lobe = mix(64.0, 4.0, roughness) / (1.0 + abs(axis.w) * 1.5);
        // Specular must obey the same N.L gate as diffuse. A light behind the
        // surface leaves the shadow-mask bit clear now, but the gate keeps a
        // grazing highlight from leaking through a wall even if a mask tap is
        // filtered across a silhouette.
        let spec = pow(max(dot(normal, half_dir), 0.0), lobe) * select(0.0, 1.0, nld > 0.0);
        let light = falloff * light_visibility;
        color = color + frame.light_color[i].rgb * light * (diffuse_albedo * nld / PI + vec3<f32>(spec) * f0);
    }

    color = color + albedo * emissive;

    // Volumetric fog from `fog.wgsl` replaces the old analytic haze: it knows
    // about the sun shafts carved through the arena and about the light glow.
    // With the effect off, a weak analytic approximation keeps the depth cue.
    let fog_on = (u32(frame.params3.x) & 32u) != 0u;
    if (fog_on) {
        let fog = upsample_half(fog_tex, pixel, vec2<i32>(dims), depth, normal, world);
        color = color * fog.a + fog.rgb;
    } else {
        let fog_amount = (1.0 - exp(-max(depth, 0.0) * 0.0008)) * mix(0.55, 1.0, clamp(rd.y + 0.3, 0.0, 1.0));
        let fog_day = mix(vec3<f32>(0.34, 0.45, 0.62), sky_color(rd), 0.30);
        let fog_night = mix(vec3<f32>(0.015, 0.022, 0.040), sky_color(rd), 0.30);
        let fog_color = mix(fog_night, fog_day, day);
        color = mix(color, fog_color, clamp(fog_amount, 0.0, 0.6));
    }

    // `SHATTERBOX_DEBUG_HIT`: magenta disc centred on the CPU aim raycast. If
    // the crater is carved anywhere else, the offset is visible in a single
    // screenshot instead of being inferred from geometry.
    let debug_radius = frame.debug_hit.w;
    if (debug_radius > 0.0 && distance(world, frame.debug_hit.xyz) <= debug_radius) {
        color = mix(color, vec3<f32>(1.0, 0.05, 0.85), 0.85);
    }

    textureStore(out_hdr, pixel, vec4<f32>(max(color, vec3<f32>(0.0)), 1.0));
}
