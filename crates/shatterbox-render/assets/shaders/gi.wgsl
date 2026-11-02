// Screen-space GI pass (half internal resolution).
//
// Runs after the G-buffer and before deferred lighting, and produces:
//   rgb = one-bounce indirect diffuse radiance (SSGI + voxel cone trace)
//   a   = screen-space ambient occlusion
//
// The lighting pass multiplies the AO into the ambient term and adds the
// indirect radiance to the diffuse albedo. Every sub-effect can be disabled
// independently through `params3.x`; disabled paths write neutral values, so
// the bind groups never have to change at runtime.
//
// Notes on correctness:
// - SSGI reads the *previous* frame's HDR result, so it is a lagged one-bounce
//   bounce. That is the standard trick for avoiding a lighting feedback loop;
//   TAA hides the one-frame lag on moving surfaces.
// - Depth is linear distance along the view ray (not NDC depth), which makes
//   occlusion tests plain distance comparisons.

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
    // x = effect bitmask (1 shadows, 2 ssao, 4 ssgi, 8 vct, 16 taa, 32 fog,
    // 64 ssr).
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var color_tex: texture_2d<f32>;
@group(0) @binding(4) var probe_tex: texture_3d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;
@group(0) @binding(6) var out_gi: texture_storage_2d<rgba16float, write>;
/// Screen-space reflection for glossy surfaces; consumed by `lighting.wgsl`.
/// rgb = reflected radiance, a = screen-space hit confidence (0 when the ray
/// left the screen or hit nothing, so the lighting pass can fade in the sky).
@group(0) @binding(7) var out_reflection: texture_storage_2d<rgba16float, write>;
/// Octahedral DDGI atlas (irradiance) and the depth moments used for the
/// Chebyshev visibility test. Both are written by `ddgi_blend.wgsl`.
@group(0) @binding(8) var ddgi_irradiance: texture_2d<f32>;
@group(0) @binding(9) var ddgi_moments: texture_2d<f32>;
/// `{ slice, hysteresis, spacing, reset }` shared with the DDGI passes.
@group(0) @binding(10) var<uniform> ddgi_params: DdgiParams;

struct DdgiParams {
    slice: u32,
    hysteresis: f32,
    spacing: f32,
    reset: u32,
    tile: u32,
    samples: u32,
    slices: u32,
    priority: u32,
    priority_center: vec3<f32>,
    priority_radius: f32,
    priority_hysteresis: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

const AO_SAMPLES: i32 = 8;
const AO_RADIUS: f32 = 1.1;
const SSGI_STEPS: i32 = 4;
const SSGI_DIRECTIONS: i32 = 4;
const VCT_CONES: i32 = 3;

// Must match DDGI_GRID / DDGI_TILE / MAX_DISTANCE in `ddgi_update.wgsl`.
const DDGI_GRID_X: i32 = 64;
const DDGI_GRID_Y: i32 = 16;
const DDGI_GRID_Z: i32 = 64;
const DDGI_TILE: i32 = 8;
const DDGI_MAX_DISTANCE: f32 = 24.0;

fn effect_on(bit: u32) -> bool {
    return (u32(frame.params3.x) & bit) != 0u;
}

fn world_from_uv(uv: vec2<f32>, distance: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    return frame.camera_pos.xyz + normalize(far - near) * distance;
}

fn hash12(p: vec2<f32>, seed: f32) -> f32 {
    return fract(sin(dot(p + vec2<f32>(seed * 13.37), vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

/// Tangent basis for the hemisphere used by the cone trace.
fn basis(n: vec3<f32>) -> mat3x3<f32> {
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.9);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return mat3x3<f32>(t, b, n);
}

fn ddgi_octahedral_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-4);
    return select(p, vec2<f32>((1.0 - abs(p.yx)) * sign(p.xy)), n.z < 0.0);
}

/// Nearest texel of a probe tile for `dir`. Used by the moment (visibility)
/// lookup, where bilinear filtering would invent moments between directions.
fn ddgi_texel(probe: vec3<i32>, dir: vec3<f32>) -> vec2<i32> {
    let tile = f32(clamp(ddgi_params.tile, 1u, u32(DDGI_TILE)));
    let oct = ddgi_octahedral_encode(dir);
    let uv = clamp(oct * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0)) * tile - 0.5;
    let texel = clamp(floor(uv), vec2<f32>(0.0), vec2<f32>(tile - 1.0));
    return vec2<i32>(
        probe.x * DDGI_TILE + i32(texel.x),
        probe.z * DDGI_GRID_Y * DDGI_TILE + probe.y * DDGI_TILE + i32(texel.y),
    );
}

/// Manually bilinear irradiance inside one probe tile. A filtered sampler would
/// bleed across the 1-texel seam between neighbouring probes, which shows up as
/// a grid of bright lines on flat walls.
fn ddgi_probe_irradiance(probe: vec3<i32>, dir: vec3<f32>) -> vec3<f32> {
    let tile = f32(clamp(ddgi_params.tile, 1u, u32(DDGI_TILE)));
    let oct = ddgi_octahedral_encode(dir);
    let uv = clamp(oct * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0)) * tile - 0.5;
    let base = floor(uv);
    let frac = uv - base;
    let hi = vec2<f32>(tile - 1.0);
    let x0 = i32(clamp(base.x, 0.0, hi.x));
    let y0 = i32(clamp(base.y, 0.0, hi.y));
    let x1 = i32(clamp(base.x + 1.0, 0.0, hi.x));
    let y1 = i32(clamp(base.y + 1.0, 0.0, hi.y));
    let column = probe.x * DDGI_TILE;
    let row = probe.z * DDGI_GRID_Y * DDGI_TILE + probe.y * DDGI_TILE;
    let c00 = textureLoad(ddgi_irradiance, vec2<i32>(column + x0, row + y0), 0).rgb;
    let c10 = textureLoad(ddgi_irradiance, vec2<i32>(column + x1, row + y0), 0).rgb;
    let c01 = textureLoad(ddgi_irradiance, vec2<i32>(column + x0, row + y1), 0).rgb;
    let c11 = textureLoad(ddgi_irradiance, vec2<i32>(column + x1, row + y1), 0).rgb;
    return mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);
}

/// Chebyshev visibility from the depth moments. A probe whose mean hit distance
/// along `dir` is much closer than the shaded point is occluded, which is what
/// keeps bounce from leaking through the wall into the neighbouring room.
fn ddgi_visibility(probe: vec3<i32>, dir: vec3<f32>, distance: f32) -> f32 {
    let moments = textureLoad(ddgi_moments, ddgi_texel(probe, dir), 0);
    let mean = moments.r * DDGI_MAX_DISTANCE;
    let variance = max(moments.g * DDGI_MAX_DISTANCE * DDGI_MAX_DISTANCE - mean * mean, 0.0025);
    if (distance <= mean) {
        return 1.0;
    }
    let delta = distance - mean;
    // Soft Chebyshev: a hard test turns every probe whose mean hit is a few
    // centimetres short into a black block. Blend the occluded answer with 1.0
    // and floor it, so a single near wall darkens the probe instead of killing it.
    let cheb = clamp(variance / (variance + delta * delta), 0.0, 1.0);
    return clamp(mix(1.0, cheb, 0.65), 0.08, 1.0);
}

/// Trilinear probe interpolation of irradiance for a surface point/normal.
/// Probes whose moments say "a wall is closer than this point" are dropped; if
/// all eight are occluded the result fades to black rather than leaking.
fn ddgi_sample_irradiance(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let spacing = max(ddgi_params.spacing, 1.0);
    let grid = vec3<i32>(DDGI_GRID_X, DDGI_GRID_Y, DDGI_GRID_Z);
    let base = (p - frame.world_min.xyz) / spacing - vec3<f32>(0.5);
    let cell = floor(base);
    let frac = clamp(base - cell, vec3<f32>(0.0), vec3<f32>(1.0));
    var sum = vec3<f32>(0.0);
    var weight_sum = 0.0;
    for (var corner = 0; corner < 8; corner = corner + 1) {
        let offset = vec3<f32>(f32(corner & 1), f32((corner >> 1) & 1), f32((corner >> 2) & 1));
        let index = clamp(vec3<i32>(cell) + vec3<i32>(offset), vec3<i32>(0), grid - vec3<i32>(1));
        let w = mix(1.0 - frac, frac, offset).x
            * mix(1.0 - frac, frac, offset).y
            * mix(1.0 - frac, frac, offset).z;
        let probe_center = frame.world_min.xyz + (vec3<f32>(index) + vec3<f32>(0.5)) * spacing;
        // Visibility is measured along the direction from the probe to the
        // surface, not along the surface normal. That is what stops the
        // trilinear blend from leaking sky from a probe sitting outside the
        // room: the wall between them is closer than the point, so the
        // Chebyshev test drops that probe.
        let to_point = p - probe_center;
        let distance = length(to_point);
        let direction = to_point / max(distance, 1e-4);
        let visibility = ddgi_visibility(index, direction, distance);
        let weight = w * visibility;
        sum = sum + ddgi_probe_irradiance(index, n) * weight;
        weight_sum = weight_sum + weight;
    }
    return sum / max(weight_sum, 1e-4);
}


@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_gi);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let dims_i = vec2<i32>(dims);
    let full_dims = vec2<i32>(textureDimensions(depth_tex));
    let full_pixel = clamp(pixel * 2, vec2<i32>(0), full_dims - vec2<i32>(1));

    let distance = textureLoad(depth_tex, full_pixel, 0).r;
    if (distance < 0.0) {
        // Sky: no AO, no indirect, no reflection.
        textureStore(out_gi, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(out_reflection, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let normal_sample = textureLoad(normal_tex, full_pixel, 0);
    let normal = normal_sample.xyz;
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let world = world_from_uv(uv, distance);
    let noise = hash12(vec2<f32>(pixel), frame.params.y);
    // After a resize or a TAA toggle the history target holds uninitialised
    // memory; SSGI would turn that into a full-screen flash.
    let history_valid = frame.params2.y < 0.5;

    // ---- SSAO -----------------------------------------------------------
    var ao = 1.0;
    if (effect_on(2u)) {
        var occlusion = 0.0;
        let pixel_radius = clamp(52.0 / max(distance, 0.75), 2.0, 22.0);
        for (var i = 0; i < AO_SAMPLES; i = i + 1) {
            let fi = f32(i);
            let angle = (fi + noise) * 6.2831853 / f32(AO_SAMPLES);
            let radius = pixel_radius * ((fi + 1.0) / f32(AO_SAMPLES));
            let offset = vec2<f32>(cos(angle), sin(angle)) * radius;
            let sample_uv = uv + offset / vec2<f32>(dims);
            let sample_pixel = clamp(
                vec2<i32>(sample_uv * vec2<f32>(dims)),
                vec2<i32>(0),
                dims_i - vec2<i32>(1),
            );
            // The depth buffer is at internal resolution; the AO pass is at
            // half resolution, so take the nearest full-resolution texel.
            let inner = clamp(sample_pixel * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
            let sample_distance = textureLoad(depth_tex, inner, 0).r;
            if (sample_distance < 0.0) {
                continue;
            }
            let sample_world = world_from_uv(
                (vec2<f32>(sample_pixel) + vec2<f32>(0.5)) / vec2<f32>(dims),
                sample_distance,
            );
            let delta = sample_world - world;
            let span = length(delta);
            if (span < 0.05 || span > AO_RADIUS) {
                continue;
            }
            // Only accept occluders in front of the tangent plane.
            let facing = dot(delta / span, normal);
            if (facing < 0.08) {
                continue;
            }
            occlusion = occlusion + (1.0 - span / AO_RADIUS) * facing;
        }
        ao = clamp(1.0 - occlusion * (1.35 / f32(AO_SAMPLES)), 0.0, 1.0);
    }

    // Screen-space diffuse bounce removed: sampling the previous frame's bright
    // floor gathered a soft ball of light that followed the camera. Indirect
    // light now comes only from the probe cone trace below.
    var indirect = vec3<f32>(0.0);

    // ---- Voxel cone trace ------------------------------------------------
    // When DDGI is enabled its octahedral atlas replaces the colourless probe
    // cone trace; running both would double the ambient bounce.
    let ddgi_on = effect_on(2048u);
    if (effect_on(8u) && !ddgi_on) {
        let size = frame.world_max.xyz - frame.world_min.xyz;
        let base = basis(normal);
        let origin = world + normal * 0.18;
        var cone = vec3<f32>(0.0);
        var cone_weight = 0.0;
        for (var c = 0; c < VCT_CONES; c = c + 1) {
            let angle = (f32(c) + noise * 0.7) * 6.2831853 / f32(VCT_CONES);
            let tilt = 0.55 + 0.35 * f32(c) / f32(VCT_CONES);
            let local = normalize(vec3<f32>(cos(angle) * tilt, sin(angle) * tilt, 1.0));
            let dir = normalize(base * local);
            // Geometric cone: radius doubles with distance, so far taps cover
            // large solid angles with a couple of lookups.
            for (var t = 0; t < 3; t = t + 1) {
                let reach = 1.2 * pow(2.15, f32(t));
                let sample_world = origin + dir * reach;
                let uvw = (sample_world - frame.world_min.xyz) / size;
                if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
                    break;
                }
                let probe = textureSampleLevel(probe_tex, linear_sampler, uvw, 0.0).rgb;
                let weight = exp(-reach * 0.22);
                cone = cone + probe * weight;
                cone_weight = cone_weight + weight;
                // Widened footprint: stop tracing once the cone is wider than
                // a few probe cells, the extra taps add nothing.
                if (reach > 8.0) {
                    break;
                }
            }
        }
        if (cone_weight > 0.0) {
            indirect = indirect + cone / cone_weight * 0.9 * ao;
        }
    }

    // ---- DDGI indirect ---------------------------------------------------
    // Irradiance from the octahedral atlas built by `ddgi_update.wgsl` +
    // `ddgi_blend.wgsl`. The atlas carries the colour of the surfaces the probe
    // rays hit, so this is the term that lets a red wall tint the floor next to
    // it, and a blue wall tint the other side.
    if (ddgi_on) {
        let irradiance = ddgi_sample_irradiance(world + normal * 0.25, normal);
        indirect = indirect + irradiance / 3.14159265 * ao;
    }

    // ---- Screen-space reflection ----------------------------------------
    // Mirrored ray for glossy surfaces (mirror, glass, metal, wet concrete).
    // The ray is marched in world space and projected every step, so the
    // half-resolution pass can reuse the previous frame's HDR colour. The
    // marched point and the depth-buffer surface are compared with a thickness
    // that grows with distance and a five-step binary refine, otherwise the
    // old fixed 0.35 m / 0.55 m tests missed thin glass and accepted walls
    // behind the ray, which is what made the pane read as a dark slab.
    var reflection = vec3<f32>(0.0);
    var reflection_hit = 0.0;
    let surface_roughness = clamp(normal_sample.w, 0.0, 1.0);
    let view_ray = normalize(world - frame.camera_pos.xyz);
    let mirror = reflect(view_ray, normal);
    // SSR has its own bit (64): turning SSGI off must not turn reflections off.
    if (effect_on(64u) && history_valid && surface_roughness < 0.45) {
        let origin = world + normal * 0.06;
        var previous_t = 0.0;
        // Jittered start hides the marching banding without a per-pixel loop.
        var t = 0.12 + noise * 0.20;
        for (var i = 0; i < 30; i = i + 1) {
            let probe = origin + mirror * t;
            let clip = frame.view_proj * vec4<f32>(probe, 1.0);
            if (clip.w > 0.0) {
                let ndc = clip.xyz / clip.w;
                let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                if (all(suv > vec2<f32>(0.0)) && all(suv < vec2<f32>(1.0))) {
                    let sp = clamp(
                        vec2<i32>(suv * vec2<f32>(dims)),
                        vec2<i32>(0),
                        dims_i - vec2<i32>(1),
                    );
                    let inner = clamp(sp * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
                    let sd = textureLoad(depth_tex, inner, 0).r;
                    if (sd >= 0.0) {
                        let scene_point = world_from_uv(
                            (vec2<f32>(sp) + vec2<f32>(0.5)) / vec2<f32>(dims),
                            sd,
                        );
                        let thickness = 0.18 + t * 0.035;
                        let gap = length(scene_point - probe);
                        if (gap < thickness) {
                            // Walk back to the actual intersection: the first
                            // coarse hit is up to one step in front of it.
                            var lo = previous_t;
                            var hi = t;
                            for (var r = 0; r < 5; r = r + 1) {
                                let mid = 0.5 * (lo + hi);
                                let marched = origin + mirror * mid;
                                let mclip = frame.view_proj * vec4<f32>(marched, 1.0);
                                let mndc = mclip.xyz / max(mclip.w, 1e-4);
                                let msuv = vec2<f32>(mndc.x * 0.5 + 0.5, 0.5 - mndc.y * 0.5);
                                let msp = clamp(
                                    vec2<i32>(msuv * vec2<f32>(dims)),
                                    vec2<i32>(0),
                                    dims_i - vec2<i32>(1),
                                );
                                let minner = clamp(msp * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
                                let msd = textureLoad(depth_tex, minner, 0).r;
                                let mpoint = select(
                                    marched,
                                    world_from_uv(msuv, max(msd, 0.0)),
                                    msd >= 0.0,
                                );
                                if (length(mpoint - marched) < thickness) {
                                    hi = mid;
                                } else {
                                    lo = mid;
                                }
                            }
                            let hit_t = 0.5 * (lo + hi);
                            let hclip = frame.view_proj * vec4<f32>(origin + mirror * hit_t, 1.0);
                            let hndc = hclip.xyz / max(hclip.w, 1e-4);
                            let hsuv = vec2<f32>(hndc.x * 0.5 + 0.5, 0.5 - hndc.y * 0.5);
                            let edge = smoothstep(0.0, 0.06, hsuv.x)
                                * smoothstep(0.0, 0.06, 1.0 - hsuv.x)
                                * smoothstep(0.0, 0.06, hsuv.y)
                                * smoothstep(0.0, 0.06, 1.0 - hsuv.y);
                            reflection = textureSampleLevel(
                                color_tex,
                                linear_sampler,
                                hsuv,
                                0.0,
                            ).rgb * edge;
                            reflection_hit = edge;
                            break;
                        }
                    }
                }
            }
            previous_t = t;
            t = t + max(0.12, t * 0.11);
        }
    }
    // Off-screen / lost rays: the DDGI atlas knows the coloured light around the
    // reflection direction, which is far better than painting a dark box black.
    // Confidence stays 0, so `lighting.wgsl` still adds the sky reflection.
    if (ddgi_on && reflection_hit <= 0.0) {
        reflection = ddgi_sample_irradiance(world + normal * 0.1, mirror) / 3.14159265;
    }

    textureStore(
        out_reflection,
        pixel,
        vec4<f32>(clamp(reflection, vec3<f32>(0.0), vec3<f32>(8.0)), clamp(reflection_hit, 0.0, 1.0)),
    );
    textureStore(
        out_gi,
        pixel,
        vec4<f32>(clamp(indirect, vec3<f32>(0.0), vec3<f32>(16.0)), clamp(ao, 0.0, 1.0)),
    );
}
