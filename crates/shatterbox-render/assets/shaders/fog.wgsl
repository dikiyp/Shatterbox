// Volumetric fog (half internal resolution).
//
// An exponential height fog with a sun in-scattering term. The sun visibility
// comes from the coarse irradiance volume built by `probe.wgsl` (its alpha
// channel stores the voxel-traced visibility towards the sun), so the fog gets
// real shafts through the arena geometry without a second DDA per step.
//
// Output: rgb = in-scattered radiance, a = transmittance. The lighting pass
// composites it as `color * a + rgb` before the temporal pass, which means TAA
// also resolves the (deliberately sparse) sampling pattern.

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
    // x = effect bitmask (1 shadows, 2 ssao, 4 ssgi, 8 vct, 16 taa, 32 fog),
    // y = fog density.
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
    // Declared only to keep the offsets right: the fog pass does not read area
    // lights or the debris bounds. `params4` must stay last, as in `FrameUniform`.
    light_axis: array<vec4<f32>, 12>,
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
    debug_hit: vec4<f32>,
    // x = debris reactive mask (unused here), y = fog height falloff in metres.
    params4: vec4<f32>,
    /// x = 1: trilinear probe sampling (default), 0: legacy nearest cell.
    params5: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var probe_tex: texture_3d<f32>;
@group(0) @binding(3) var out_fog: texture_storage_2d<rgba16float, write>;
/// Voxel-accurate sun visibility from `shadow.wgsl`, used by the screen-space
/// god-ray march. The probe volume is 8 m coarse and cannot resolve a window.
@group(0) @binding(4) var sun_shadow: texture_2d<f32>;
/// Dynamic smoke field (64 x 16 x 64, 4 m cells) advected on the CPU.
@group(0) @binding(5) var smoke_tex: texture_3d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;

const STEPS: i32 = 16;
const MAX_FOG_DISTANCE: f32 = 140.0;
/// Default height falloff in metres, used when the uniform carries no value.
/// 5 m keeps the haze on the ground instead of washing the upper storeys; the
/// F1 slider / `config/sky.ron` overrides it through `params4.y`.
const HEIGHT_SCALE_DEFAULT: f32 = 5.0;
const BASE_HEIGHT: f32 = 1.5;

fn world_from_uv(uv: vec2<f32>, distance: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    return frame.camera_pos.xyz + normalize(far - near) * distance;
}

fn probe_sun_visibility(p: vec3<f32>) -> f32 {
    let dims = vec3<f32>(textureDimensions(probe_tex));
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let uvw = (p - frame.world_min.xyz) / size;
    if (frame.params5.x > 0.5) {
        // The probe volume is 8 m per cell. Nearest sampling made the sun
        // visibility (and the interior mask derived from it) switch in hard
        // 8 m squares around the camera, visible as a lit/dark box edge in fog.
        // Trilinear filtering keeps the same data, so the boundary becomes a
        // gradient instead of a step. The half-texel inset avoids sampling the
        // clamped border on both sides.
        let inset = 0.5 / dims;
        let u = clamp(uvw, inset, vec3<f32>(1.0) - inset);
        return clamp(textureSampleLevel(probe_tex, linear_sampler, u, 0.0).a, 0.0, 1.0);
    }
    let cell = clamp(vec3<i32>(uvw * dims), vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(1));
    return clamp(textureLoad(probe_tex, cell, 0).a, 0.0, 1.0);
}

/// Henyey-Greenstein: strongly forward-scattering haze, which is what makes
/// the sun shafts around the silhouette pop.
fn phase_sun(cos_theta: f32) -> f32 {
    let g = 0.62;
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (12.566 * max(denom, 1e-3) * sqrt(max(denom, 1e-3)));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_fog);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let full_dims = vec2<i32>(textureDimensions(depth_tex));
    let full_pixel = clamp(pixel * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
    let distance = textureLoad(depth_tex, full_pixel, 0).r;
    if (distance < 0.0) {
        // Sky: the gradient already carries its own aerial perspective.
        textureStore(out_fog, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dims);
    let view_dir = normalize(world_from_uv(uv, 1.0) - frame.camera_pos.xyz);
    let sun = normalize(frame.sun_dir.xyz);
    let cos_theta = dot(view_dir, sun);
    let phase = phase_sun(cos_theta);
    let sun_radiance = frame.sun_color.rgb;
    let daylight = clamp(frame.sun_color.w, 0.0, 1.0);
    let ambient = mix(vec3<f32>(0.02, 0.026, 0.038), vec3<f32>(0.20, 0.23, 0.28), daylight);

    let density_scale = clamp(frame.params3.y, 0.0, 0.5);
    let smoke_on = (u32(frame.params3.x) & 8192u) != 0u;
    let world_size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    // Thickness of the fog layer: 5 m by default, adjustable from the overlay.
    let height_scale = select(HEIGHT_SCALE_DEFAULT, frame.params4.y, frame.params4.y > 0.0);
    let far = min(distance, MAX_FOG_DISTANCE);
    let offset = fract(sin(dot(vec2<f32>(pixel), vec2<f32>(12.9898, 78.233))) * 43758.5);

    var transmittance = 1.0;
    var inscatter = vec3<f32>(0.0);
    var medium = 0.0;
    var previous = 0.0;
    for (var i = 0; i < STEPS; i = i + 1) {
        // Quadratic distribution: most of the fog is within the first 40 m, so
        // spend the samples there instead of evenly across the whole ray.
        let x = (f32(i) + offset) / f32(STEPS);
        let t = far * x * x;
        let step_len = max(t - previous, 0.0);
        previous = t;
        if (step_len <= 0.0) {
            continue;
        }
        let p = frame.camera_pos.xyz + view_dir * t;
        let height = max(p.y - BASE_HEIGHT, 0.0);
        // Two octaves of cheap value noise break the perfectly smooth slab the
        // height ramp alone produces, so the haze has wisps without a particle
        // system. The amplitude stays low: this is atmosphere, not smoke.
        let drift = frame.params.x * 0.35;
        let wisp = 0.72
            + 0.34 * sin(p.x * 0.085 + drift) * sin(p.z * 0.067 - drift * 0.7)
            + 0.18 * sin(p.y * 0.21 + p.x * 0.05 + drift * 1.3);
        // Cheap single lookup of the coarse sun visibility in the probe volume.
        let sun_vis = probe_sun_visibility(p);
        // Fog is an outdoor atmosphere. The probe's alpha is the voxel-traced
        // sky visibility, so it doubles as an interior mask: indoors the
        // density drops to 6 %, which keeps window shafts but stops the
        // whole room from being veiled in beige in-scatter. At night the
        // point-light glow term was the worst offender; without this mask it
        // repainted every wall from the nearest lamp.
        let interior = mix(0.06, 1.0, sun_vis);
        let atmosphere =
            density_scale * exp(-height / height_scale) * clamp(wisp, 0.35, 1.45) * interior;
        // Dynamic smoke is not atmosphere: it is not masked by sky visibility,
        // it fills rooms, and it has its own (much higher) density.
        var smoke = 0.0;
        if (smoke_on) {
            let uvw = (p - frame.world_min.xyz) / world_size;
            if (all(uvw > vec3<f32>(0.0)) && all(uvw < vec3<f32>(1.0))) {
                smoke = textureSampleLevel(smoke_tex, linear_sampler, uvw, 0.0).r * 0.6;
            }
        }
        let density = atmosphere + smoke;
        if (density <= 1e-5) {
            continue;
        }
        let extinction = density * step_len;
        medium = medium + density * step_len;
        let lit = sun_radiance * (phase * sun_vis * 2.4) + ambient * 0.35;
        inscatter = inscatter + lit * density * step_len * transmittance;
        transmittance = transmittance * exp(-extinction);
    }

    // --- God rays --------------------------------------------------------
    // Screen-space radial march from this pixel towards the sun's projection.
    // Each tap reads the voxel-accurate shadow mask, so a beam is the sum of
    // lit segments along the line of sight. `1 - pixel_vis` keeps the effect in
    // shadowed air instead of washing out already sunlit surfaces.
    var shafts = 0.0;
    let sun_far = frame.camera_pos.xyz + sun * 200.0;
    let sun_clip = frame.view_proj * vec4<f32>(sun_far, 1.0);
    if (sun_clip.w > 0.0 && daylight > 0.01) {
        let sun_ndc = sun_clip.xyz / sun_clip.w;
        let sun_uv = vec2<f32>(sun_ndc.x * 0.5 + 0.5, 0.5 - sun_ndc.y * 0.5);
        let shadow_dims = vec2<i32>(textureDimensions(sun_shadow));
        let pixel_shadow = clamp(
            vec2<i32>(uv * vec2<f32>(shadow_dims)),
            vec2<i32>(0),
            shadow_dims - vec2<i32>(1),
        );
        let pixel_vis = clamp(textureLoad(sun_shadow, pixel_shadow, 0).r, 0.0, 1.0);
        let delta = sun_uv - uv;
        var lit_sum = 0.0;
        var weight_sum = 0.0;
        for (var i = 0; i < 24; i = i + 1) {
            let f = f32(i) / 24.0;
            let suv = uv + delta * (0.10 + 0.90 * f);
            if (any(suv < vec2<f32>(0.0)) || any(suv > vec2<f32>(1.0))) {
                continue;
            }
            let sp = clamp(
                vec2<i32>(suv * vec2<f32>(shadow_dims)),
                vec2<i32>(0),
                shadow_dims - vec2<i32>(1),
            );
            let weight = 1.0 - f;
            lit_sum = lit_sum + textureLoad(sun_shadow, sp, 0).r * weight;
            weight_sum = weight_sum + weight;
        }
        shafts = (lit_sum / max(weight_sum, 1e-4)) * (1.0 - pixel_vis);
        // A soft glow around the sun's projection makes the shaft read even
        // when the march crosses a narrow window slit between mostly lit rows.
        let screen_gap = length(uv - sun_uv);
        shafts = shafts + 0.55 * exp(-screen_gap * 3.5) * (1.0 - pixel_vis);
    }
    // God rays are light scattered by the participating medium, so they scale
    // with how much fog/smoke the ray actually crossed. Tying the term to the
    // marched density keeps the effect out of clear air instead of brightening
    // every shadowed floor.
    inscatter = inscatter
        + sun_radiance * shafts * phase * medium * transmittance * 1.2;

    // A single analytic glow from the point lights: cheap, and it is what makes
    // the neon corridor read as volumetric without marching every light.
    let count = i32(clamp(frame.params2.z, 0.0, 12.0));
    let mid = frame.camera_pos.xyz + view_dir * (far * 0.35);
    var light_glow = vec3<f32>(0.0);
    for (var i = 0; i < count; i = i + 1) {
        let lp = frame.light_pos_radius[i];
        let radius = max(lp.w, 0.001);
        let delta = lp.xyz - mid;
        let d2 = dot(delta, delta);
        if (d2 > radius * radius * 2.25) {
            continue;
        }
        // Same plateau-free kernel as the lighting pass, so the in-scattered
        // glow has no hard sphere edge either.
        let normalized = sqrt(d2) / max(radius, 0.001);
        let cutoff = max(1.0 - normalized * normalized, 0.0);
        light_glow = light_glow
            + frame.light_color[i].rgb * (cutoff * cutoff / (1.0 + d2 * 0.35));
    }
    let mid_height = max(mid.y - BASE_HEIGHT, 0.0);
    let mid_vis = probe_sun_visibility(mid);
    let mid_density = density_scale * exp(-mid_height / height_scale) * mix(0.06, 1.0, mid_vis);
    inscatter = inscatter + light_glow * mid_density * 6.0 * transmittance;

    textureStore(
        out_fog,
        pixel,
        vec4<f32>(clamp(inscatter, vec3<f32>(0.0), vec3<f32>(32.0)), clamp(transmittance, 0.0, 1.0)),
    );
}
