// Simplified voxel irradiance volume ("VCT lite", first half).
//
// A coarse 3D grid over the arena is filled with diffuse irradiance: sky
// ambient with a voxel-visibility test towards the sun, plus every point light
// with an occupancy-marched visibility term. The grid is tiny (a few thousand
// cells), so rebuilding it every frame costs well under 0.1 ms and reacts to
// destruction immediately.
//
// `gi.wgsl` cone-traces this volume per pixel; that is the "voxel cone
// tracing" half of the effect.

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
// Brick occupancy: one texel per 2 m brick (R32Uint brick index, 0 = empty).
@group(0) @binding(1) var brick_indices: texture_3d<u32>;
@group(0) @binding(2) var out_probe: texture_storage_3d<rgba16float, write>;

const PROBE_STEPS: i32 = 24;

fn world_to_cell(p: vec3<f32>) -> vec3<i32> {
    let dims = vec3<f32>(textureDimensions(brick_indices));
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let uv = (p - frame.world_min.xyz) / size;
    return vec3<i32>(clamp(uv * dims, vec3<f32>(0.0), dims - vec3<f32>(1.0)));
}

fn occupancy(p: vec3<f32>) -> u32 {
    return textureLoad(brick_indices, world_to_cell(p), 0).r;
}

/// Coarse visibility from `origin` towards `dir`. The march starts just outside
/// the probe (0.32 m) and steps 0.55 m, so a 0.25 m wall two voxels thick cannot
/// be stepped over: the old 2.2 m start skipped straight through house walls and
/// let interior lamp light leak onto the facade. A probe whose own 2 m brick is
/// occupied is inside geometry and sees nothing.
fn visible(origin: vec3<f32>, dir: vec3<f32>, length: f32) -> f32 {
    if (occupancy(origin) != 0u) {
        return 0.0;
    }
    var t = 0.32;
    for (var i = 0; i < PROBE_STEPS; i = i + 1) {
        if (t > length) {
            break;
        }
        if (occupancy(origin + dir * t) != 0u) {
            return 0.0;
        }
        t = t + 0.55;
    }
    return 1.0;
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_probe);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let cell = vec3<f32>(gid) + vec3<f32>(0.5);
    let size = frame.world_max.xyz - frame.world_min.xyz;
    let world = frame.world_min.xyz + cell / vec3<f32>(dims) * size;

    let daylight = clamp(frame.sun_color.w, 0.0, 1.0);
    // Same magnitudes as the analytic hemisphere in `lighting.wgsl`, so that
    // swapping one for the other does not change the overall exposure.
    let sky = mix(vec3<f32>(0.016, 0.022, 0.040), vec3<f32>(0.19, 0.20, 0.215), daylight);
    let ground = mix(vec3<f32>(0.010, 0.012, 0.018), vec3<f32>(0.11, 0.10, 0.085), daylight);

    // Ambient: mostly "upwards" but filtered by occupancy so interiors stay
    // dim instead of glowing. Two taps are enough at this resolution.
    let sun = normalize(frame.sun_dir.xyz);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sky_vis = 0.5 * (visible(world, up, 12.0) + visible(world, normalize(up + sun), 12.0));
    var irradiance = mix(ground, sky, clamp(up.y * 0.5 + 0.5, 0.0, 1.0)) * sky_vis;

    // One warm sun bounce. Without it the probe only knows the sky, so the
    // shaded side of the house receives no light from the sunlit ground and
    // reads black even at noon. 0.10 approximates a diffuse albedo / PI.
    let sun_vis = visible(world, sun, 16.0);
    irradiance = irradiance + frame.sun_color.rgb * sun_vis * 0.10;

    // Point lights with voxel visibility.
    let count = i32(clamp(frame.params2.z, 0.0, 12.0));
    for (var i = 0; i < count; i = i + 1) {
        let lp = frame.light_pos_radius[i];
        let radius = max(lp.w, 0.001);
        let delta = lp.xyz - world;
        let distance = length(delta);
        if (distance > radius) {
            continue;
        }
        let falloff = 1.0 - distance / radius;
        let weight = falloff * falloff / max(distance * distance, 0.25);
        let vis = visible(world, delta / max(distance, 0.001), distance);
        // One diffuse bounce, not direct light. The old term added the full
        // direct lamp irradiance to the probe and `lighting.wgsl` added it on
        // top of the direct light again. A coarse 8 m cell then summed all
        // five house lamps into a nearly uniform beige wash that no exposure
        // setting could fix. 0.06 is an approximate wall albedo / PI.
        irradiance = irradiance + frame.light_color[i].rgb * weight * vis * 0.06;
    }

    // Emissive trim gets a small self-glow contribution so the neon complex
    // lights its own corridor even before the point lights reach it.
    // The alpha channel stores sky visibility, which the volumetric fog pass
    // uses for cheap sun shafts without tracing its own shadow rays.
    textureStore(out_probe, vec3<i32>(gid), vec4<f32>(clamp(irradiance, vec3<f32>(0.0), vec3<f32>(24.0)), sky_vis));
}
