// Raymarched dynamic smoke volume (half internal resolution).
//
// The simulation lives in a local 64 x 32 x 64 box around one active source.
// This pass walks that box for every half-res pixel, stops at the G-buffer
// depth, and integrates in-scattering with:
//  * sun with Beer self-shadowing (a short opacity march towards the sun),
//  * sky/probe ambient (the coarse irradiance volume already carries the
//    voxel-traced sun visibility in alpha),
//  * nearby point lights (lamps, later the fire light),
//  * blackbody-ish emission from the temperature channel, tinted orange and
//    fading to grey as the smoke cools.
//
// Output: rgb = in-scattered radiance, a = transmittance. The composite pass
// blends it over the lit HDR image with One / OneMinusSrcAlpha.

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
    params3: vec4<f32>,
    internal_size: vec4<f32>,
    light_pos_radius: array<vec4<f32>, 12>,
    light_color: array<vec4<f32>, 12>,
    light_axis: array<vec4<f32>, 12>,
    debris_min: vec4<f32>,
    debris_max: vec4<f32>,
    debug_hit: vec4<f32>,
    params4: vec4<f32>,
};

struct SmokeSim {
    origin: vec4<f32>,
    spacing: vec4<f32>,
    grid: vec4<u32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    params3: vec4<f32>,
    source: array<vec4<f32>, 4>,
    source2: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> sim: SmokeSim;
@group(0) @binding(3) var<storage, read> state_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> solid: array<u32>;
@group(0) @binding(5) var probe_tex: texture_3d<f32>;
@group(0) @binding(6) var linear_sampler: sampler;
@group(0) @binding(7) var out_smoke: texture_storage_2d<rgba16float, write>;

const STEPS: i32 = 32;
const EXTINCTION: f32 = 2.0;

fn dims() -> vec3<i32> {
    return vec3<i32>(sim.grid.xyz);
}

fn index(x: i32, y: i32, z: i32) -> u32 {
    let d = dims();
    let cx = clamp(x, 0, d.x - 1);
    let cy = clamp(y, 0, d.y - 1);
    let cz = clamp(z, 0, d.z - 1);
    return u32(cx) + u32(d.x) * (u32(cy) + u32(d.y) * u32(cz));
}

fn solid_at(x: i32, y: i32, z: i32) -> bool {
    return solid[index(x, y, z)] != 0u;
}

fn density_at(p: vec3<f32>) -> f32 {
    let local = (p - sim.origin.xyz) / sim.spacing.xyz - vec3<f32>(0.5);
    let base = vec3<i32>(floor(local));
    let f = local - floor(local);
    var sum = 0.0;
    var wsum = 0.0;
    for (var dz = 0; dz < 2; dz = dz + 1) {
        for (var dy = 0; dy < 2; dy = dy + 1) {
            for (var dx = 0; dx < 2; dx = dx + 1) {
                let x = base.x + dx;
                let y = base.y + dy;
                let z = base.z + dz;
                if (solid_at(x, y, z)) {
                    continue;
                }
                let wx = select(1.0 - f.x, f.x, dx == 1);
                let wy = select(1.0 - f.y, f.y, dy == 1);
                let wz = select(1.0 - f.z, f.z, dz == 1);
                let w = wx * wy * wz;
                sum += state_in[index(x, y, z)].x * w;
                wsum += w;
            }
        }
    }
    return select(0.0, sum / wsum, wsum > 1.0e-4);
}

fn state_at(p: vec3<f32>) -> vec4<f32> {
    let local = (p - sim.origin.xyz) / sim.spacing.xyz - vec3<f32>(0.5);
    let base = vec3<i32>(floor(local));
    let f = local - floor(local);
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dz = 0; dz < 2; dz = dz + 1) {
        for (var dy = 0; dy < 2; dy = dy + 1) {
            for (var dx = 0; dx < 2; dx = dx + 1) {
                let x = base.x + dx;
                let y = base.y + dy;
                let z = base.z + dz;
                if (solid_at(x, y, z)) {
                    continue;
                }
                let wx = select(1.0 - f.x, f.x, dx == 1);
                let wy = select(1.0 - f.y, f.y, dy == 1);
                let wz = select(1.0 - f.z, f.z, dz == 1);
                let w = wx * wy * wz;
                sum += state_in[index(x, y, z)] * w;
                wsum += w;
            }
        }
    }
    return select(vec4<f32>(0.0), sum / wsum, wsum > 1.0e-4);
}

fn world_from_uv(uv: vec2<f32>, distance: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let near = near_h.xyz / near_h.w;
    let far = far_h.xyz / far_h.w;
    return frame.camera_pos.xyz + normalize(far - near) * distance;
}

fn sun_visibility(p: vec3<f32>) -> f32 {
    let dims3 = vec3<f32>(textureDimensions(probe_tex));
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let uvw = (p - frame.world_min.xyz) / size;
    let cell = clamp(vec3<i32>(uvw * dims3), vec3<i32>(0), vec3<i32>(dims3) - vec3<i32>(1));
    return clamp(textureLoad(probe_tex, cell, 0).a, 0.0, 1.0);
}

fn probe_ambient(p: vec3<f32>) -> vec3<f32> {
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let uvw = clamp((p - frame.world_min.xyz) / size, vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(probe_tex, linear_sampler, uvw, 0.0).rgb;
}

fn hash12(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(0.1031, 0.1030));
    q += dot(q, q.yx + 33.33);
    return fract((q.x + q.y) * q.x);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i.xy + vec2<f32>(i.z, 0.0));
    let b = hash12(i.xy + vec2<f32>(1.0, 0.0) + vec2<f32>(i.z, 0.0));
    let c = hash12(i.xy + vec2<f32>(0.0, 1.0) + vec2<f32>(i.z, 0.0));
    let d = hash12(i.xy + vec2<f32>(1.0, 1.0) + vec2<f32>(i.z, 0.0));
    let e = hash12(i.xy + vec2<f32>(i.z + 1.0, 0.0));
    let g = hash12(i.xy + vec2<f32>(1.0, 0.0) + vec2<f32>(i.z + 1.0, 0.0));
    let h = hash12(i.xy + vec2<f32>(0.0, 1.0) + vec2<f32>(i.z + 1.0, 0.0));
    let k = hash12(i.xy + vec2<f32>(1.0, 1.0) + vec2<f32>(i.z + 1.0, 0.0));
    return mix(
        mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
        mix(mix(e, g, u.x), mix(h, k, u.x), u.y),
        u.z,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(out_smoke);
    if (gid.x >= out_dims.x || gid.y >= out_dims.y) {
        return;
    }
    let pixel = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(out_dims);
    let view_dir = normalize(world_from_uv(uv, 1.0) - frame.camera_pos.xyz);
    let camera = frame.camera_pos.xyz;

    // Scene depth: smoke can never be composited in front of the surface.
    let full_dims = vec2<i32>(textureDimensions(depth_tex));
    let full_pixel = clamp(pixel * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
    let scene_distance = textureLoad(depth_tex, full_pixel, 0).r;

    let lo = sim.origin.xyz;
    let hi = lo + sim.spacing.xyz * vec3<f32>(sim.grid.xyz);
    let inv_dir = 1.0 / max(abs(view_dir), vec3<f32>(1.0e-5)) * sign(view_dir);
    let t0 = (lo - camera) * inv_dir;
    let t1 = (hi - camera) * inv_dir;
    let near_box = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let far_box = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    var start = max(near_box, 0.0);
    var end = far_box;
    if (scene_distance > 0.0) {
        end = min(end, scene_distance);
    }
    if (end <= start || far_box <= 0.0) {
        textureStore(out_smoke, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    // Never march more than the box diagonal: the ray is local.
    end = min(end, start + 32.0);

    let jitter = hash12(vec2<f32>(pixel) + vec2<f32>(frame.params.x * 0.37));
    let step_len = (end - start) / f32(STEPS);
    let sun = normalize(frame.sun_dir.xyz);
    let daylight = clamp(frame.sun_color.w, 0.0, 1.0);
    let sky = mix(vec3<f32>(0.02, 0.025, 0.04), vec3<f32>(0.55, 0.62, 0.72), daylight);
    let phase = 0.35 + 0.65 * pow(max(dot(view_dir, sun), 0.0), 8.0);

    var transmittance = 1.0;
    var inscatter = vec3<f32>(0.0);

    for (var i = 0; i < STEPS; i = i + 1) {
        let t = start + step_len * (f32(i) + jitter);
        let p = camera + view_dir * t;
        var state = state_at(p);
        // Fade density near the simulation box boundary. Without it the finite
        // box cut a visible cube out of a smoke cloud that reached the wall.
        let box_size = sim.spacing.xyz * vec3<f32>(sim.grid.xyz);
        let local01 = clamp((p - sim.origin.xyz) / max(box_size, vec3<f32>(0.001)), vec3<f32>(0.0), vec3<f32>(1.0));
        let edge = smoothstep(0.0, 0.06, local01.x) * smoothstep(0.0, 0.06, 1.0 - local01.x)
            * smoothstep(0.0, 0.06, local01.y) * smoothstep(0.0, 0.06, 1.0 - local01.y)
            * smoothstep(0.0, 0.06, local01.z) * smoothstep(0.0, 0.06, 1.0 - local01.z);
        state.x *= edge;
        state.y *= edge;
        // Break up the perfectly smooth grid density with two noise octaves.
        let wisp = 0.70
            + 0.36 * value_noise(p * 0.9 + vec3<f32>(frame.params.x * 0.15, 0.0, 0.0))
            + 0.20 * value_noise(p * 2.3 - vec3<f32>(0.0, frame.params.x * 0.22, 0.0));
        state.x *= clamp(wisp, 0.35, 1.5);
        let density = max(state.x, 0.0);
        if (density <= 1.0e-4) {
            continue;
        }

        // Beer self-shadowing towards the sun: five taps inside the box.
        var optical = 0.0;
        let sun_step = 0.75;
        for (var s = 1; s <= 5; s = s + 1) {
            optical += density_at(p + sun * (sun_step * f32(s))) * sun_step;
        }
        let beer = exp(-optical * EXTINCTION);

        let probe_vis = sun_visibility(p);
        let sun_term = frame.sun_color.rgb * (phase * probe_vis * beer);
        let ambient = probe_ambient(p) * 0.35 + sky * 0.10;
        var lit = vec3<f32>(ambient);

        // Lamps (and later the fire's temporary light).
        let count = i32(clamp(frame.params2.z, 0.0, 12.0));
        for (var l = 0; l < count; l = l + 1) {
            let lp = frame.light_pos_radius[l];
            let radius = max(lp.w, 0.001);
            let delta = lp.xyz - p;
            let d2 = dot(delta, delta);
            if (d2 > radius * radius * 2.25) {
                continue;
            }
            let normalized = sqrt(d2) / radius;
            let window = max(1.0 - normalized * normalized, 0.0);
            lit += frame.light_color[l].rgb * window * window / max(d2, 0.05) * beer;
        }

        // Soot first, pale ash later. The early cloud is dark and takes its
        // colour from the fire emission below.
        let age = clamp(state.z, 0.0, 1.0);
        let albedo = mix(vec3<f32>(0.045, 0.040, 0.036), vec3<f32>(0.52, 0.52, 0.52), age);
        var radiance = albedo * (lit + sun_term);
        let heat = clamp(state.y, 0.0, 1.5);
        // Blackbody-style ramp instead of one flat orange: soot-red at the edge,
        // orange in the body, yellow-white in the hot core. The noise flicker is
        // sampled in world space and drifts upward with the flow, so the flame
        // tongue moves with the plume instead of pulsing per screen pixel.
        let flicker = 0.82
            + 0.18 * value_noise(p * 2.7 - vec3<f32>(0.0, frame.params.x * 3.2, 0.0))
            + 0.10 * value_noise(p * 6.1 + vec3<f32>(frame.params.x * 1.7));
        let deep = vec3<f32>(0.55, 0.055, 0.008);
        let orange = vec3<f32>(1.0, 0.30, 0.035);
        let core = vec3<f32>(1.0, 0.86, 0.52);
        let body = mix(deep, orange, smoothstep(0.10, 0.62, heat));
        let flame_color = mix(body, core, smoothstep(0.62, 1.30, heat));
        radiance += flame_color * pow(heat, 2.2) * 3.2 * flicker;

        let sigma = density * EXTINCTION;
        let step_t = exp(-sigma * step_len);
        inscatter += transmittance * radiance * density * step_len * 2.0;
        transmittance *= step_t;
        if (transmittance < 0.01) {
            break;
        }
    }

    textureStore(out_smoke, pixel, vec4<f32>(inscatter, clamp(transmittance, 0.0, 1.0)));
}

