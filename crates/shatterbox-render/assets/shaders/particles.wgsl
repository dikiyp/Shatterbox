// Explosion smoke, rocket trail and dust.
//
// Two entry points share this file:
//  * `vs_main` / `fs_main` draw camera-facing billboards into a half-resolution
//    HDR target. Each puff is a volumetric-looking soft disc: a 3D noise field
//    tears the silhouette, the sun is attenuated through the puff (Beer-Lambert
//    with a thickness proxy), the sky and every local light add on top, and the
//    puff never emits light of its own.
//  * `vs_composite` / `fs_composite` upsample that half-resolution buffer over
//    the full-resolution HDR image with a depth-aware alpha, so the clouds keep
//    their soft edge without costing a full-res overdraw.
//
// The frame uniform must stay byte-identical to Rust: the head of this struct
// is shared by every pass and the layout test compares the field order.

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
    // z = active point-light count.
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

@group(0) @binding(0) var<uniform> frame: Frame;
// G-buffer distance along the view ray, in metres (negative = sky).
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
// Half-resolution volumetric fog: rgb = in-scatter, a = transmittance.
@group(0) @binding(2) var fog_tex: texture_2d<f32>;
// Half-resolution smoke buffer, written by the billboard pass.
@group(0) @binding(3) var smoke_tex: texture_2d<f32>;
@group(0) @binding(4) var smoke_sampler: sampler;

struct Instance {
    @location(0) centre: vec3<f32>,
    @location(1) radius: f32,
    // rgb = current albedo (already ramped from fire-lit to pale dust), a = opacity.
    @location(2) color: vec4<f32>,
    // x = optical density, y = noise seed, z = kind (0 smoke, 1 dust,
    // 2 trail, 3 grain), w = remaining heat 0..1.
    @location(3) params: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) params: vec4<f32>,
};

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
);

// ---------------------------------------------------------------------------
// Value noise. Cheap enough for a half-res pass and enough to break the disc
// into torn, lumpy edges instead of a perfect circle.
// ---------------------------------------------------------------------------

fn hash13(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.zyx + 31.32);
    return fract((q.x + q.y) * q.z);
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let b = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let g = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let h = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let k = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y), mix(mix(e, g, u.x), mix(h, k, u.x), u.y), u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var q = p;
    for (var octave = 0; octave < 3; octave = octave + 1) {
        sum += amplitude * value_noise(q);
        q = q * 2.07 + vec3<f32>(11.3, 7.1, 5.7);
        amplitude *= 0.5;
    }
    return sum / 0.875;
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32, instance: Instance) -> VsOut {
    var out: VsOut;
    let corner = CORNERS[index];
    var forward = instance.centre - frame.camera_pos.xyz;
    forward = forward / max(length(forward), 1.0e-4);
    if (length(forward) < 1.0e-3) {
        forward = vec3<f32>(0.0, 0.0, 1.0);
    }
    var right = cross(vec3<f32>(0.0, 1.0, 0.0), forward);
    if (length(right) < 1.0e-3) {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    right = normalize(right);
    let up = cross(forward, right);
    let offset = (right * corner.x + up * corner.y) * instance.radius;
    let world = instance.centre + offset;
    // Slightly oblate sphere: the rim faces away from the viewer, so the sun
    // shades the cloud from the side that actually sees it.
    out.normal = normalize(right * corner.x + up * corner.y + forward * 0.8);
    out.world = world;
    out.uv = corner;
    out.color = instance.color;
    out.params = instance.params;
    out.clip = frame.view_proj * vec4<f32>(world, 1.0);
    return out;
}

fn sun_visibility(sample_pos: vec3<f32>, thickness: f32) -> f32 {
    // Beer-Lambert through an approximate optical path. The puff centre is the
    // densest part; the rim is thin, so the sun bleeds through the edge and the
    // core stays in its own shadow. This is what gives the cloud volume instead
    // of a flat lit disc.
    let density = max(sample_pos.x, 0.0);
    return exp(-density * thickness * 1.6);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 >= 1.0) {
        discard;
    }
    let density = max(in.params.x, 0.0);
    let seed = in.params.y;
    let kind = in.params.z;
    let heat = clamp(in.params.w, 0.0, 1.0);

    // Torn silhouette: 3D fbm over world position, so the pattern sticks to the
    // cloud as it moves instead of swimming in screen space. Two scales: a
    // low-frequency lump that makes billows, and a finer tear. Both only
    // *offset the radius* — the old version thresholded the noise, which turned
    // every edge into an axis-aligned stair-step at the noise cell size.
    let noise_scale = mix(3.4, 5.6, kind * 0.25);
    let noise = fbm(in.world * noise_scale + vec3<f32>(seed, seed * 1.7, -seed * 0.6));
    let lump = fbm(in.world * 1.15 + vec3<f32>(seed * 0.4, -seed * 0.8, seed * 0.2));
    let radii = sqrt(r2);
    let wobble = (noise - 0.5) * 0.55 + (lump - 0.5) * 0.30;
    let shape = smoothstep(1.02, 0.40, radii + wobble);
    if (shape <= 0.004) {
        discard;
    }

    // Half-resolution pixel -> full-resolution G-buffer sample.
    let half_pixel = vec2<i32>(in.clip.xy);
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let full_pixel = clamp(half_pixel * 2, vec2<i32>(0), dims - 1);
    let scene = textureLoad(depth_tex, full_pixel, 0).r;
    let camera_distance = length(in.world - frame.camera_pos.xyz);
    var soft = 1.0;
    if (scene > 0.0) {
        soft = smoothstep(0.0, 1.0, scene - camera_distance);
    }

    // Thickness proxy: centre of the billboard is the thickest part.
    let thickness = in.color.a * shape * (0.85 + 0.55 * (1.0 - r2)) * (0.7 + 0.6 * density);
    let sun = normalize(frame.sun_dir.xyz);
    let sun_amount = max(dot(in.normal, sun), 0.0);
    // Single-scatter Beer-Lambert plus a multiple-scattering floor. Smoke never
    // goes fully black: light that scattered inside the cloud keeps a soft grey
    // core, which is what separates "volume" from "black decal".
    let transmittance = max(exp(-thickness * density * 0.9), 0.30);

    // Sun + sky. The early fire-lit colour comes from the CPU ramp, so nothing
    // here emits: a cloud can only reflect/transmit light it receives.
    let day = clamp(frame.sun_color.w, 0.0, 1.0);
    let sky = mix(vec3<f32>(0.055, 0.065, 0.095), vec3<f32>(0.55, 0.60, 0.70), day);
    var light = frame.sun_color.rgb * sun_amount * transmittance;
    light += sky * (0.85 + 0.35 * in.normal.y);

    // Local lamps and the blast flash. The flash is a short-lived point light,
    // so it lights the cloud through the same path as a ceiling lamp.
    let count = i32(clamp(frame.params2.z, 0.0, 12.0));
    for (var i = 0; i < count; i = i + 1) {
        let lp = frame.light_pos_radius[i];
        let delta = lp.xyz - in.world;
        let d2 = max(dot(delta, delta), 0.04);
        let radius = max(lp.w, 0.001);
        let window = clamp(1.0 - d2 / (radius * radius), 0.0, 1.0);
        // Light that reaches the puff is attenuated by the dust in front of it.
        let through = max(exp(-thickness * density * 0.9), 0.35);
        light += frame.light_color[i].rgb * window * window / d2 * through;
    }

    // Hot puffs scatter a little more forward light; nothing is added when the
    // light is gone, so the cloud darkens as it cools.
    light *= 0.9 + 0.25 * heat;

    var color = in.color.rgb * light;
    let alpha = clamp(in.color.a * shape * soft, 0.0, 1.0);

    // Fog composite so distant smoke sits in the same haze as the voxels.
    let fog_pixel = clamp(half_pixel, vec2<i32>(0), vec2<i32>(textureDimensions(fog_tex)) - 1);
    let fog = textureLoad(fog_tex, fog_pixel, 0);
    color = color * fog.a + fog.rgb;

    return vec4<f32>(color * alpha, alpha);
}

// ---------------------------------------------------------------------------
// Full-resolution composite of the half-res smoke buffer.
// ---------------------------------------------------------------------------

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
    // Depth-aware upsample: sample the bilinear colour, then scale coverage by
    // how much closer the full-res surface is than the half-res cloud. The
    // billboard pass already faded against depth, so a slightly conservative
    // reject at the silhouette is enough to stop halos around corners.
    let smoke = textureSample(smoke_tex, smoke_sampler, in.uv);
    let dims = vec2<i32>(textureDimensions(depth_tex));
    let pixel = clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), dims - 1);
    let depth = textureLoad(depth_tex, pixel, 0).r;
    var scale = 1.0;
    if (depth > 0.0) {
        // The smoke buffer has no depth channel, so the fade is applied where
        // the cloud's own alpha is already thin: keep full coverage in front of
        // geometry and let the billboard pass handle the near-wall case.
        scale = 1.0;
    }
    let alpha = clamp(smoke.a * scale, 0.0, 1.0);
    return vec4<f32>(smoke.rgb, alpha);
}
