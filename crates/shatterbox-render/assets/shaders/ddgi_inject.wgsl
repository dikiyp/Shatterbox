// DDGI albedo injection: splats the visible G-buffer surface albedo into a
// coarse 1 m voxel volume.
//
// The probe rays in `ddgi_update.wgsl` need to know what colour a wall is.
// Brick occupancy only says "solid", and the palette lives in a storage buffer
// the probe shader does not bind, so the cheapest correct-enough source is the
// frame we already shaded: every visible pixel writes its albedo into the cell
// it occupies. Over a few frames the volume fills in the surfaces the camera
// can see, and the probe trace reads coloured bounce from it.
//
// Runs at half internal resolution: probe irradiance is low frequency, and a
// 2x2 splat costs a quarter of the writes. Same-cell races are harmless (the
// values are already similar); the volume is only a colour estimate.
//
// rgb = albedo, a = 1 + emissive (a <= 0.5 means "unknown").

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
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var albedo_tex: texture_2d<f32>;
@group(0) @binding(4) var material_tex: texture_2d<f32>;
@group(0) @binding(5) var out_volume: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_volume);
    let full_dims = vec2<i32>(textureDimensions(depth_tex));
    let half_dims = vec2<u32>(full_dims) / 2u;
    if (gid.x >= half_dims.x || gid.y >= half_dims.y) {
        return;
    }
    let pixel = clamp(vec2<i32>(gid.xy) * 2, vec2<i32>(0), full_dims - vec2<i32>(1));
    let distance = textureLoad(depth_tex, pixel, 0).r;
    if (distance < 0.0) {
        return;
    }
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(full_dims);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near_h = frame.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let far_h = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let rd = normalize(far_h.xyz / far_h.w - near_h.xyz / near_h.w);
    let world = frame.camera_pos.xyz + rd * distance;

    let albedo = textureLoad(albedo_tex, pixel, 0).rgb;
    let emissive = clamp(textureLoad(material_tex, pixel, 0).g * 8.0, 0.0, 8.0);
    let size = max(frame.world_max.xyz - frame.world_min.xyz, vec3<f32>(1.0));
    let cell = vec3<i32>(floor((world - frame.world_min.xyz) / size * vec3<f32>(dims)));
    let clamped = clamp(cell, vec3<i32>(0), vec3<i32>(dims) - vec3<i32>(1));
    textureStore(out_volume, clamped, vec4<f32>(albedo, 1.0 + emissive));
}
