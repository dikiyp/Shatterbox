// Per-tile debris candidate lists, one thread per 8x8 pixel tile.
//
// Why this pass exists: the G-buffer used to walk every live debris instance per
// pixel. With ~250 pieces after one rocket that is 250 inverse-matrix transforms
// and slab tests per pixel, which measured 18 ms of a 27 ms G-buffer pass on the
// RTX 3050 at 1600x900 — the single biggest cost of "it lags after the shot".
// The pieces are also not spread over the screen: each one covers a few hundred
// pixels, so almost every pixel pays for ~240 pieces it can never hit.
//
// Here each tile collects the pieces whose conservative screen-space rectangle
// overlaps it, as a 512-bit mask (16 u32 words). The G-buffer then walks only the
// bits of its own tile, typically a handful of pieces.
//
// The rectangles are computed on the CPU (projected sphere AABB, full screen when
// the piece straddles the near plane), so this pass only does rectangle/tile
// overlap tests. Rejections are conservative: a piece that overlaps the tile is
// always kept, so the image is identical to the brute-force loop — this is a pure
// performance change, verified by comparing captures with `SHATTERBOX_DEBRIS_TILES=0`.

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
    // x = debris instance count (see `cast_debris` in gbuffer.wgsl).
    params2: vec4<f32>,
    params3: vec4<f32>,
    internal_size: vec4<f32>,
};

const TILE: u32 = 8u;
const WORDS: u32 = 16u;

@group(0) @binding(0) var<uniform> frame: Frame;
// xy = top-left pixel, zw = bottom-right pixel, inclusive.
@group(0) @binding(1) var<storage, read> rects: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> tiles: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(max(frame.internal_size.xy, vec2<f32>(1.0)));
    let tiles_x = (size.x + TILE - 1u) / TILE;
    let tiles_y = (size.y + TILE - 1u) / TILE;
    if (gid.x >= tiles_x || gid.y >= tiles_y) {
        return;
    }
    let tile_min = vec2<f32>(gid.xy * TILE);
    let tile_max = tile_min + vec2<f32>(f32(TILE));
    let count = min(u32(max(frame.params2.x, 0.0)), WORDS * 32u);

    var mask: array<u32, WORDS>;
    for (var word = 0u; word < WORDS; word = word + 1u) {
        mask[word] = 0u;
    }
    for (var piece = 0u; piece < count; piece = piece + 1u) {
        let rect = rects[piece];
        // Rects are inclusive and already padded by a pixel on the CPU, so a
        // piece touching the tile edge is kept.
        if (rect.z < tile_min.x || rect.x > tile_max.x || rect.w < tile_min.y || rect.y > tile_max.y) {
            continue;
        }
        let word = piece / 32u;
        mask[word] = mask[word] | (1u << (piece % 32u));
    }

    // One tile owns `WORDS` consecutive words, so the G-buffer indexes the same
    // buffer without a second indirection table.
    let base = (gid.y * tiles_x + gid.x) * WORDS;
    for (var word = 0u; word < WORDS; word = word + 1u) {
        tiles[base + word] = mask[word];
    }
}
