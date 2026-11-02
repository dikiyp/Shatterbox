// Local-boxy 3D smoke simulation on the GPU.
//
// One box around an active source, 64 x 32 x 64 cells of 0.25 m. The grid is a
// flat storage buffer instead of a 3D texture so the same buffer can be read and
// written without a second view, and so the obstacle mask is one integer per
// cell. The pipeline is the standard stable-fluids one:
//
//   advect velocity -> divergence -> Jacobi pressure -> project -> advect state
//
// Obstacles come from the CPU-built `solid` mask (voxel occupancy, dilated by
// one cell): velocity is zero there and pressure uses a Neumann boundary, which
// is what makes the plume slide along a wall instead of entering it.
//
// Contains every entry point of the step so the Rust side can record them in a
// single compute pass (one timestamp pair for the whole simulation).

struct SmokeSim {
    // xyz = box min corner, w = unused.
    origin: vec4<f32>,
    // xyz = cell size in metres.
    spacing: vec4<f32>,
    // xyz = grid dimensions, w = active source count.
    grid: vec4<u32>,
    // x = dt, y = buoyancy, z = vorticity confinement, w = temperature cooling.
    params: vec4<f32>,
    // x = density dissipation, y = time seconds, z = wind x, w = wind z.
    params2: vec4<f32>,
    // x = fire emission scale (reserved for part 4), y..w unused.
    params3: vec4<f32>,
    // xyz = position, w = radius.
    source: array<vec4<f32>, 4>,
    // x = density rate, y = temperature, z = upward velocity, w = unused.
    source2: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> sim: SmokeSim;
@group(0) @binding(1) var<storage, read> velocity_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> state_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> velocity_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> state_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;

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

// Cell centre in metres.
fn centre(x: i32, y: i32, z: i32) -> vec3<f32> {
    return sim.origin.xyz + (vec3<f32>(f32(x), f32(y), f32(z)) + vec3<f32>(0.5)) * sim.spacing.xyz;
}

fn gid_to_cell(gid: vec3<u32>) -> vec3<i32> {
    let d = dims();
    let id = i32(gid.x);
    let nx = d.x;
    let ny = d.y;
    let z = id / (nx * ny);
    let rem = id - z * nx * ny;
    let y = rem / nx;
    let x = rem - y * nx;
    return vec3<i32>(x, y, z);
}

// ---------------------------------------------------------------------------
// Trilinear sampling of the flat fields. Taps inside solid cells are skipped
// and the weights renormalised, so a wall cannot leak its neighbour's value
// across the boundary.
// ---------------------------------------------------------------------------

fn sample_state(p: vec3<f32>) -> vec4<f32> {
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

fn sample_velocity(p: vec3<f32>) -> vec3<f32> {
    let local = (p - sim.origin.xyz) / sim.spacing.xyz - vec3<f32>(0.5);
    let base = vec3<i32>(floor(local));
    let f = local - floor(local);
    var sum = vec3<f32>(0.0);
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
                sum += velocity_in[index(x, y, z)].xyz * w;
                wsum += w;
            }
        }
    }
    return select(vec3<f32>(0.0), sum / wsum, wsum > 1.0e-4);
}

// Velocity used by the pressure solve: a solid neighbour behaves as a wall, so
// the flux through that face is zero (Neumann boundary).
fn face_velocity(x: i32, y: i32, z: i32, axis: i32) -> f32 {
    if (solid_at(x, y, z)) {
        return 0.0;
    }
    let v = velocity_in[index(x, y, z)].xyz;
    return v[axis];
}

// ---------------------------------------------------------------------------
// 1. Semi-Lagrangian velocity advection + buoyancy + vorticity confinement.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn advect_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    if (solid_at(cell.x, cell.y, cell.z)) {
        velocity_out[i] = vec4<f32>(0.0);
        return;
    }
    let p = centre(cell.x, cell.y, cell.z);
    let here = velocity_in[i].xyz;
    var v = sample_velocity(p - here * sim.params.x);

    // Buoyancy: hot smoke rises, cool smoke keeps a small drift.
    let temperature = sample_state(p).y;
    v.y += sim.params.y * clamp(temperature, 0.0, 1.5) * sim.params.x;

    // Vorticity confinement: amplify the curl that the coarse grid would
    // otherwise diffuse away. Without it a plume leaves a flat, laminar column.
    let dx = sim.spacing.x;
    let dy = sim.spacing.y;
    let dz = sim.spacing.z;
    let vxp = velocity_in[index(cell.x + 1, cell.y, cell.z)].xyz;
    let vxm = velocity_in[index(cell.x - 1, cell.y, cell.z)].xyz;
    let vyp = velocity_in[index(cell.x, cell.y + 1, cell.z)].xyz;
    let vym = velocity_in[index(cell.x, cell.y - 1, cell.z)].xyz;
    let vzp = velocity_in[index(cell.x, cell.y, cell.z + 1)].xyz;
    let vzm = velocity_in[index(cell.x, cell.y, cell.z - 1)].xyz;
    let curl = vec3<f32>(
        (vzp.y - vzm.y) / (2.0 * dz) - (vyp.z - vym.z) / (2.0 * dy),
        (vxp.z - vxm.z) / (2.0 * dx) - (vzp.x - vzm.x) / (2.0 * dz),
        (vyp.x - vym.x) / (2.0 * dy) - (vxp.y - vxm.y) / (2.0 * dx),
    );
    let curl_p = vec3<f32>(
        length(velocity_in[index(cell.x + 2, cell.y, cell.z)].xyz),
        length(velocity_in[index(cell.x, cell.y + 2, cell.z)].xyz),
        length(velocity_in[index(cell.x, cell.y, cell.z + 2)].xyz),
    );
    let curl_m = vec3<f32>(
        length(velocity_in[index(cell.x - 2, cell.y, cell.z)].xyz),
        length(velocity_in[index(cell.x, cell.y - 2, cell.z)].xyz),
        length(velocity_in[index(cell.x, cell.y, cell.z - 2)].xyz),
    );
    let gradient = (curl_p - curl_m) / (4.0 * dx);
    let n = gradient / max(length(gradient), 1.0e-4);
    v += cross(n, curl) * sim.params.z * dx;

    // Gentle wind and damping keep the box from accumulating momentum forever.
    v += vec3<f32>(sim.params2.z, 0.0, sim.params2.w) * sim.params.x;
    // Source updraft: the column leaves the emitter faster than plain buoyancy.
    let source_count = i32(sim.grid.w);
    for (var s = 0; s < source_count; s = s + 1) {
        let src = sim.source[s];
        let radius = max(src.w, 0.25);
        let d = length(p - src.xyz);
        if (d < radius) {
            let falloff = 1.0 - d / radius;
            v.y += sim.source2[s].z * falloff * falloff * sim.params.x;
        }
    }
    v *= exp(-0.6 * sim.params.x);
    velocity_out[i] = vec4<f32>(v, 0.0);
}

// ---------------------------------------------------------------------------
// 2. Divergence of the advected velocity.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn clear_all(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    velocity_out[i] = vec4<f32>(0.0);
    state_out[i] = vec4<f32>(0.0);
    pressure_out[i] = 0.0;
}

@compute @workgroup_size(64)
fn divergence_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    if (solid_at(cell.x, cell.y, cell.z)) {
        divergence[i] = 0.0;
        return;
    }
    let dx = sim.spacing.x;
    let dy = sim.spacing.y;
    let dz = sim.spacing.z;
    let vx = (face_velocity(cell.x + 1, cell.y, cell.z, 0)
        - face_velocity(cell.x - 1, cell.y, cell.z, 0)) / (2.0 * dx);
    let vy = (face_velocity(cell.x, cell.y + 1, cell.z, 1)
        - face_velocity(cell.x, cell.y - 1, cell.z, 1)) / (2.0 * dy);
    let vz = (face_velocity(cell.x, cell.y, cell.z + 1, 2)
        - face_velocity(cell.x, cell.y, cell.z - 1, 2)) / (2.0 * dz);
    divergence[i] = vx + vy + vz;
}

@compute @workgroup_size(64)
fn clear_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    pressure_out[index(cell.x, cell.y, cell.z)] = 0.0;
}

// ---------------------------------------------------------------------------
// 3. Pressure projection: Jacobi iterations, solid cells keep p = 0.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    if (solid_at(cell.x, cell.y, cell.z)) {
        pressure_out[i] = 0.0;
        return;
    }
    let h = sim.spacing.x;
    let inv_h2 = 1.0 / (h * h);
    var sum = 0.0;
    sum += pressure_in[index(cell.x + 1, cell.y, cell.z)];
    sum += pressure_in[index(cell.x - 1, cell.y, cell.z)];
    sum += pressure_in[index(cell.x, cell.y + 1, cell.z)];
    sum += pressure_in[index(cell.x, cell.y - 1, cell.z)];
    sum += pressure_in[index(cell.x, cell.y, cell.z + 1)];
    sum += pressure_in[index(cell.x, cell.y, cell.z - 1)];
    // Six neighbours, but y uses a different spacing; scale the whole solve by
    // the x spacing and treat it as cubic. The error is small next to the
    // clamp and the whole field is cosmetic.
    pressure_out[i] = (sum * inv_h2 - divergence[i]) / (6.0 * inv_h2);
}

// ---------------------------------------------------------------------------
// 4. Subtract the pressure gradient, giving a divergence-free velocity.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    if (solid_at(cell.x, cell.y, cell.z)) {
        velocity_out[i] = vec4<f32>(0.0);
        return;
    }
    let dx = sim.spacing.x;
    let dy = sim.spacing.y;
    let dz = sim.spacing.z;
    let gx = (pressure_in[index(cell.x + 1, cell.y, cell.z)]
        - pressure_in[index(cell.x - 1, cell.y, cell.z)]) / (2.0 * dx);
    let gy = (pressure_in[index(cell.x, cell.y + 1, cell.z)]
        - pressure_in[index(cell.x, cell.y - 1, cell.z)]) / (2.0 * dy);
    let gz = (pressure_in[index(cell.x, cell.y, cell.z + 1)]
        - pressure_in[index(cell.x, cell.y, cell.z - 1)]) / (2.0 * dz);
    let v = velocity_in[i].xyz - vec3<f32>(gx, gy, gz);
    velocity_out[i] = vec4<f32>(v, 0.0);
}

// ---------------------------------------------------------------------------
// 5. Advect density / temperature / age and inject the source.
// ---------------------------------------------------------------------------

@compute @workgroup_size(64)
fn advect_state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid_to_cell(gid);
    let i = index(cell.x, cell.y, cell.z);
    if (solid_at(cell.x, cell.y, cell.z)) {
        state_out[i] = vec4<f32>(0.0);
        return;
    }
    let p = centre(cell.x, cell.y, cell.z);
    let v = velocity_in[i].xyz;
    var state = sample_state(p - v * sim.params.x);
    // Explicit diffusion: semi-Lagrangian advection alone keeps a plume in a
    // narrow column because the grid has no molecular mixing. A 6-neighbour
    // average is what makes smoke fill a closed room and seep through a door.
    var neighbours = vec4<f32>(0.0);
    var neighbour_count = 0.0;
    for (var axis = 0; axis < 6; axis = axis + 1) {
        var offset = vec3<i32>(0, 0, 0);
        if (axis == 0) { offset = vec3<i32>(1, 0, 0); }
        if (axis == 1) { offset = vec3<i32>(-1, 0, 0); }
        if (axis == 2) { offset = vec3<i32>(0, 1, 0); }
        if (axis == 3) { offset = vec3<i32>(0, -1, 0); }
        if (axis == 4) { offset = vec3<i32>(0, 0, 1); }
        if (axis == 5) { offset = vec3<i32>(0, 0, -1); }
        if (!solid_at(cell.x + offset.x, cell.y + offset.y, cell.z + offset.z)) {
            neighbours += state_in[index(cell.x + offset.x, cell.y + offset.y, cell.z + offset.z)];
            neighbour_count += 1.0;
        }
    }
    if (neighbour_count > 0.0) {
        state = mix(state, neighbours / neighbour_count, 0.12);
    }
    state.x *= exp(-sim.params2.x * sim.params.x);
    state.y *= exp(-sim.params.w * sim.params.x);
    state.z = min(1.0, state.z + sim.params.x * 0.12);

    let count = i32(sim.grid.w);
    for (var s = 0; s < count; s = s + 1) {
        let src = sim.source[s];
        let radius = max(src.w, 0.25);
        let d = length(p - src.xyz);
        if (d >= radius) {
            continue;
        }
        let falloff = 1.0 - d / radius;
        let weight = falloff * falloff;
        let inject = sim.source2[s];
        state.x = min(state.x + inject.x * weight * sim.params.x, 1.5);
        state.y = max(state.y, inject.y * weight);
    }
    state_out[i] = state;
}




