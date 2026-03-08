//! Inline WGSL shader sources for 3D graph visualization.

/// Common struct definitions shared across all shaders.
const COMMON_STRUCTS: &str = "
struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
};

struct NodeGpu {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct EdgeGpu {
    src: u32,
    dst: u32,
    flags: u32,
    _pad: u32,
};

// Node flag bits (must match types.rs constants)
const FLAG_START: u32 = 1u;
const FLAG_WIN: u32 = 2u;
const FLAG_DEAD: u32 = 4u;
const FLAG_HOVERED: u32 = 8u;

// Edge flag bits (must match types.rs constants)
const EDGE_FLAG_BIDI: u32 = 1u;
";

/// Build a complete shader source by prepending common structs.
fn shader(body: &str) -> String {
    format!("{COMMON_STRUCTS}\n{body}")
}

pub fn node_shader() -> String {
    shader(
        r#"
struct NodeInfo {
    color: vec4<f32>,
    flags: u32,
    bfs_depth: u32,
    radius: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> nodes: array<NodeGpu>;
@group(1) @binding(1) var<storage, read> node_info: array<NodeInfo>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_uv: vec2<f32>,
    @location(2) outline: f32,
    @location(3) is_current: f32,
};

@vertex
fn vs_node(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    let node = nodes[iid];
    let info = node_info[iid];

    // current_node_idx is packed into camera_right.w as bitcast u32
    let current_node_idx = bitcast<u32>(camera.camera_right.w);

    // Unit quad: 2 triangles, 6 vertices
    var quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let uv = quad[vid];

    // Billboard: offset in camera-local right/up directions
    let r = info.radius;
    let world = node.pos.xyz
        + camera.camera_right.xyz * (uv.x * r)
        + camera.camera_up.xyz * (uv.y * r);

    let clip = camera.view_proj * vec4<f32>(world, 1.0);

    var out: VsOut;
    out.pos = clip;
    out.color = info.color;
    out.local_uv = uv;
    // Outline for start or hovered nodes
    out.outline = select(0.0, 1.0, (info.flags & (FLAG_START | FLAG_HOVERED)) != 0u);
    out.is_current = select(0.0, 1.0, iid == current_node_idx);
    return out;
}

@fragment
fn fs_node(in: VsOut) -> @location(0) vec4<f32> {
    let dist = length(in.local_uv);
    let fw = fwidth(dist);

    // Pixel-perfect anti-aliased circle edge
    let aa = 1.0 - smoothstep(1.0 - fw, 1.0 + fw, dist);
    if aa <= 0.0 {
        discard;
    }
    var col = in.color;
    col.a *= aa;

    // White outline ring for start/hovered nodes
    if in.outline > 0.5 {
        let ring_inner = 0.7;
        let ring_outer = 0.85;
        let ring = smoothstep(ring_inner - fw, ring_inner + fw, dist)
                 * (1.0 - smoothstep(ring_outer - fw, ring_outer + fw, dist));
        col = mix(col, vec4<f32>(1.0, 1.0, 1.0, col.a), ring * 0.8);
    }

    // Blue outline ring for current game state node
    if in.is_current > 0.5 {
        let ring_inner = 0.6;
        let ring_outer = 0.8;
        let ring = smoothstep(ring_inner - fw, ring_inner + fw, dist)
                 * (1.0 - smoothstep(ring_outer - fw, ring_outer + fw, dist));
        col = mix(col, vec4<f32>(0.3, 0.7, 1.0, 1.0), ring * 0.9);
    }

    return col;
}
"#,
    )
}

pub fn edge_shader() -> String {
    shader(
        r#"
struct NodeInfoGpu {
    color: vec4<f32>,
    flags: u32,
    bfs_depth: u32,
    radius: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> nodes: array<NodeGpu>;
@group(1) @binding(1) var<storage, read> edges: array<EdgeGpu>;
@group(1) @binding(2) var<storage, read> highlights: array<u32>;
@group(1) @binding(3) var<storage, read> node_info: array<NodeInfoGpu>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) highlight: u32,
    @location(1) edge_uv: vec2<f32>,
    @location(2) edge_len: f32,
    @location(3) @interpolate(flat) is_bidi: u32,
};

@vertex
fn vs_edge(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    let edge = edges[iid];
    let raw_p0 = nodes[edge.src].pos.xyz;
    let raw_p1 = nodes[edge.dst].pos.xyz;
    let hl = highlights[iid];

    let raw_dir = raw_p1 - raw_p0;
    let raw_len = length(raw_dir);
    let fwd = select(vec3<f32>(1.0, 0.0, 0.0), raw_dir / raw_len, raw_len > 0.001);

    // Inset endpoints by node radii so edges don't overlap node billboards
    let r0 = node_info[edge.src].radius;
    let r1 = node_info[edge.dst].radius;
    let p0 = raw_p0 + fwd * r0;
    let p1 = raw_p1 - fwd * r1;
    let dir = p1 - p0;
    let len = max(length(dir), 0.0);

    // Camera forward = cross(right, up)
    let cam_fwd = cross(camera.camera_right.xyz, camera.camera_up.xyz);

    // Perpendicular in camera-facing plane
    var perp = cross(fwd, cam_fwd);
    let perp_len = length(perp);
    if perp_len < 0.001 {
        perp = camera.camera_up.xyz;
    } else {
        perp = perp / perp_len;
    }

    let is_bidi = (edge.flags & EDGE_FLAG_BIDI) != 0u;
    let any_highlight = hl != 0u;
    let line_thick = select(0.1, 0.2, any_highlight);
    let half_width = line_thick * 3.0;

    var quad = array<vec2<f32>, 6>(
        vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let q = quad[vid];
    let world = p0 + fwd * q.x * len + perp * q.y * half_width;

    let clip = camera.view_proj * vec4<f32>(world, 1.0);

    var out: VsOut;
    out.pos = clip;
    out.highlight = hl;
    out.edge_uv = q;
    out.edge_len = len;
    out.is_bidi = select(0u, 1u, is_bidi);
    return out;
}

// SDF for a rectangle centered at origin with half-extents (hx, hy).
// Returns negative inside, positive outside.
fn sd_box(p: vec2<f32>, h: vec2<f32>) -> f32 {
    let d = abs(p) - h;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

// SDF for an arrowhead triangle pointing in the +x direction.
// The triangle spans x: [arrow_start, 1.0], y: [-1, 1] at base tapering to 0 at tip.
// `t` is position along edge (0..1), `y` is signed lateral (-1..1).
fn sd_arrowhead(t: f32, y: f32, arrow_start: f32) -> f32 {
    let arrow_len = 1.0 - arrow_start;
    if arrow_len < 0.001 { return 1e6; }
    // Local coords: lx in [0, arrow_len], ly = y
    let lx = t - arrow_start;
    // The triangle boundary: |y| = 1.0 - lx/arrow_len, i.e. |y| + lx/arrow_len = 1
    // Signed distance to the triangle (isosceles, tip at right)
    // Edges: left (x=0), and two diagonal sides
    let progress = lx / arrow_len;  // 0 at base, 1 at tip
    let abs_y = abs(y);
    // Distance to the diagonal edge: |y| + progress <= 1 is inside
    // The diagonal normal (unnormalized): (1/arrow_len, 1) → normalized
    let diag_n = normalize(vec2(1.0 / arrow_len, 1.0));
    let diag_d = dot(vec2(lx, abs_y), diag_n) - dot(vec2(0.0, 1.0), diag_n);
    // Distance to left edge (x = arrow_start)
    let left_d = -lx;
    return max(diag_d, left_d);
}

@fragment
fn fs_edge(in: VsOut) -> @location(0) vec4<f32> {
    let t = in.edge_uv.x;  // 0 at src, 1 at dst
    let y = in.edge_uv.y;  // -1..1 across width

    // Fixed world-space arrowhead size
    let arrow_world_len = 0.5;
    let arrow_start = max(0.0, 1.0 - arrow_world_len / max(in.edge_len, 0.001));
    let line_half = 0.33;

    // Line body SDF: rectangle from body_start..arrow_start, half-height = line_half
    let body_start = select(0.0, 1.0 - arrow_start, in.is_bidi != 0u);
    let body_center_t = (body_start + arrow_start) * 0.5;
    let body_half_t = (arrow_start - body_start) * 0.5;
    let body_d = sd_box(vec2(t - body_center_t, y), vec2(body_half_t, line_half));

    // Forward arrowhead SDF (at dst end)
    let fwd_d = sd_arrowhead(t, y, arrow_start);

    // Reverse arrowhead SDF (at src end) — mirror t
    var rev_d = 1e6;
    if in.is_bidi != 0u {
        rev_d = sd_arrowhead(1.0 - t, y, arrow_start);
    }

    // Union: minimum distance
    let d = min(min(body_d, fwd_d), rev_d);

    // Anti-alias using screen-space derivatives
    let fw = fwidth(d);
    let alpha_shape = 1.0 - smoothstep(-fw, fw, d);
    if alpha_shape <= 0.0 {
        discard;
    }

    let base_color = vec4<f32>(0.5, 0.5, 0.5, 1.0);
    let hl_color = vec4<f32>(0.3, 0.7, 1.0, 1.0);
    var col = select(base_color, hl_color, in.highlight != 0u);
    col.a = alpha_shape;
    return col;
}
"#,
    )
}

pub fn hit_test_compute_shader() -> String {
    shader(
        r#"
struct HitTestParams {
    view_proj: mat4x4<f32>,
    cursor: vec2<f32>,
    half_size: vec2<f32>,
    rect_center: vec2<f32>,
    threshold_sq: f32,
    n_nodes: u32,
};

@group(0) @binding(0) var<storage, read> nodes: array<NodeGpu>;
@group(0) @binding(1) var<uniform> params: HitTestParams;
@group(0) @binding(2) var<storage, read_write> result: atomic<u32>;

@compute @workgroup_size(64)
fn cs_hit_test(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_nodes { return; }

    let pos = nodes[i].pos.xyz;
    let clip = params.view_proj * vec4<f32>(pos, 1.0);
    if clip.w <= 0.0 { return; }

    let ndc = clip.xy / clip.w;
    let screen = vec2<f32>(
        params.rect_center.x + ndc.x * params.half_size.x,
        params.rect_center.y - ndc.y * params.half_size.y,
    );

    let diff = screen - params.cursor;
    let dist_sq = dot(diff, diff);
    if dist_sq >= params.threshold_sq { return; }

    // Pack: upper 16 bits = quantized distance, lower 16 bits = node index.
    // atomicMin naturally picks the smallest distance, breaking ties by smallest index.
    let dist_quant = u32(clamp(dist_sq, 0.0, 65535.0));
    let packed = (dist_quant << 16u) | (i & 0xFFFFu);
    atomicMin(&result, packed);
}
"#,
    )
}

pub fn force_compute_shader() -> String {
    shader(
        r#"
struct SimParams {
    n_nodes: u32,
    n_edges: u32,
    repel: f32,
    attract: f32,
    decay: f32,
    speed_limit: f32,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> nodes: array<NodeGpu>;
@group(0) @binding(1) var<storage, read> edges: array<EdgeGpu>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn cs_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n_nodes {
        return;
    }

    var pos_i = nodes[i].pos.xyz;
    var vel_i = nodes[i].vel.xyz;
    var force = vec3<f32>(0.0, 0.0, 0.0);

    // All-pairs repulsion (softened inverse-square)
    for (var j = 0u; j < params.n_nodes; j++) {
        if j == i { continue; }
        let diff = pos_i - nodes[j].pos.xyz;
        let dist_sq = dot(diff, diff) + 1.0;
        force += normalize(diff) / (dist_sq * 10.0 + 2.0);
    }
    force *= params.repel;

    // Edge attraction (non-linear spring)
    for (var e = 0u; e < params.n_edges; e++) {
        let edge = edges[e];
        var neighbor_idx = 0xFFFFFFFFu;
        if edge.src == i {
            neighbor_idx = edge.dst;
        } else if edge.dst == i {
            neighbor_idx = edge.src;
        }
        if neighbor_idx == 0xFFFFFFFFu { continue; }

        let diff = nodes[neighbor_idx].pos.xyz - pos_i;
        let dist_sq = dot(diff, diff);
        let d6 = dist_sq * dist_sq * dist_sq * 0.05;
        let force_mag = (d6 - 1.0) / (d6 + 1.0) * 0.2 - 0.1;
        let d = length(diff);
        if d > 0.001 {
            force += (diff / d) * force_mag * params.attract;
        }
    }

    // Integration
    vel_i += force;
    let speed = length(vel_i);
    if speed > params.speed_limit {
        vel_i *= params.speed_limit / speed;
    }
    vel_i *= params.decay;
    pos_i += vel_i;

    nodes[i] = NodeGpu(vec4<f32>(pos_i, 0.0), vec4<f32>(vel_i, 0.0));
}
"#,
    )
}
