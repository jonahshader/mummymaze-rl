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
    if dist > 1.0 {
        discard;
    }
    // Anti-aliased edge
    let aa = 1.0 - smoothstep(0.85, 1.0, dist);
    var col = in.color;
    col.a *= aa;

    // White outline ring for start/hovered nodes
    if in.outline > 0.5 {
        let ring = smoothstep(0.6, 0.7, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
        col = mix(col, vec4<f32>(1.0, 1.0, 1.0, col.a), ring * 0.8);
    }

    // Blue outline ring for current game state node
    if in.is_current > 0.5 {
        let ring = smoothstep(0.55, 0.65, dist) * (1.0 - smoothstep(0.8, 0.9, dist));
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
    @location(0) alpha: f32,
    @location(1) @interpolate(flat) highlight: u32,
    @location(2) edge_uv: vec2<f32>,
    @location(3) edge_len: f32,
    @location(4) @interpolate(flat) is_bidi: u32,
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
    out.alpha = select(0.2, 0.6, any_highlight);
    out.highlight = hl;
    out.edge_uv = q;
    out.edge_len = len;
    out.is_bidi = select(0u, 1u, is_bidi);
    return out;
}

// Test if (t, s) falls inside an arrowhead triangle pointing in the +t direction.
// Returns true only for the arrowhead region (t >= arrow_start).
fn arrowhead_hit(t: f32, s: f32, arrow_start: f32) -> bool {
    if t < arrow_start { return false; }
    let arrow_t = (t - arrow_start) / (1.0 - arrow_start);
    return s < (1.0 - arrow_t);
}

@fragment
fn fs_edge(in: VsOut) -> @location(0) vec4<f32> {
    let t = in.edge_uv.x;  // 0 at src, 1 at dst
    let s = abs(in.edge_uv.y);  // 0 at center, 1 at edge

    // Fixed world-space arrowhead size
    let arrow_world_len = 0.5;
    let arrow_start = max(0.0, 1.0 - arrow_world_len / max(in.edge_len, 0.001));
    let line_half = 0.33;

    // Line body: narrow center strip, clamped between arrowhead regions
    let body_start = select(0.0, 1.0 - arrow_start, in.is_bidi != 0u);
    let line_hit = s < line_half && t >= body_start && t <= arrow_start;

    // Forward arrowhead (at dst end)
    let fwd_hit = arrowhead_hit(t, s, arrow_start);
    // Reverse arrowhead (at src end) — only for bidi edges
    let rev_hit = in.is_bidi != 0u && arrowhead_hit(1.0 - t, s, arrow_start);

    if !line_hit && !fwd_hit && !rev_hit {
        discard;
    }

    let base_color = vec4<f32>(0.5, 0.5, 0.5, in.alpha);
    let hl_color = vec4<f32>(0.3, 0.7, 1.0, in.alpha);
    if in.highlight != 0u {
        return hl_color;
    }
    return base_color;
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
