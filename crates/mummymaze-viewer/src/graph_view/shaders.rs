//! Inline WGSL shader sources for graph visualization.

/// Common struct definitions shared across all shaders.
const COMMON_STRUCTS: &str = "
struct CameraUniform {
    pan: vec2<f32>,
    zoom: f32,
    aspect: f32,
};

struct NodeGpu {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct EdgeGpu {
    src: u32,
    dst: u32,
};

// Node flag bits (must match types.rs constants)
const FLAG_START: u32 = 1u;
const FLAG_WIN: u32 = 2u;
const FLAG_DEAD: u32 = 4u;
const FLAG_HOVERED: u32 = 8u;
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
};

@vertex
fn vs_node(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    let node = nodes[iid];
    let info = node_info[iid];

    // Unit quad: 2 triangles, 6 vertices
    var quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let uv = quad[vid];

    let r = info.radius;
    let world = node.pos + uv * r;

    // Camera transform: world -> clip
    let clip = vec2<f32>(
        (world.x - camera.pan.x) * camera.zoom / camera.aspect,
        (world.y - camera.pan.y) * camera.zoom,
    );

    var out: VsOut;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.color = info.color;
    out.local_uv = uv;
    // Outline for start or hovered nodes
    out.outline = select(0.0, 1.0, (info.flags & (FLAG_START | FLAG_HOVERED)) != 0u);
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

    // White outline ring for flagged nodes
    if in.outline > 0.5 {
        let ring = smoothstep(0.6, 0.7, dist) * (1.0 - smoothstep(0.85, 1.0, dist));
        col = mix(col, vec4<f32>(1.0, 1.0, 1.0, col.a), ring * 0.8);
    }

    return col;
}
"#,
    )
}

pub fn edge_shader() -> String {
    shader(
        r#"
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> nodes: array<NodeGpu>;
@group(1) @binding(1) var<storage, read> edges: array<EdgeGpu>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) alpha: f32,
};

@vertex
fn vs_edge(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
    let edge = edges[iid];
    let p0 = nodes[edge.src].pos;
    let p1 = nodes[edge.dst].pos;

    let dir = p1 - p0;
    let len = length(dir);
    let fwd = select(vec2(1.0, 0.0), dir / len, len > 0.001);
    let perp = vec2(-fwd.y, fwd.x);

    // Thin quad along the edge
    let thickness = 0.1;
    var quad = array<vec2<f32>, 6>(
        vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0),  vec2(1.0, -1.0), vec2(1.0, 1.0),
    );
    let q = quad[vid];
    let world = p0 + fwd * q.x * len + perp * q.y * thickness;

    let clip = vec2<f32>(
        (world.x - camera.pan.x) * camera.zoom / camera.aspect,
        (world.y - camera.pan.y) * camera.zoom,
    );

    var out: VsOut;
    out.pos = vec4<f32>(clip, 0.1, 1.0); // z=0.1: behind nodes
    out.alpha = 0.2;
    return out;
}

@fragment
fn fs_edge(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.5, 0.5, in.alpha);
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

    var pos_i = nodes[i].pos;
    var vel_i = nodes[i].vel;
    var force = vec2<f32>(0.0, 0.0);

    // All-pairs repulsion (softened inverse-square)
    for (var j = 0u; j < params.n_nodes; j++) {
        if j == i { continue; }
        let diff = pos_i - nodes[j].pos;
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

        let diff = nodes[neighbor_idx].pos - pos_i;
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

    nodes[i] = NodeGpu(pos_i, vel_i);
}
"#,
    )
}
