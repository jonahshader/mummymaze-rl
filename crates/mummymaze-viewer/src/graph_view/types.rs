//! GPU buffer structs for the state graph visualization.

/// Per-node position and velocity (read/write by compute and vertex shaders).
/// Uses vec4 (xyz + pad) to avoid vec3 alignment issues in storage buffers.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeGpu {
    pub pos: [f32; 4],
    pub vel: [f32; 4],
}

/// Per-node static info (read-only in vertex shader).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeInfo {
    pub color: [f32; 4],
    pub flags: u32,
    pub bfs_depth: u32,
    pub radius: f32,
    pub _pad: f32,
}

/// Edge defined by source and destination node indices.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeGpu {
    pub src: u32,
    pub dst: u32,
}

/// Camera transform for 3D rendering: view-projection matrix + billboard vectors.
/// `camera_right.w` carries the current node index (as f32-reinterpreted u32).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4], // 64 bytes
    pub camera_right: [f32; 4],   // 16 bytes (w = current_node_idx bits)
    pub camera_up: [f32; 4],      // 16 bytes (w unused)
}

/// Force simulation parameters.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub n_nodes: u32,
    pub n_edges: u32,
    pub repel: f32,
    pub attract: f32,
    pub decay: f32,
    pub speed_limit: f32,
    pub _pad: [f32; 2],
}

/// GPU hit-test parameters: cursor + projection info for the compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HitTestParams {
    pub view_proj: [[f32; 4]; 4], // 64 bytes
    pub cursor: [f32; 2],        // 8 bytes
    pub half_size: [f32; 2],     // 8 bytes
    pub rect_center: [f32; 2],   // 8 bytes
    pub threshold_sq: f32,       // 4 bytes
    pub n_nodes: u32,            // 4 bytes
} // Total: 96 bytes (16-byte aligned)

// Node flag bits (also used in WGSL shaders)
pub const FLAG_START: u32 = 1;
pub const FLAG_WIN: u32 = 2;
pub const FLAG_DEAD: u32 = 4;
#[allow(dead_code)]
pub const FLAG_HOVERED: u32 = 8;
