//! GPU buffer structs for the state graph visualization.

/// Per-node position and velocity (read/write by compute and vertex shaders).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeGpu {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
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

/// Camera transform: pan + zoom.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub pan: [f32; 2],
    pub zoom: f32,
    pub aspect: f32,
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

// Node flag bits (also used in WGSL shaders)
pub const FLAG_START: u32 = 1;
pub const FLAG_WIN: u32 = 2;
pub const FLAG_DEAD: u32 = 4;
#[allow(dead_code)]
pub const FLAG_HOVERED: u32 = 8;
