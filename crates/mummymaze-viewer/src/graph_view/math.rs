//! Camera types for graph visualization: 2D pan/zoom and 3D orbital.

use glam::{Mat4, Vec3, Vec4};

use super::types::CameraUniform;

// ---------------------------------------------------------------------------
// 2D pan/zoom camera (for BFS Layers, Radial Tree)
// ---------------------------------------------------------------------------

/// 2D camera: orthographic projection in the XY plane, looking along -Z.
/// Zoom-toward-cursor support via `zoom_at()`.
pub struct PanZoomCamera {
    pub pan: [f32; 2],
    pub zoom: f32,
    pub aspect: f32,
}

impl PanZoomCamera {
    pub fn new() -> Self {
        PanZoomCamera {
            pan: [0.0, 0.0],
            zoom: 0.1,
            aspect: 1.0,
        }
    }

    /// Convert screen position to clip-space coordinates (Y-up).
    pub fn screen_to_clip(pos: [f32; 2], rect_center: [f32; 2], half_size: [f32; 2]) -> [f32; 2] {
        [
            (pos[0] - rect_center[0]) / half_size[0],
            -(pos[1] - rect_center[1]) / half_size[1],
        ]
    }

    /// Convert clip-space to world XY at a given zoom level.
    pub fn clip_to_world(&self, clip: [f32; 2], zoom: f32) -> [f32; 2] {
        [
            clip[0] * self.aspect / zoom + self.pan[0],
            clip[1] / zoom + self.pan[1],
        ]
    }

    /// Zoom toward a cursor position (in screen coords), keeping the world point under it fixed.
    pub fn zoom_at(
        &mut self,
        factor: f32,
        cursor: [f32; 2],
        rect_center: [f32; 2],
        half_size: [f32; 2],
    ) {
        let clip = Self::screen_to_clip(cursor, rect_center, half_size);
        let old_zoom = self.zoom;
        self.zoom *= factor;
        self.zoom = self.zoom.clamp(0.001, 100.0);
        let world_before = self.clip_to_world(clip, old_zoom);
        let world_after = self.clip_to_world(clip, self.zoom);
        self.pan[0] += world_before[0] - world_after[0];
        self.pan[1] += world_before[1] - world_after[1];
    }

    /// Build the GPU uniform. Produces an orthographic view_proj that maps
    /// world XY through pan/zoom/aspect, with billboard vectors = +X, +Y.
    pub fn to_uniform(&self) -> CameraUniform {
        // Build a view-projection that replicates the old 2D shader math:
        //   clip.x = (world.x - pan.x) * zoom / aspect
        //   clip.y = (world.y - pan.y) * zoom
        //   clip.z = 0  (flat)
        // As a matrix (column-major for wgpu):
        let sx = self.zoom / self.aspect;
        let sy = self.zoom;
        let tx = -self.pan[0] * sx;
        let ty = -self.pan[1] * sy;

        let view_proj = Mat4::from_cols(
            Vec4::new(sx, 0.0, 0.0, 0.0),
            Vec4::new(0.0, sy, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(tx, ty, 0.0, 1.0),
        );

        CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_right: [1.0, 0.0, 0.0, 0.0],
            camera_up: [0.0, 1.0, 0.0, 0.0],
        }
    }

    /// Fit camera to a 2D bounding box.
    pub fn fit(&mut self, positions: &[[f32; 3]]) {
        if positions.is_empty() {
            return;
        }
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for p in positions {
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }
        self.pan = [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0];
        let range_x = (max_x - min_x).max(1.0);
        let range_y = (max_y - min_y).max(1.0);
        let range = range_x.max(range_y) * 1.2;
        self.zoom = 2.0 / range;
    }
}

// ---------------------------------------------------------------------------
// 3D orbital camera (for force-directed layout)
// ---------------------------------------------------------------------------

/// Orbital camera with spherical coordinates around a target point.
pub struct OrbitalCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl OrbitalCamera {
    pub fn new() -> Self {
        OrbitalCamera {
            yaw: 0.0,
            pitch: 0.3,
            distance: 30.0,
            target: Vec3::ZERO,
            fov_y: std::f32::consts::FRAC_PI_4,
            aspect: 1.0,
            near: 0.1,
            far: 10000.0,
        }
    }

    /// Eye position from spherical coordinates.
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target
            + Vec3::new(
                self.distance * cp * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * cp * self.yaw.cos(),
            )
    }

    /// Camera right vector (world space, in XZ plane).
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// Camera up vector (world space, accounting for pitch).
    pub fn up(&self) -> Vec3 {
        let r = self.right();
        let fwd = (self.target - self.eye()).normalize_or_zero();
        r.cross(fwd)
    }

    /// Build the view-projection matrix and billboard vectors for the GPU.
    pub fn to_uniform(&self) -> CameraUniform {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far);
        let view_proj = proj * view;

        let r = self.right();
        let u = self.up();

        CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_right: [r.x, r.y, r.z, 0.0],
            camera_up: [u.x, u.y, u.z, 0.0],
        }
    }

    /// Fit camera to bounding sphere of 3D positions.
    pub fn fit(&mut self, positions: &[[f32; 3]]) {
        if positions.is_empty() {
            return;
        }
        let mut centroid = Vec3::ZERO;
        let n = positions.len() as f32;
        for p in positions {
            centroid += Vec3::from_array(*p);
        }
        centroid /= n;

        let mut max_r_sq = 0.0f32;
        for p in positions {
            max_r_sq = max_r_sq.max((Vec3::from_array(*p) - centroid).length_squared());
        }
        let radius = max_r_sq.sqrt().max(1.0);

        self.target = centroid;
        self.distance = radius / (self.fov_y / 2.0).sin() * 1.3;
        self.yaw = 0.0;
        self.pitch = 0.3;
    }
}
