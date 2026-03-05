use eframe::egui;
use std::sync::Arc;

use super::math::PanZoomCamera;
use super::types::HitTestParams;
use super::GraphView;
use super::NodeKind;

impl GraphView {
    pub(super) fn handle_interaction(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let rect = response.rect;

        // Detect clicks for node navigation
        if response.clicked() {
            self.click_pending = true;
        }

        if self.layout_mode.is_3d() {
            self.handle_3d_interaction(ui, response, rect);
        } else {
            self.handle_2d_interaction(ui, response, rect);
        }

        // Process click
        if self.click_pending {
            self.click_pending = false;
            if let Some(hovered) = self.hovered_node {
                self.clicked_node = Some(hovered);
            }
        }
    }

    fn handle_3d_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // Orbit: left-drag (trackball — rotate around camera-local axes)
        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            let right = self.cam_3d.right();
            let up = self.cam_3d.up();
            let rot_x = glam::Quat::from_axis_angle(up, -delta.x * 0.005);
            let rot_y = glam::Quat::from_axis_angle(right, -delta.y * 0.005);
            self.cam_3d.orientation = (rot_x * rot_y * self.cam_3d.orientation).normalize();
            self.auto_follow = false;
        }

        // Pan: right-drag or middle-drag
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            let speed = self.cam_3d.distance * 0.002;
            let r = self.cam_3d.right();
            let u = self.cam_3d.up();
            self.cam_3d.target -= r * delta.x * speed - u * delta.y * speed;
            self.auto_follow = false;
        }

        // Dolly: scroll wheel
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            self.cam_3d.distance *= (-scroll * 0.003).exp();
            self.cam_3d.distance = self.cam_3d.distance.clamp(0.1, 10000.0);
        }

        // Hover hit test — store cursor/rect info; view_proj is filled in draw()
        if let Some(cursor) = response.hover_pos() {
            if let Some(buffers) = &self.buffers {
                self.pending_hit_test = Some(HitTestParams {
                    view_proj: [[0.0; 4]; 4], // filled by draw() before dispatch
                    cursor: [cursor.x, cursor.y],
                    half_size: [rect.width() / 2.0, rect.height() / 2.0],
                    rect_center: [rect.center().x, rect.center().y],
                    threshold_sq: 100.0,
                    n_nodes: buffers.n_nodes,
                });
            }
        } else {
            self.pending_hit_test = None;
            self.hovered_node = None;
        }
    }

    fn handle_2d_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &egui::Response,
        rect: egui::Rect,
    ) {
        // Pan: any drag
        if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
        {
            let delta = response.drag_delta();
            self.cam_2d.pan[0] -=
                delta.x * self.cam_2d.aspect / (self.cam_2d.zoom * rect.width() / 2.0);
            self.cam_2d.pan[1] += delta.y / (self.cam_2d.zoom * rect.height() / 2.0);
            self.auto_follow = false;
        }

        // Zoom toward cursor
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            let factor = (scroll * 0.005).exp();
            if let Some(cursor) = response.hover_pos() {
                self.cam_2d.zoom_at(
                    factor,
                    [cursor.x, cursor.y],
                    [rect.center().x, rect.center().y],
                    [rect.width() / 2.0, rect.height() / 2.0],
                );
            } else {
                self.cam_2d.zoom *= factor;
                self.cam_2d.zoom = self.cam_2d.zoom.clamp(0.001, 100.0);
            }
        }

        // Hover hit test (world-space, same as old 2D approach)
        self.hovered_node = None;
        if let Some(cursor) = response.hover_pos() {
            let clip = PanZoomCamera::screen_to_clip(
                [cursor.x, cursor.y],
                [rect.center().x, rect.center().y],
                [rect.width() / 2.0, rect.height() / 2.0],
            );
            let world = self.cam_2d.clip_to_world(clip, self.cam_2d.zoom);

            let hit_r_sq = 1.0f32;
            let mut best_dist_sq = f32::MAX;
            for (i, pos) in self.initial_positions.iter().enumerate() {
                let dx = pos[0] - world[0];
                let dy = pos[1] - world[1];
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < hit_r_sq && dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    self.hovered_node = Some(i);
                }
            }
        }
    }

    pub(super) fn draw_hover_tooltip(&self, _ui: &egui::Ui, response: &egui::Response) {
        let Some(node_idx) = self.hovered_node else {
            return;
        };
        let Some(ref level) = self.level else {
            return;
        };
        let Some(&kind) = self.node_kinds.get(node_idx) else {
            return;
        };

        // Get the state for this node (parent state for terminals)
        let state = match self.idx_to_state.get(node_idx) {
            Some(Some(s)) => *s,
            _ => return,
        };

        let label = match kind {
            NodeKind::Win => Some("WIN"),
            NodeKind::Dead => Some("DEAD"),
            NodeKind::Transient => None,
        };

        // Win% is metric index 0 (WinProb), get raw value for tooltip
        let win_prob = self
            .node_metrics
            .first()
            .and_then(|v| v.get(node_idx))
            .copied()
            .unwrap_or(0.0);
        let level = Arc::clone(level);

        response.clone().on_hover_ui(move |ui: &mut egui::Ui| {
            if let Some(l) = label {
                ui.label(l);
            }
            ui.label(format!("Win prob: {:.1}%", win_prob * 100.0));

            let size = egui::Vec2::new(200.0, 200.0);
            let (resp, painter) = ui.allocate_painter(size, egui::Sense::hover());
            crate::render::draw_maze_state(&painter, resp.rect, &level, &state);
        });
    }
}
