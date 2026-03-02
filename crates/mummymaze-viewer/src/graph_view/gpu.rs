//! wgpu pipeline creation, buffer management, and CallbackTrait impl.

use super::shaders;
use super::types::*;
use eframe::egui;
use eframe::wgpu;
use egui_wgpu::{CallbackTrait, RenderState};
use std::sync::Arc;

/// Device-level GPU objects created once and reused across level changes.
pub struct GraphPipelines {
    pub node_pipeline: wgpu::RenderPipeline,
    pub edge_pipeline: wgpu::RenderPipeline,
    pub force_pipeline: wgpu::ComputePipeline,
    pub hit_test_pipeline: wgpu::ComputePipeline,
    pub camera_bgl: wgpu::BindGroupLayout,
    /// Shared layout for node render bind group (2x storage read-only).
    pub storage_ro_2_bgl: wgpu::BindGroupLayout,
    /// Layout for edge render bind group (3x storage read-only: nodes, edges, highlights).
    pub edge_render_bgl: wgpu::BindGroupLayout,
    pub compute_bgl: wgpu::BindGroupLayout,
    pub hit_test_bgl: wgpu::BindGroupLayout,
}

/// Per-level GPU buffers and bind groups, recreated when the graph changes.
pub struct GraphBuffers {
    pub node_buf: wgpu::Buffer,
    pub node_info_buf: wgpu::Buffer,
    pub edge_buf: wgpu::Buffer,
    pub edge_highlight_buf: wgpu::Buffer,
    pub camera_buf: wgpu::Buffer,
    pub sim_params_buf: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub node_render_bind_group: wgpu::BindGroup,
    pub edge_render_bind_group: wgpu::BindGroup,
    pub compute_bind_group: wgpu::BindGroup,
    // Hit-test compute resources
    pub hit_test_params_buf: wgpu::Buffer,
    pub hit_test_result_buf: wgpu::Buffer,
    pub hit_test_staging_buf: wgpu::Buffer,
    pub hit_test_bind_group: wgpu::BindGroup,
    // Tracked node readback
    pub tracked_node_staging_buf: wgpu::Buffer,
    pub n_nodes: u32,
    pub n_edges: u32,
}

/// Depth stencil state shared by both render pipelines.
fn depth_stencil() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

impl GraphPipelines {
    pub fn new(render_state: &RenderState) -> Self {
        let device = &render_state.device;

        // --- Bind group layouts ---
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let storage_ro_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let storage_ro_2_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("storage_ro_2_bgl"),
                entries: &[storage_ro_entry(0), storage_ro_entry(1)],
            });

        // Edge render: nodes(0), edges(1), highlights(2) — all storage read-only
        let edge_render_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("edge_render_bgl"),
                entries: &[
                    storage_ro_entry(0),
                    storage_ro_entry(1),
                    storage_ro_entry(2),
                ],
            });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Hit test: nodes(storage RO), params(uniform), result(storage RW)
        let hit_test_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hit_test_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // --- Shaders ---
        let node_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("node_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::node_shader().into()),
        });

        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::edge_shader().into()),
        });

        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("force_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::force_compute_shader().into()),
        });

        // --- Pipelines ---
        let target_format = render_state.target_format;
        let blend_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let node_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("node_pipeline_layout"),
                bind_group_layouts: &[&camera_bgl, &storage_ro_2_bgl],
                push_constant_ranges: &[],
            });

        let node_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("node_pipeline"),
            layout: Some(&node_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &node_shader,
                entry_point: Some("vs_node"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &node_shader,
                entry_point: Some("fs_node"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let edge_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("edge_pipeline_layout"),
                bind_group_layouts: &[&camera_bgl, &edge_render_bgl],
                push_constant_ranges: &[],
            });

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_pipeline"),
            layout: Some(&edge_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_edge"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_edge"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(blend_state),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("force_pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &force_shader,
            entry_point: Some("cs_force"),
            compilation_options: Default::default(),
            cache: None,
        });

        let hit_test_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hit_test_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::hit_test_compute_shader().into()),
        });

        let hit_test_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hit_test_pipeline_layout"),
                bind_group_layouts: &[&hit_test_bgl],
                push_constant_ranges: &[],
            });

        let hit_test_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("hit_test_pipeline"),
                layout: Some(&hit_test_pipeline_layout),
                module: &hit_test_shader,
                entry_point: Some("cs_hit_test"),
                compilation_options: Default::default(),
                cache: None,
            });

        GraphPipelines {
            node_pipeline,
            edge_pipeline,
            force_pipeline,
            hit_test_pipeline,
            camera_bgl,
            storage_ro_2_bgl,
            edge_render_bgl,
            compute_bgl,
            hit_test_bgl,
        }
    }
}

impl GraphBuffers {
    pub fn new(
        device: &wgpu::Device,
        pipelines: &GraphPipelines,
        n_nodes: u32,
        n_edges: u32,
    ) -> Self {
        // Minimum 1 element for valid bind groups
        let node_count = (n_nodes as u64).max(1);
        let edge_count = (n_edges as u64).max(1);

        let node_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_buf"),
            size: node_count * std::mem::size_of::<NodeGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let node_info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("node_info_buf"),
            size: node_count * std::mem::size_of::<NodeInfo>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_buf"),
            size: edge_count * std::mem::size_of::<EdgeGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Edge highlight buffer: one u32 per edge (0 = normal, 1 = highlighted)
        let edge_highlight_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("edge_highlight_buf"),
            size: edge_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sim_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim_params_buf"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &pipelines.camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let node_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("node_render_bg"),
            layout: &pipelines.storage_ro_2_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_info_buf.as_entire_binding(),
                },
            ],
        });

        let edge_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge_render_bg"),
            layout: &pipelines.edge_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_highlight_buf.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &pipelines.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sim_params_buf.as_entire_binding(),
                },
            ],
        });

        // Hit-test compute buffers
        let hit_test_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_test_params_buf"),
            size: std::mem::size_of::<HitTestParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_test_result_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_test_result_buf"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_test_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hit_test_staging_buf"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hit_test_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hit_test_bg"),
            layout: &pipelines.hit_test_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hit_test_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hit_test_result_buf.as_entire_binding(),
                },
            ],
        });

        // Tracked node position readback staging buffer (16 bytes = vec4<f32>)
        let tracked_node_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tracked_node_staging_buf"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        GraphBuffers {
            node_buf,
            node_info_buf,
            edge_buf,
            edge_highlight_buf,
            camera_buf,
            sim_params_buf,
            camera_bind_group,
            node_render_bind_group,
            edge_render_bind_group,
            compute_bind_group,
            hit_test_params_buf,
            hit_test_result_buf,
            hit_test_staging_buf,
            hit_test_bind_group,
            tracked_node_staging_buf,
            n_nodes,
            n_edges,
        }
    }
}

/// Paint callback that runs the force simulation + renders nodes and edges.
pub struct GraphPaintCallback {
    pub pipelines: Arc<GraphPipelines>,
    pub buffers: Arc<GraphBuffers>,
    pub camera: CameraUniform,
    pub run_compute: bool,
    pub iterations_per_frame: u32,
    /// If set, dispatch the hit-test compute shader this frame.
    pub hit_test_params: Option<HitTestParams>,
    /// If set, copy this node's position to the tracked node staging buffer.
    pub tracked_node_idx: Option<u32>,
}

impl CallbackTrait for GraphPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Upload camera uniform
        queue.write_buffer(
            &self.buffers.camera_buf,
            0,
            bytemuck::bytes_of(&self.camera),
        );

        // Run force compute (sim params already uploaded at load time)
        if self.run_compute && self.buffers.n_nodes > 0 {
            let workgroups = (self.buffers.n_nodes + 63) / 64;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("force_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.force_pipeline);
            pass.set_bind_group(0, &self.buffers.compute_bind_group, &[]);
            for _ in 0..self.iterations_per_frame {
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        // Hit-test compute: find nearest node to cursor on GPU
        if let Some(params) = &self.hit_test_params {
            queue.write_buffer(
                &self.buffers.hit_test_params_buf,
                0,
                bytemuck::bytes_of(params),
            );
            // Reset result to 0xFFFFFFFF (no-hit sentinel)
            queue.write_buffer(
                &self.buffers.hit_test_result_buf,
                0,
                &0xFFFF_FFFFu32.to_ne_bytes(),
            );

            let workgroups = (params.n_nodes + 63) / 64;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hit_test_compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.hit_test_pipeline);
            pass.set_bind_group(0, &self.buffers.hit_test_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            drop(pass);

            // Copy result to staging buffer for CPU readback
            encoder.copy_buffer_to_buffer(
                &self.buffers.hit_test_result_buf,
                0,
                &self.buffers.hit_test_staging_buf,
                0,
                4,
            );
        }

        // Copy tracked node position to staging buffer for CPU readback
        if let Some(idx) = self.tracked_node_idx {
            let offset = idx as u64 * std::mem::size_of::<NodeGpu>() as u64;
            encoder.copy_buffer_to_buffer(
                &self.buffers.node_buf,
                offset,
                &self.buffers.tracked_node_staging_buf,
                0,
                16, // copy first 16 bytes (pos vec4)
            );
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if self.buffers.n_nodes == 0 {
            return;
        }

        // Draw edges first (behind nodes via depth test)
        if self.buffers.n_edges > 0 {
            render_pass.set_pipeline(&self.pipelines.edge_pipeline);
            render_pass.set_bind_group(0, &self.buffers.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.buffers.edge_render_bind_group, &[]);
            render_pass.draw(0..6, 0..self.buffers.n_edges);
        }

        // Draw nodes on top
        render_pass.set_pipeline(&self.pipelines.node_pipeline);
        render_pass.set_bind_group(0, &self.buffers.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.buffers.node_render_bind_group, &[]);
        render_pass.draw(0..6, 0..self.buffers.n_nodes);
    }
}
