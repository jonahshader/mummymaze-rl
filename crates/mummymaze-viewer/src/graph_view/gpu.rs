//! wgpu pipeline creation, buffer management, and CallbackTrait impl.

use super::shaders;
use super::types::*;
use eframe::egui;
use egui_wgpu::{CallbackTrait, RenderState};
use eframe::wgpu;
use std::sync::Arc;

/// All GPU resources for graph rendering.
pub struct GraphGpuResources {
    // Buffers
    pub node_buf: wgpu::Buffer,
    pub node_info_buf: wgpu::Buffer,
    pub edge_buf: wgpu::Buffer,
    pub camera_buf: wgpu::Buffer,
    pub sim_params_buf: wgpu::Buffer,

    // Render pipelines
    pub node_pipeline: wgpu::RenderPipeline,
    pub edge_pipeline: wgpu::RenderPipeline,

    // Compute pipeline
    pub force_pipeline: wgpu::ComputePipeline,

    // Bind groups
    pub camera_bind_group: wgpu::BindGroup,
    pub node_render_bind_group: wgpu::BindGroup,
    pub edge_render_bind_group: wgpu::BindGroup,
    pub compute_bind_group: wgpu::BindGroup,

    // Counts
    pub n_nodes: u32,
    pub n_edges: u32,
}

impl GraphGpuResources {
    pub fn new(render_state: &RenderState, n_nodes: u32, n_edges: u32) -> Self {
        let device = &render_state.device;

        // Allocate buffers sized to actual data (minimum 1 element for valid bind groups)
        let node_count = (n_nodes as u64).max(1);
        let edge_count = (n_edges as u64).max(1);

        // --- Buffers ---
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

        let node_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("node_render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let edge_render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edge_render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        // --- Bind groups ---
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let node_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("node_render_bg"),
            layout: &node_render_bgl,
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
            layout: &edge_render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: edge_buf.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &compute_bgl,
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

        // --- Render pipelines ---
        let node_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("node_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::NODE_SHADER.into()),
        });

        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edge_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::EDGE_SHADER.into()),
        });

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
                bind_group_layouts: &[&camera_bgl, &node_render_bgl],
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
            depth_stencil: None,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Compute pipeline ---
        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("force_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FORCE_COMPUTE_SHADER.into()),
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

        GraphGpuResources {
            node_buf,
            node_info_buf,
            edge_buf,
            camera_buf,
            sim_params_buf,
            node_pipeline,
            edge_pipeline,
            force_pipeline,
            camera_bind_group,
            node_render_bind_group,
            edge_render_bind_group,
            compute_bind_group,
            n_nodes: 0,
            n_edges: 0,
        }
    }
}

/// Paint callback that runs the force simulation + renders nodes and edges.
pub struct GraphPaintCallback {
    pub gpu: Arc<GraphGpuResources>,
    pub camera: CameraUniform,
    pub sim_params: SimParams,
    pub run_compute: bool,
    pub iterations_per_frame: u32,
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
            &self.gpu.camera_buf,
            0,
            bytemuck::bytes_of(&self.camera),
        );

        // Upload sim params and run compute
        if self.run_compute && self.gpu.n_nodes > 0 {
            queue.write_buffer(
                &self.gpu.sim_params_buf,
                0,
                bytemuck::bytes_of(&self.sim_params),
            );

            let workgroups = (self.gpu.n_nodes + 63) / 64;
            for _ in 0..self.iterations_per_frame {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("force_compute"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.gpu.force_pipeline);
                pass.set_bind_group(0, &self.gpu.compute_bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if self.gpu.n_nodes == 0 {
            return;
        }

        // Draw edges first (behind nodes)
        if self.gpu.n_edges > 0 {
            render_pass.set_pipeline(&self.gpu.edge_pipeline);
            render_pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.gpu.edge_render_bind_group, &[]);
            render_pass.draw(0..6, 0..self.gpu.n_edges);
        }

        // Draw nodes on top
        render_pass.set_pipeline(&self.gpu.node_pipeline);
        render_pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &self.gpu.node_render_bind_group, &[]);
        render_pass.draw(0..6, 0..self.gpu.n_nodes);
    }
}
