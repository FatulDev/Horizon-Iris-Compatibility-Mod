// Iris Features: the block outline while a pack runs. Iris draws it after the pack's deferred
// programs, into what translucent terrain writes, so the pack neither lights nor reflects it and
// it follows the pack's own camera; Horizon's own outline goes into its world pass instead.
use super::*;

/// This frame's outline, camera-relative line segments (`Frame::outline`).
static OUTLINE: std::sync::Mutex<Vec<[f32; 6]>> = std::sync::Mutex::new(Vec::new());

/// Notes the frame's outline (before the pack runs).
pub fn set_outline(frame: &crate::gfx::Frame) {
    let edges = crate::gfx::outline_vertices(&frame.outline, frame.camera);
    *OUTLINE.lock().unwrap_or_else(|e| e.into_inner()) = edges;
}

struct Lines {
    format: wgpu::TextureFormat,
    depth: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

static LINES: std::sync::Mutex<Option<Lines>> = std::sync::Mutex::new(None);

const SHADER: &str = "
struct U { m: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.m * vec4<f32>(p, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 0.4);
}
";

/// Draws the outline into the buffer translucent terrain writes at location 0: black at 40%,
/// as vanilla's, behind what stands in front of it.
pub fn draw_outline(pp: &PackPost, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
    let edges = OUTLINE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if edges.is_empty() {
        return;
    }
    let (Some(water), Some(targets)) = (&pp.water, &pp.targets) else {
        return;
    };
    let Some(&(_, buffer)) = water.writes.iter().find(|w| w.0 == 0) else {
        return;
    };
    let Some(sides) = &targets.color[buffer] else {
        return;
    };
    let view = &sides[usize::from(pp.main & 1 << buffer != 0)];
    let format = view.texture().format();
    let depth_format = targets.depth_texture.format();
    // The pack's camera: its projection (GL depth, -1 to 1, made wgpu's 0 to 1) and view.
    let matrix = |name: &str| {
        let i = pack_uniforms::index(name)?;
        match pp.uniforms.values.get(i)? {
            pack_uniforms::Value::F(f) => Some(glam::Mat4::from_cols_array(f)),
            _ => None,
        }
    };
    let (Some(projection), Some(model_view)) = (matrix("gbufferProjection"), matrix("gbufferModelView")) else {
        return;
    };
    // As the translator ends a vertex stage drawing into buffers: y flipped (the buffers are
    // stored as GL lays textures out), depth from -1..1 to 0..1.
    let depth = glam::Mat4::from_cols_array(&[
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
    ]);
    let m = depth * projection * model_view;
    let mut lines = LINES.lock().unwrap_or_else(|e| e.into_inner());
    if lines.as_ref().is_none_or(|l| l.format != format || l.depth != depth_format) {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Iris Features outline"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Iris Features outline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        *lines = Some(Lines { format, depth: depth_format, pipeline, layout });
    }
    let Some(lines) = lines.as_ref() else { return };
    let vertices: Vec<f32> = edges.iter().flatten().copied().collect();
    let vb = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ub = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&m.to_cols_array()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &lines.layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Iris Features outline"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    pass.set_pipeline(&lines.pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.set_vertex_buffer(0, vb.slice(..));
    pass.draw(0..(edges.len() * 2) as u32, 0..1);
}
