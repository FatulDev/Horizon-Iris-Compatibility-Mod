// Iris Features: the block outline while a pack runs. Iris draws it after the pack's deferred
// programs, into what translucent terrain writes, so the pack neither lights nor reflects it and
// it follows the pack's own camera; Horizon's own outline goes into its world pass instead.
use super::*;

/// This frame's outline, camera-relative line segments (`Frame::outline`).
static OUTLINE: std::sync::Mutex<Vec<[f32; 6]>> = std::sync::Mutex::new(Vec::new());

/// Notes the frame's outline (before the pack runs).
pub fn set_outline(frame: &crate::gfx::Frame) {
    crate::gfx::iris::set_crumbling(frame.crumbling_sprite);
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
static OUTLINE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const SHADER: &str = "
struct U { m: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@vertex fn vs(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
    // Pulled toward the camera as vanilla's lines are (`rendertype_lines`' view shrink and
    // perspective layering): drawn on the block's own faces, they lost the depth test.
    return u.m * vec4<f32>(p * (1.0 - 1.0 / 256.0) * (1.0 - 1.0 / 4096.0), 1.0);
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
    let skip = |why: &str| {
        if !OUTLINE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            log::info!("shader pack: no block outline: {why}");
        }
    };
    let (Some(water), Some(targets)) = (&pp.water, &pp.targets) else {
        return skip("the pack has no translucent terrain pass");
    };
    let Some(&(_, buffer)) = water.writes.iter().find(|w| w.0 == 0) else {
        return skip("its translucent terrain writes no colour");
    };
    let Some(sides) = &targets.color[buffer] else {
        return skip("no buffer behind its translucent colour");
    };
    let view = &sides[usize::from(pp.main & 1 << buffer != 0)];
    let format = view.texture().format();
    let depth_format = targets.depth_texture.format();
    // Once per pack, what the outline draws into, for reports (Iris Features).
    if !OUTLINE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::info!(
            "shader pack: block outline drawn after translucent terrain into colortex{buffer} \
             ({format:?}, depth {depth_format:?}), {} edges",
            edges.len()
        );
    }
    // Blending needs a blendable format; else the lines are written as they are.
    let blendable = format
        .guaranteed_format_features(device.features())
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE);
    // The pack's camera: its projection (GL depth, -1 to 1, made wgpu's 0 to 1) and view.
    let matrix = |name: &str| {
        let i = pack_uniforms::index(name)?;
        match pp.uniforms.values.get(i)? {
            pack_uniforms::Value::F(f) => Some(glam::Mat4::from_cols_array(f)),
            _ => None,
        }
    };
    let (Some(projection), Some(model_view)) = (matrix("gbufferProjection"), matrix("gbufferModelView")) else {
        return skip("no camera matrices");
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
                    blend: blendable.then_some(wgpu::BlendState::ALPHA_BLENDING),
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

// --- Passes Horizon has no place for ---------------------------------------------------------
// Built from the programs the pack's `load_post` put aside (`horizon_shaders::iris::take_extra`),
// with the pack's own resources, the first time a frame needs them after the pack loaded.

static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A pack was built: its extra passes are built again.
pub fn new_pack() {
    OUTLINE_LOGGED.store(false, std::sync::atomic::Ordering::Relaxed);
    // A new pack may come with a new device: the outline's pipeline is made again.
    *LINES.lock().unwrap_or_else(|e| e.into_inner()) = None;
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

struct Extra {
    generation: u64,
    /// `gbuffers_hand_water`, and the held translucent items it draws this frame.
    hand_water: Option<(Pass, super::super::pack_hand::PackHand)>,
    translucent_hand: Vec<crate::mesher::Vertex>,
    hand_projection: glam::Mat4,
    /// Rain and snow without depth (`rain.depth` unset).
    weather: Option<[Pass; 2]>,
    /// Block-breaking cracks and translucent particles (`pack_entities::DAMAGED`...).
    damaged: Option<Pass>,
    particles_translucent: Option<Pass>,
    /// The weather sprites' generation the depth-less passes were bound for.
    weather_generation: Option<u64>,
}

thread_local! {
    static EXTRA: std::cell::RefCell<Option<Extra>> = const { std::cell::RefCell::new(None) };
}

/// A pass of program `p` drawn over `geometry`, with the pack's resources.
fn build_extra(
    pp: &PackPost,
    device: &wgpu::Device,
    p: horizon_shaders::PostProgram,
    geometry: &Geometry<'_>,
) -> Option<Pass> {
    let shadow: Option<[wgpu::TextureFormat; 2]> = pp.shadow_targets.as_ref().map(|s| {
        std::array::from_fn(|n| {
            s.color[n]
                .as_ref()
                .map_or(pp.formats[n], |c| c[0].texture().format())
        })
    });
    let resources = Resources {
        textures: &pp.textures,
        images: &pp.images,
        left_out: &[],
        buffers: &pp.buffers,
        indices: pp.indices.as_ref(),
        gpu: pp.gpu,
        storable: pp.storage,
    };
    let custom: Vec<String> = pp.custom.uniforms().map(|u| u.0.to_owned()).collect();
    let mut notes = Vec::new();
    let name = p.name.clone();
    let pass = build_pass(
        device,
        p,
        &pp.formats,
        shadow.as_ref(),
        pp.formats[0],
        &mut notes,
        Some(geometry),
        &custom,
        &pp.draws.layout,
        &resources,
    );
    for n in notes {
        if cfg!(test) {
            println!("  note: {n}");
        }
        log::info!("shader pack: {n}");
    }
    if cfg!(test) {
        println!("  note: {name} built: {}", pass.is_some());
    }
    if pass.is_none() {
        log::warn!("shader pack: {name} could not be built (Iris Features)");
    }
    pass
}

fn with_extra<R>(pp: &PackPost, device: &wgpu::Device, f: impl FnOnce(&mut Extra) -> R) -> R {
    let generation = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
    EXTRA.with(|e| {
        let mut e = e.borrow_mut();
        if e.as_ref().is_none_or(|x| x.generation != generation) {
            let entity_buffers = super::super::pack_entities::buffers();
            let hand_water = horizon_shaders::iris::take_extra("hand_water").and_then(|p| {
                let geometry = super::super::pack_hand::PackHand::new(device);
                let g = Geometry {
                    buffers: &entity_buffers,
                    table: Some(&geometry.layout),
                    depth_write: true,
                    atlas: pp.hand.as_ref().and_then(|h| h.0.atlas.clone()),
                    ids: None,
                    // Back faces culled, as vanilla draws items: drawn both ways, held glass
                    // stacked its far faces under its near ones and turned near opaque.
                    cull: true,
                };
                build_extra(pp, device, p, &g).map(|pass| (pass, geometry))
            });
            let weather = horizon_shaders::iris::take_extra("weather").and_then(|p| {
                let g = Geometry {
                    buffers: &entity_buffers,
                    table: None,
                    depth_write: false,
                    atlas: None,
                    ids: None,
                    cull: false,
                };
                build_extra(pp, device, p, &g).map(|rain| {
                    let mut snow = rain.clone();
                    snow.name = format!("{} (snow)", rain.name);
                    [rain, snow]
                })
            });
            let entity = |name: &str, depth_write: bool| {
                let e = pp.entities.as_ref()?;
                let p = horizon_shaders::iris::take_extra(name)?;
                let atlas = e.passes.iter().flatten().find_map(|p| p.atlas.clone());
                let g = Geometry {
                    buffers: &entity_buffers,
                    table: Some(&e.geometry.table),
                    depth_write,
                    atlas,
                    ids: None,
                    cull: false,
                };
                build_extra(pp, device, p, &g)
            };
            let damaged = entity("damaged", false);
            let particles_translucent = entity("particles_translucent", true);
            *e = Some(Extra {
                generation,
                damaged,
                particles_translucent,
                hand_water,
                translucent_hand: Vec::new(),
                hand_projection: glam::Mat4::IDENTITY,
                weather,
                weather_generation: None,
            });
        }
        f(e.as_mut().expect("built"))
    })
}

/// Whether held quad `q` is a translucent item's (stained glass, ice, slime: the item renderer's
/// translucent layers).
fn translucent_quad(q: &[crate::mesher::Vertex]) -> bool {
    matches!(
        q[0].layer,
        super::super::entity_batch::TRANSLUCENT_ITEM_OVERLAY
            | super::super::entity_batch::TRANSLUCENT_GLINT_ITEM_OVERLAY
    )
}

/// `PackPost::draw_hand`, with held translucent items kept for `gbuffers_hand_water` when the
/// pack has it (Iris draws them after translucent terrain with that program; with
/// `gbuffers_hand`, Eclipse dithered stained glass into grain).
pub fn draw_hand(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    size: [u32; 2],
    frame: &crate::gfx::Frame,
) {
    // Built while the hand's pass is in place: `gbuffers_hand_water` takes its atlas (built
    // without it, held glass sampled no texture).
    let split = with_extra(pp, device, |e| e.hand_water.is_some());
    let Some((mut pass, mut geometry)) = pp.hand.take() else {
        return;
    };
    let unspin = frame.bob * frame.spin.inverse() * frame.bob.inverse();
    let projection = super::super::pack_hand::projection(
        frame.hand_fov_degrees,
        size[0] as f32 / size[1].max(1) as f32,
        frame.depth_far,
    ) * unspin;
    let (solid, translucent): (Vec<crate::mesher::Vertex>, Vec<crate::mesher::Vertex>) = if split {
        let mut solid = Vec::new();
        let mut translucent = Vec::new();
        for q in frame.hand.chunks_exact(4) {
            if translucent_quad(q) {
                translucent.extend_from_slice(q);
            } else {
                solid.extend_from_slice(q);
            }
        }
        (solid, translucent)
    } else {
        (frame.hand.to_vec(), Vec::new())
    };
    EXTRA.with(|e| {
        if let Some(e) = e.borrow_mut().as_mut() {
            e.translucent_hand = translucent;
            e.hand_projection = projection;
        }
    });
    geometry.prepare(device, queue, &solid, projection);
    if geometry.quads > 0 {
        let group = pp.bind(device, queue, encoder, &mut pass, size);
        {
            let mut rp = pp.begin(encoder, &pass);
            rp.set_pipeline(pass.render());
            rp.set_bind_group(0, &group, &[]);
            if pass.buffers
                && let Some(g) = &pp.buffers.group
            {
                rp.set_bind_group(4, g, &[]);
            }
            rp.set_bind_group(1, &geometry.group, &[]);
            if pass.draw {
                pp.draws.bind(&mut rp, "HAND_SOLID");
            }
            rp.set_vertex_buffer(0, geometry.vertices());
            rp.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..geometry.quads * 6, 0, 0..1);
        }
        pp.mips_valid &= !pass.written;
    }
    pp.hand = Some((pass, geometry));
}

/// The held translucent items `draw_hand` kept, with `gbuffers_hand_water`, after translucent
/// terrain.
pub fn draw_hand_water(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    size: [u32; 2],
) {
    let taken = EXTRA.with(|e| {
        let mut e = e.borrow_mut();
        let e = e.as_mut()?;
        if e.translucent_hand.is_empty() {
            return None;
        }
        let (pass, geometry) = e.hand_water.take()?;
        Some((pass, geometry, std::mem::take(&mut e.translucent_hand), e.hand_projection))
    });
    let Some((mut pass, mut geometry, quads, projection)) = taken else {
        return;
    };
    geometry.prepare(device, queue, &quads, projection);
    if geometry.quads > 0 {
        let group = pp.bind(device, queue, encoder, &mut pass, size);
        {
            let mut rp = pp.begin(encoder, &pass);
            rp.set_pipeline(pass.render());
            rp.set_bind_group(0, &group, &[]);
            if pass.buffers
                && let Some(g) = &pp.buffers.group
            {
                rp.set_bind_group(4, g, &[]);
            }
            rp.set_bind_group(1, &geometry.group, &[]);
            if pass.draw {
                pp.draws.bind(&mut rp, "HAND_TRANSLUCENT");
            }
            rp.set_vertex_buffer(0, geometry.vertices());
            rp.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..geometry.quads * 6, 0, 0..1);
        }
        pp.mips_valid &= !pass.written;
    }
    EXTRA.with(|e| {
        if let Some(e) = e.borrow_mut().as_mut() {
            e.hand_water = Some((pass, geometry));
        }
    });
}

/// Rain and snow drawn without depth unless the pack sets `rain.depth=true` (Iris): with depth,
/// a pack's later programs took each drop for a surface (Eclipse's water and fog behind rain).
/// Swaps the depth-less passes in for `PackPost::draw_weather`, and back after.
pub fn weather_passes(pp: &mut PackPost, device: &wgpu::Device) {
    if horizon_shaders::iris::rain_depth() || pp.weather.is_none() {
        return;
    }
    let generation = pp.weather_generation;
    let ours = with_extra(pp, device, |e| {
        let stale = e.weather_generation != Some(generation);
        e.weather_generation = Some(generation);
        e.weather.take().map(|w| (w, stale))
    });
    if let Some((mut ours, stale)) = ours {
        // The sprites the game's passes were last bound with (`draw_weather` rebinds them when
        // the generation changes).
        if let Some(theirs) = &pp.weather {
            for (o, t) in ours.iter_mut().zip(theirs) {
                if stale || o.atlas.is_none() {
                    o.atlas = t.atlas.clone();
                    o.groups.clear();
                }
            }
        }
        let theirs = pp.weather.replace(ours);
        EXTRA.with(|e| {
            if let Some(e) = e.borrow_mut().as_mut() {
                e.weather = theirs;
            }
        });
    }
}

/// The entity runs of the mod's programs (`pack_entities::DAMAGED` after the opaque entities,
/// `PARTICLE_TRANSLUCENT` with the translucent ones), as `PackPost::draw_entities` draws its own.
pub fn draw_extra_entities(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    size: [u32; 2],
    late: bool,
) {
    let program = if late {
        crate::gfx::iris::PARTICLE_TRANSLUCENT
    } else {
        crate::gfx::iris::DAMAGED
    };
    if !pp.entities.as_ref().is_some_and(|e| {
        e.geometry.draws.iter().any(|d| d.pack && d.program == program)
    }) {
        return;
    }
    // Built while the entities are still in place: their passes give the atlas and table.
    let taken = with_extra(pp, device, |e| {
        if late { e.particles_translucent.take() } else { e.damaged.take() }
    });
    let Some(entities) = pp.entities.take() else {
        return;
    };
    {
        if let Some(mut pass) = taken {
            let geometry = &entities.geometry;
            let group = pp.bind(device, queue, encoder, &mut pass, size);
            {
                let mut rp = pp.begin(encoder, &pass);
                rp.set_pipeline(pass.render());
                rp.set_bind_group(0, &group, &[]);
                if pass.buffers
                    && let Some(g) = &pp.buffers.group
                {
                    rp.set_bind_group(4, g, &[]);
                }
                rp.set_bind_group(1, &geometry.table_group, &[]);
                rp.set_vertex_buffer(0, geometry.vertices());
                rp.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
                for d in geometry.draws.iter().filter(|d| d.pack && d.program == program) {
                    if pass.draw {
                        rp.set_bind_group(2, &geometry.group, &[geometry.offset(d.slot)]);
                    }
                    rp.draw_indexed(0..d.quads * 6, d.base as i32, 0..1);
                }
            }
            pp.mips_valid &= !pass.written;
            EXTRA.with(|e| {
                if let Some(e) = e.borrow_mut().as_mut() {
                    if late {
                        e.particles_translucent = Some(pass);
                    } else {
                        e.damaged = Some(pass);
                    }
                }
            });
        }
    }
    pp.entities = Some(entities);
}
