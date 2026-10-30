// Where the mod keeps a pack's resources: beside its `PackPost`, found by the pack's stand-in
// texture (the renderer's structs have no field for them). One pack runs at a time.

/// A running pack's resources and compute programs, and what each pass binds of them.
pub(super) struct IrisState {
    pub iris: Iris,
    /// What each pass binds of ours, by its uniform buffer: one per built pass (its clones,
    /// the weather's two, share it). Not by bind group layout: wgpu gives passes with the same
    /// entries the same layout.
    pub passes: HashMap<wgpu::Buffer, PassIris>,
    /// `gbuffers_particles_translucent`'s pass, and whether it draws after deferred.
    pub particles_translucent: Option<Pass>,
    pub particles_translucent_late: bool,
    /// Floodfill diagnostics (`probe_floodfill`): frames since the pack started, and the
    /// read-backs in flight.
    pub frames: u64,
    pub probes: Vec<Probe>,
}

pub(super) type Shared = std::rc::Rc<RefCell<IrisState>>;

thread_local! {
    static STATE: RefCell<Option<(wgpu::TextureView, Shared)>> = const { RefCell::new(None) };
    // What the passes built so far bind of ours, until their pack is registered.
    static BUILT: RefCell<HashMap<wgpu::Buffer, PassIris>> = RefCell::new(HashMap::new());
}

/// The resources of `pp`, when the mod built it.
pub(super) fn state(pp: &PackPost) -> Option<Shared> {
    STATE.with(|s| {
        s.borrow()
            .as_ref()
            .filter(|(key, _)| *key == pp.dummy)
            .map(|(_, shared)| shared.clone())
    })
}

/// `pp` runs with `iris` and the passes built since the last registration.
pub(super) fn register(pp: &PackPost, iris: Iris) {
    let passes = BUILT.with(|b| std::mem::take(&mut *b.borrow_mut()));
    let shared = std::rc::Rc::new(RefCell::new(IrisState {
        iris,
        passes,
        particles_translucent: None,
        particles_translucent_late: false,
        frames: 0,
        probes: Vec::new(),
    }));
    STATE.with(|s| *s.borrow_mut() = Some((pp.dummy.clone(), shared)));
}

/// `pp` draws the particles on the translucent layer with `pass`, after deferred when `late`.
pub(super) fn set_particles_translucent(pp: &PackPost, pass: Option<Pass>, late: bool) {
    if let Some(s) = state(pp) {
        let mut s = s.borrow_mut();
        s.particles_translucent = pass;
        s.particles_translucent_late = late;
    }
}

/// Drops the resources of the pack that ran (the renderer dropped it).
pub(crate) fn forget() {
    STATE.with(|s| *s.borrow_mut() = None);
    BUILT.with(|b| b.borrow_mut().clear());
}

/// A pass being built binds `iris` of ours.
pub(super) fn record_pass(ubo: &wgpu::Buffer, iris: PassIris) {
    BUILT.with(|b| b.borrow_mut().insert(ubo.clone(), iris));
}

/// What a pass built since the last registration binds of ours.
pub(super) fn built_pass(ubo: &wgpu::Buffer) -> PassIris {
    BUILT.with(|b| b.borrow().get(ubo).cloned().unwrap_or_default())
}

/// An image a program loads or stores, as its modules declare it.
pub(super) struct ImageUse {
    pub name: String,
    pub binding: u32,
    pub access: wgpu::StorageTextureAccess,
}

/// The images (set 0, from binding 2000) and storage buffers (from 1000: their pack index) the
/// modules declare, before `read_only_where_unwritten`: each image with the access its stages
/// declare (read and write where they differ).
pub(super) fn storage_uses(modules: &[(wgpu::ShaderStages, &naga::Module)]) -> (Vec<ImageUse>, Vec<u32>) {
    let mut images: Vec<ImageUse> = Vec::new();
    let mut buffers: Vec<u32> = Vec::new();
    for (_, m) in modules {
        for (_, g) in m.global_variables.iter() {
            let Some(b) = &g.binding else { continue };
            if b.group != 0 {
                continue;
            }
            match (&g.space, &m.types[g.ty].inner) {
                (naga::AddressSpace::Storage { .. }, _)
                    if (BUFFER_BINDING..BUFFER_BINDING + 1000).contains(&b.binding) =>
                {
                    let index = b.binding - BUFFER_BINDING;
                    if !buffers.contains(&index) {
                        buffers.push(index);
                    }
                }
                (
                    _,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { access, .. },
                        ..
                    },
                ) => {
                    let load = access.contains(naga::StorageAccess::LOAD);
                    let store = access.contains(naga::StorageAccess::STORE);
                    let access = match (load, store) {
                        (true, false) => wgpu::StorageTextureAccess::ReadOnly,
                        (false, true) => wgpu::StorageTextureAccess::WriteOnly,
                        _ => wgpu::StorageTextureAccess::ReadWrite,
                    };
                    match images.iter_mut().find(|i| i.binding == b.binding) {
                        Some(i) if i.access != access => i.access = wgpu::StorageTextureAccess::ReadWrite,
                        Some(_) => {}
                        None => images.push(ImageUse {
                            name: g.name.clone().unwrap_or_default(),
                            binding: b.binding,
                            access,
                        }),
                    }
                }
                _ => {}
            }
        }
    }
    images.sort_by_key(|i| i.binding);
    buffers.sort_unstable();
    (images, buffers)
}

/// The bind group entries of our resources a pass uses (images, buffers).
pub(super) fn iris_entries<'a>(
    iris: Option<&'a Iris>,
    pass: &PassIris,
    entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
) {
    let Some(iris) = iris else { return };
    for &(binding, k) in &pass.images {
        if let Some(view) = iris.image_view(k) {
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
    }
    for &index in &pass.buffers {
        if let Some(buffer) = iris.buffer(index) {
            entries.push(wgpu::BindGroupEntry {
                binding: BUFFER_BINDING + index,
                resource: buffer.as_entire_binding(),
            });
        }
    }
}

impl PackPost {
    /// Runs the compute programs of ranks below `rank`, and of `rank` with a number up to
    /// `number`, that have not run this frame, in order.
    pub(super) fn run_computes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        rank: u8,
        number: u32,
    ) {
        let Some(st) = state(self) else { return };
        let mut computes = {
            let mut s = st.borrow_mut();
            let n = s.iris.computes.len();
            s.iris.ran.resize(n, false);
            std::mem::take(&mut s.iris.computes)
        };
        for (k, c) in computes.iter_mut().enumerate() {
            if st.borrow().iris.ran[k] || c.setup {
                continue;
            }
            if c.rank > rank || (c.rank == rank && c.number > number) {
                continue;
            }
            self.dispatch(device, queue, encoder, c, size);
            st.borrow_mut().iris.ran[k] = true;
        }
        st.borrow_mut().iris.computes = computes;
    }

    /// The `setup` programs, once, before the first frame.
    pub(super) fn run_setup(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
    ) {
        let Some(st) = state(self) else { return };
        let mut computes = {
            let mut s = st.borrow_mut();
            if s.iris.setup_done {
                return;
            }
            s.iris.setup_done = true;
            std::mem::take(&mut s.iris.computes)
        };
        for c in computes.iter_mut().filter(|c| c.setup) {
            self.dispatch(device, queue, encoder, c, size);
        }
        st.borrow_mut().iris.computes = computes;
    }

    /// The start of a frame: images cleared, nothing run yet.
    pub(super) fn iris_begin_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(st) = state(self) {
            let mut s = st.borrow_mut();
            s.iris.clear_images(encoder);
            s.iris.ran.iter_mut().for_each(|r| *r = false);
        }
    }

    /// Screen-relative images and buffers follow the render size; bind groups are made again
    /// when they changed.
    pub(super) fn iris_resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let Some(st) = state(self) else { return };
        let changed = st.borrow_mut().iris.resize(device, size, &mut self.notes);
        if changed {
            self.clear_all_groups();
        }
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        c: &mut ComputeRun,
        size: [u32; 2],
    ) {
        let group = self.bind_compute(device, queue, c, size);
        let groups = match c.work {
            WorkGroups::Fixed(g) => g,
            WorkGroups::Render([x, y]) => [
                ((size[0] as f32 * x).ceil() as u32).div_ceil(c.local[0].max(1)),
                ((size[1] as f32 * y).ceil() as u32).div_ceil(c.local[1].max(1)),
                1,
            ],
        };
        if groups.contains(&0) {
            return;
        }
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&c.name),
            timestamp_writes: None,
        });
        cp.set_pipeline(&c.pipeline);
        cp.set_bind_group(0, &group, &[]);
        cp.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }

    /// A compute program's uniforms written and its bind group for the buffers' mapping now.
    fn bind_compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        c: &mut ComputeRun,
        out: [u32; 2],
    ) -> wgpu::BindGroup {
        let fog = self.uniforms.fog();
        let ours = [
            [
                out[0] as f32,
                out[1] as f32,
                1.0 / out[0].max(1) as f32,
                1.0 / out[1].max(1) as f32,
            ],
            fog[0].to_array(),
            fog[1].to_array(),
        ];
        let head: &[u8] = bytemuck::cast_slice(&ours);
        let n = head.len().min(c.block.len());
        c.block[..n].copy_from_slice(&head[..n]);
        for (index, ty, offset) in &c.uniforms {
            if let Some(v) = index.and_then(|i| self.uniforms.values.get(i)) {
                v.write(ty, &mut c.block, *offset as usize);
            }
        }
        if !c.block.is_empty() {
            queue.write_buffer(&c.ubo, 0, &c.block);
        }
        let shadow_main = self.shadow_targets.as_ref().map_or(0, |s| s.main);
        let key = u64::from(self.main & c.reads) | u64::from(shadow_main & c.shadow_reads) << 32;
        let st = state(self);
        let guard = st.as_ref().map(|s| s.borrow());
        let iris = guard.as_deref().map(|s| &s.iris);
        let group = c.groups.entry(key).or_insert_with(|| {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: c.ubo.as_entire_binding(),
            }];
            self.sampler_entries(iris, Group::Composite, &c.samplers, &c.iris.textures, key, 0, None, &mut entries);
            iris_entries(iris, &c.iris, &mut entries);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&c.name),
                layout: &c.layout,
                entries: &entries,
            })
        });
        group.clone()
    }

    /// The texture and sampler entries of `samplers` for a pass of `group`, with the buffers'
    /// sides in `key` (`bind`'s, shared with the compute programs).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sampler_entries<'a>(
        &'a self,
        iris: Option<&'a Iris>,
        group: Group,
        samplers: &[Binding],
        extras: &[(u32, Extra)],
        key: u64,
        current: u16,
        pass_atlas: Option<&'a wgpu::TextureView>,
        entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) {
        let targets = self.targets.as_ref().expect("resized");
        let shadow = self.shadow_targets.as_ref();
        let shadow_space = group.shadow();
        for s in samplers {
            let shadow_color =
                |i: usize, side: usize| shadow.and_then(|t| t.color[i].as_ref()).map(|c| &c[side]);
            let ours = iris.and_then(|iris| {
                let extra = extras.iter().find(|e| e.0 == s.binding)?.1;
                iris.extra_view(extra)
            });
            let texture = match ours {
                Some(view) => view,
                None => match s.source {
                    Source::Color(i) if shadow_space => {
                        shadow_color(i, usize::from(key & 1 << i != 0)).unwrap_or(&self.dummy)
                    }
                    Source::ShadowColor(i) => {
                        shadow_color(i, usize::from(key & 1 << (32 + i) != 0)).unwrap_or(&self.dummy)
                    }
                    Source::ShadowDepth(i, _) => match shadow {
                        Some(t) => match (i, &t.depth1) {
                            (1, Some(d)) => &d.1,
                            _ => &t.depth.1,
                        },
                        None => &self.shadow.0,
                    },
                    Source::Color(i) => {
                        let side = usize::from(key & 1 << i != 0);
                        match (&targets.mips[i], &targets.color[i]) {
                            (Some(m), _) if current & 1 << i != 0 => &m[side].all,
                            (_, Some(t)) => &t[side],
                            (_, None) => &self.dummy,
                        }
                    }
                    Source::Depth => &targets.depth,
                    Source::Depth1 => match &targets.depth1 {
                        Some(d)
                            if matches!(group, Group::Composite | Group::Final | Group::Translucent) =>
                        {
                            &d.1
                        }
                        _ => &targets.depth,
                    },
                    Source::Noise => &self.noise,
                    Source::Dummy => &self.dummy,
                    Source::Shadow => &self.shadow.0,
                    Source::Atlas => pass_atlas.or(self.atlas.as_ref()).unwrap_or(&self.dummy),
                    Source::Lightmap => &self.lightmap,
                },
            };
            let sampler = match SamplerKind::of(s.source, s.filtering) {
                SamplerKind::Linear => &self.linear,
                SamplerKind::Nearest => &self.nearest,
                SamplerKind::Repeat => &self.repeat,
                SamplerKind::NearestRepeat => &self.nearest_repeat,
                SamplerKind::Comparison => shadow.map_or(&self.shadow.1, |t| &t.comparison),
            };
            entries.push(wgpu::BindGroupEntry {
                binding: s.binding,
                resource: wgpu::BindingResource::TextureView(texture),
            });
            if s.sampler == s.binding + 1 {
                entries.push(wgpu::BindGroupEntry {
                    binding: s.sampler,
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            }
        }
    }

    /// Every pass's bind groups dropped (an image or buffer they bind was made again).
    pub(super) fn clear_all_groups(&mut self) {
        let clear = |p: &mut Pass| p.groups.clear();
        for p in self.passes.iter_mut() {
            clear(p);
        }
        for p in self.shadow_comp.iter_mut() {
            clear(p);
        }
        for p in [&mut self.terrain, &mut self.water, &mut self.clouds, &mut self.shadow_terrain, &mut self.shadow_water]
            .into_iter()
            .flatten()
        {
            clear(p);
        }
        if let Some(sky) = &mut self.sky {
            for p in [&mut sky.basic, &mut sky.textured].into_iter().flatten() {
                clear(p);
            }
        }
        if let Some(e) = &mut self.entities {
            for p in e.passes.iter_mut().chain(e.shadow.iter_mut()).flatten() {
                clear(p);
            }
        }
        if let Some(w) = &mut self.weather {
            for p in w.iter_mut() {
                clear(p);
            }
        }
        if let Some((h, _)) = &mut self.hand {
            clear(h);
        }
        if let Some(st) = state(self) {
            let mut s = st.borrow_mut();
            for c in s.iris.computes.iter_mut() {
                c.groups.clear();
            }
            if let Some(p) = &mut s.particles_translucent {
                clear(p);
            }
        }
    }
}

/// Where a program runs within its stage: its number, and `final` after every composite.
pub(super) fn program_order(group: Group, name: &str) -> u32 {
    if group == Group::Final {
        u32::MAX
    } else {
        number_of(name)
    }
}

/// Records the buffers each post program writes (flips), for Iris's precedence of a stage's
/// custom textures over its buffers.
pub(super) fn record_writes(programs: &[horizon_shaders::PostProgram]) {
    let writes = programs
        .iter()
        .map(|p| {
            let mut mask = 0u16;
            for &b in &p.buffers {
                if b < 16 && !p.flips.iter().any(|f| f.0 == b && !f.1) {
                    mask |= 1 << b;
                }
            }
            for f in &p.flips {
                if f.1 && f.0 < 16 {
                    mask |= 1 << f.0;
                }
            }
            (stage_of(p.group), program_order(p.group, &p.name), mask)
        })
        .collect();
    BUILD.with(|b| {
        if let Some(c) = b.borrow_mut().as_mut() {
            c.writes = writes;
        }
    });
}

/// The buffers programs of `stage` before `order` have written: a custom texture no longer
/// replaces them.
fn flipped_before(stage: &str, order: u32) -> u16 {
    with_build(|b| {
        b.writes
            .iter()
            .filter(|w| w.0 == stage && w.1 < order)
            .fold(0u16, |m, w| m | w.2)
    })
    .unwrap_or(0)
}

use super::super::pack_entities::iris_entities::PARTICLE_TRANSLUCENT;

impl PackPost {
    /// `prepare_entities`, keeping the translucent particles apart when the pack has their
    /// program.
    pub(super) fn iris_prepare_entities(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entities: &[super::super::Vertex],
        spans: &[super::super::pack_draw::EntitySpan],
        rest: &mut Vec<super::super::Vertex>,
    ) {
        let split = state(self).is_some_and(|s| s.borrow().particles_translucent.is_some());
        let shadow = self.shadow_casters();
        let Some(e) = &mut self.entities else {
            rest.clear();
            rest.extend_from_slice(entities);
            return;
        };
        let drawn = e.passes.each_ref().map(Option::is_some);
        e.geometry.iris_prepare(
            device,
            queue,
            &self.draws.layout,
            entities,
            spans,
            drawn,
            shadow,
            rest,
            split,
        );
    }

    /// `draw_entities`, then the translucent particles with their own program.
    pub(super) fn iris_draw_entities(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        late: bool,
    ) {
        let Some(mut entities) = self.entities.take() else {
            return;
        };
        let shared = state(self);
        let mut translucent = shared.as_ref().and_then(|s| {
            let mut s = s.borrow_mut();
            (s.particles_translucent_late == late)
                .then(|| s.particles_translucent.take())
                .flatten()
        });
        let geometry = &entities.geometry;
        let passes = (entities.passes.iter_mut().enumerate())
            .filter(|(program, _)| entities.late[*program] == late)
            .filter_map(|(program, pass)| pass.as_mut().map(|p| (program, p)))
            .chain(translucent.as_mut().map(|p| (PARTICLE_TRANSLUCENT, p)));
        for (program, pass) in passes {
            let mut draws = geometry
                .draws
                .iter()
                .filter(|d| d.pack && d.program == program)
                .peekable();
            if draws.peek().is_none() {
                continue;
            }
            let group = self.bind(device, queue, pass, size);
            {
                let mut rp = self.begin(encoder, pass);
                rp.set_pipeline(&pass.pipeline);
                rp.set_bind_group(0, &group, &[]);
                rp.set_bind_group(1, &geometry.table_group, &[]);
                rp.set_vertex_buffer(0, geometry.vertices());
                rp.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
                for d in draws {
                    if pass.draw {
                        rp.set_bind_group(2, &geometry.group, &[geometry.offset(d.slot)]);
                    }
                    rp.draw_indexed(0..d.quads * 6, d.base as i32, 0..1);
                }
            }
            self.mips_valid &= !pass.written;
        }
        if let (Some(s), Some(pass)) = (shared, translucent) {
            s.borrow_mut().particles_translucent = Some(pass);
        }
        self.entities = Some(entities);
    }
}

/// A frame's floodfill images copied for reading back: each image's name, buffer, bytes per
/// texel and size, and how many buffers are mapped.
pub(super) struct Probe {
    frame: u64,
    images: Vec<(String, wgpu::Buffer, u32, [u32; 3])>,
    mapped: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    requested: bool,
}

/// Floodfill diagnostics, from Iris Features' page: every 4 seconds or so, two frames in a row
/// of the voxels and light volumes go to the log. A light that flickers shows as numbers that
/// change between the two.
pub static PROBE_FLOODFILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl PackPost {
    /// Reads back what the probes copied, and copies this frame's when it is time.
    pub(super) fn probe_floodfill(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(st) = state(self) else { return };
        let mut s = st.borrow_mut();
        s.frames += 1;
        let on = PROBE_FLOODFILL.load(Relaxed);
        if !on && s.probes.is_empty() {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let mut done = Vec::new();
        for (k, p) in s.probes.iter_mut().enumerate() {
            if !p.requested {
                // Copied by an earlier frame, which has been submitted since.
                for (_, buffer, _, _) in &p.images {
                    let mapped = p.mapped.clone();
                    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        if r.is_ok() {
                            mapped.fetch_add(1, Relaxed);
                        }
                    });
                }
                p.requested = true;
            } else if p.mapped.load(Relaxed) == p.images.len() {
                done.push(k);
            }
        }
        for k in done.into_iter().rev() {
            let p = s.probes.remove(k);
            let mut line = format!("floodfill probe, frame {}:", p.frame);
            for (name, buffer, bpt, size) in &p.images {
                {
                    let Ok(data) = buffer.slice(..).get_mapped_range() else { continue };
                    let row = (size[0] * bpt).next_multiple_of(256) as usize;
                    let rows = (size[1] * size[2]) as usize;
                    let texels = (0..rows).flat_map(|y| {
                        data[y * row..y * row + (size[0] * bpt) as usize].chunks_exact(*bpt as usize)
                    });
                    if name == "imgVoxelMask" {
                        let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
                        for t in texels {
                            let id = u32::from(t[0]) | u32::from(*t.get(1).unwrap_or(&0)) << 8;
                            *counts.entry(id).or_default() += 1;
                        }
                        let solid = counts.get(&1).copied().unwrap_or(0);
                        let mut others: Vec<(u32, usize)> =
                            counts.into_iter().filter(|(id, _)| *id > 1).collect();
                        others.sort_by(|a, b| b.1.cmp(&a.1));
                        others.truncate(8);
                        line += &format!(" voxels solid {solid}, others (id, count) {others:?};");
                    } else {
                        let (mut sum, mut lit) = (0u64, 0usize);
                        for t in texels {
                            let v = u64::from(t[0]) + u64::from(t[1]) + u64::from(t[2]);
                            sum += v;
                            lit += usize::from(v > 0);
                        }
                        line += &format!(" {name} light {sum} in {lit} voxels;");
                    }
                }
                buffer.unmap();
            }
            log::info!("{line} images cleared each frame: {}", s.iris.can_clear);
        }
        // Two frames in a row, every 256 frames.
        if !on || s.frames % 256 > 1 || s.probes.len() > 4 {
            return;
        }
        let mut images = Vec::new();
        for image in s.iris.images.iter().filter(|i| i.decl.name.starts_with("imgLpv") || i.decl.name == "imgVoxelMask") {
            let Some((texture, _)) = &image.texture else { continue };
            let bpt = texture.format().block_copy_size(None).unwrap_or(4);
            let size = texture.size();
            let row = (size.width * bpt).next_multiple_of(256);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Floodfill probe"),
                size: u64::from(row) * u64::from(size.height * size.depth_or_array_layers),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(size.height),
                    },
                },
                size,
            );
            images.push((
                image.decl.name.clone(),
                buffer,
                bpt,
                [size.width, size.height, size.depth_or_array_layers],
            ));
        }
        if !images.is_empty() {
            let frame = s.frames;
            s.probes.push(Probe {
                frame,
                images,
                mapped: Default::default(),
                requested: false,
            });
        }
    }
}
