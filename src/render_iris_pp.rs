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
    /// `gbuffers_hand_water`'s pass and geometry, when the pack has it: held items with
    /// translucent texels (stained glass, ice, slime) draw with it after translucent terrain,
    /// as in Iris, and the rest with `gbuffers_hand`. This frame's translucent quads wait here.
    pub hand_translucent: Option<(Pass, super::super::pack_hand::PackHand)>,
    pub hand_translucent_quads: Vec<super::super::Vertex>,
    /// The hit block's outline (`iris_draw_outline`), made at its first draw.
    pub outline: Option<Outline>,
    /// Floodfill diagnostics (`probe_floodfill`): frames since the pack started, and the
    /// read-backs in flight.
    pub frames: u64,
    pub probes: Vec<Probe>,
    /// The texels of the last frames read back: frame, image, texels (tight), size, bytes each.
    pub probed: Vec<(u64, String, Vec<u8>, [u32; 3], u32)>,
    /// Images a pass both samples and loads or stores: what it samples instead (`Snapshot`).
    pub snapshots: Vec<Snapshot>,
}

/// A copy of an image for a pass that samples it while it also binds it as an image (Pegasus's
/// light volume, written in one half and sampled in the other). GL lets one draw do both; wgpu
/// refuses the pass. The copy is taken at the start of each frame and before each compute
/// program that samples it: what the pack reads is the image as the frame or program began.
pub(super) struct Snapshot {
    /// The image (`Iris::images`), and the custom texture (`Iris::textures`) passes sample.
    pub image: usize,
    pub custom: usize,
    pub texture: Option<wgpu::Texture>,
    /// A colour buffer instead of an image: a compute program samples `colortexN` while it
    /// writes `colorimgN` (Photon, Noble).
    pub color: Option<usize>,
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
        hand_translucent: None,
        hand_translucent_quads: Vec::new(),
        outline: None,
        frames: 0,
        probes: Vec::new(),
        probed: Vec::new(),
        snapshots: Vec::new(),
    }));
    plan_snapshots(&mut shared.borrow_mut(), &pp.dummy);
    STATE.with(|s| *s.borrow_mut() = Some((pp.dummy.clone(), shared)));
}

/// Passes that sample an image they also bind as an image sample a copy of it instead.
fn plan_snapshots(s: &mut IrisState, dummy: &wgpu::TextureView) {
    fn fix(pi: &mut PassIris, iris: &mut Iris, snaps: &mut Vec<Snapshot>, dummy: &wgpu::TextureView) {
        for t in pi.textures.iter_mut() {
            let Extra::Image(k) = t.1 else { continue };
            if !pi.images.iter().any(|(_, i)| *i == k) {
                continue;
            }
            let custom = match snaps.iter().find(|s| s.image == k) {
                Some(s) => s.custom,
                None => {
                    let Some(image) = iris.images.get(k) else { continue };
                    let view_dimension = image.view_dimension;
                    let sampler = format!("{} (sampled copy)", image.decl.name);
                    iris.textures.push(CustomTexture {
                        decl: TextureDecl { sampler, stage: None, path: String::new(), raw: None },
                        // The copy's own view from the first frame on.
                        view: dummy.clone(),
                        view_dimension,
                    });
                    let custom = iris.textures.len() - 1;
                    snaps.push(Snapshot { image: k, custom, texture: None, color: None });
                    custom
                }
            };
            t.1 = Extra::Custom(custom);
        }
    }
    let IrisState { iris, passes, snapshots, .. } = s;
    for pi in passes.values_mut() {
        fix(pi, iris, snapshots, dummy);
    }
    let mut computes = std::mem::take(&mut iris.computes);
    for c in computes.iter_mut() {
        fix(&mut c.iris, iris, snapshots, dummy);
        let colors = color_images(&c.iris);
        for b in &c.samplers {
            let Source::Color(n) = b.source else { continue };
            if colors & 1 << n == 0 {
                continue;
            }
            let custom = match snapshots.iter().find(|s| s.color == Some(n)) {
                Some(s) => s.custom,
                None => {
                    iris.textures.push(CustomTexture {
                        decl: TextureDecl { sampler: format!("colortex{n} (sampled copy)"), stage: None, path: String::new(), raw: None },
                        view: dummy.clone(),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    });
                    let custom = iris.textures.len() - 1;
                    snapshots.push(Snapshot { image: usize::MAX, custom, texture: None, color: Some(n) });
                    custom
                }
            };
            c.iris.textures.retain(|t| t.0 != b.binding);
            c.iris.textures.push((b.binding, Extra::Custom(custom)));
        }
    }
    iris.computes = computes;
    if !snapshots.is_empty() {
        let names: Vec<&str> = snapshots.iter().map(|s| iris.textures[s.custom].decl.sampler.as_str()).collect();
        log::info!("images sampled where they are also written, read from a copy taken first: {names:?}");
    }
}

/// Brings the copies of `snapshots` up to date (made again when their image was), for the
/// passes sampling `only` (all when `None`). Whether a copy was made again (bind groups stale).
fn take_snapshots(
    s: &mut IrisState,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    only: Option<&[(u32, Extra)]>,
    color: &dyn Fn(usize) -> Option<wgpu::Texture>,
) -> bool {
    let IrisState { iris, snapshots, .. } = s;
    let mut remade = false;
    for snap in snapshots.iter_mut() {
        if let Some(only) = only
            && !only.iter().any(|(_, e)| *e == Extra::Custom(snap.custom))
        {
            continue;
        }
        let (texture, view_dimension) = match snap.color {
            Some(n) => match color(n) {
                Some(t) => (t, wgpu::TextureViewDimension::D2),
                None => continue,
            },
            None => {
                let Some(image) = iris.images.get(snap.image) else { continue };
                let Some((texture, _)) = &image.texture else { continue };
                (texture.clone(), image.view_dimension)
            }
        };
        let texture = &texture;
        let fits = snap.texture.as_ref().is_some_and(|t| t.size() == texture.size() && t.format() == texture.format());
        if !fits {
            let copy = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&iris.textures[snap.custom].decl.sampler),
                size: texture.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: texture.dimension(),
                format: texture.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            iris.textures[snap.custom].view = copy.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(view_dimension),
                ..Default::default()
            });
            snap.texture = Some(copy);
            remade = true;
        }
        if let Some(copy) = &snap.texture {
            encoder.copy_texture_to_texture(texture.as_image_copy(), copy.as_image_copy(), texture.size());
        }
    }
    remade
}

/// `pp` draws held items with translucent texels with `pass`.
pub(super) fn set_hand_translucent(pp: &PackPost, pass: Option<(Pass, super::super::pack_hand::PackHand)>) {
    if let Some(s) = state(pp) {
        s.borrow_mut().hand_translucent = pass;
    }
}

/// Whether a hand quad of 4 vertices is one of a translucent item's (`WorldSink`'s overlay).
fn translucent_hand_quad(q: &[super::super::Vertex]) -> bool {
    matches!(q[0].layer, super::super::entity_batch::TRANSLUCENT_ITEM_OVERLAY | super::super::entity_batch::TRANSLUCENT_GLINT_ITEM_OVERLAY)
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
        // Only what the code reaches: a pack may declare an image it never touches (Pegasus's
        // gbuffers declare the light volume they sample), and binding it as storage too would
        // conflict with the sampling.
        let used: std::collections::HashSet<naga::Handle<naga::GlobalVariable>> = m
            .functions
            .iter()
            .map(|(_, f)| f)
            .chain(m.entry_points.iter().map(|e| &e.function))
            .flat_map(|f| f.expressions.iter())
            .filter_map(|(_, e)| match e {
                naga::Expression::GlobalVariable(g) => Some(*g),
                _ => None,
            })
            .collect();
        for (gh, g) in m.global_variables.iter() {
            if !used.contains(&gh) {
                continue;
            }
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
                    let atomic = wgpu::StorageTextureAccess::Atomic;
                    let access = match (load, store) {
                        _ if access.contains(naga::StorageAccess::ATOMIC) => atomic,
                        (true, false) => wgpu::StorageTextureAccess::ReadOnly,
                        (false, true) => wgpu::StorageTextureAccess::WriteOnly,
                        _ => wgpu::StorageTextureAccess::ReadWrite,
                    };
                    match images.iter_mut().find(|i| i.binding == b.binding) {
                        // A stage's atomic stores decide (`atomic_vertex_stores`).
                        Some(i) if i.access == atomic || access == atomic => i.access = atomic,
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
    color: &dyn Fn(usize) -> Option<&'a wgpu::TextureView>,
    entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
) {
    let Some(iris) = iris else { return };
    for &(binding, k) in &pass.images {
        let view = match k.checked_sub(COLOR_IMAGE) {
            Some(n) => color(n),
            None => iris.image_view(k),
        };
        if let Some(view) = view {
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
    pub(super) fn iris_begin_frame(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(st) = state(self) {
            let remade = {
                let mut s = st.borrow_mut();
                s.iris.clear_images(encoder);
                s.iris.ran.iter_mut().for_each(|r| *r = false);
                take_snapshots(&mut s, device, encoder, None, &|n| self.color_view(n).map(|v| v.texture().clone()))
            };
            if remade {
                self.clear_all_groups();
            }
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
        // Images it samples while it writes them: copied as the program begins.
        if let Some(st) = state(self) {
            let mut s = st.borrow_mut();
            if !s.snapshots.is_empty() {
                take_snapshots(&mut s, device, encoder, Some(&c.iris.textures), &|n| self.color_view(n).map(|v| v.texture().clone()));
            }
        }
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
            iris_entries(iris, &c.iris, &|n| self.color_view(n), &mut entries);
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

/// Floodfill diagnostics, from Iris Features' page: every 4 seconds or so, four frames in a row
/// of the voxels and light volumes go to the log, and where each changed between frames that
/// write the same light volume. A light that flickers shows as voxels that keep changing.
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
                    let tight: Vec<u8> = (0..rows)
                        .flat_map(|y| data[y * row..y * row + (size[0] * bpt) as usize].iter().copied())
                        .collect();
                    let texels = tight.chunks_exact(*bpt as usize);
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
                    // The 64 voxels around the camera each way, for where they changed.
                    let crop = size.map(|n| n.min(64));
                    let at = |n: u32, c: u32| ((n - c) / 2) as usize;
                    let (x0, y0, z0) = (at(size[0], crop[0]), at(size[1], crop[1]), at(size[2], crop[2]));
                    let b = *bpt as usize;
                    let mut near = Vec::with_capacity((crop[0] * crop[1] * crop[2]) as usize * b);
                    for z in z0..z0 + crop[2] as usize {
                        for y in y0..y0 + crop[1] as usize {
                            let start = ((z * size[1] as usize + y) * size[0] as usize + x0) * b;
                            near.extend_from_slice(&tight[start..start + crop[0] as usize * b]);
                        }
                    }
                    s.probed.push((p.frame, name.clone(), near, crop, *bpt));
                }
                buffer.unmap();
            }
            log::info!("{line} images cleared each frame: {}", s.iris.can_clear);
        }
        // All four frames read back (or nothing more coming): where they changed.
        if let Some(last) = s.probed.iter().map(|p| p.0).find(|f| f % 256 == 3) {
            let all = (last.saturating_sub(3)..=last).all(|f| s.probed.iter().any(|p| p.0 == f));
            if all || s.probes.is_empty() {
                log_changes(&std::mem::take(&mut s.probed));
            }
        }
        // Four frames in a row, every 256 frames.
        if !on || s.frames % 256 > 3 || s.probes.len() > 8 {
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

/// Logs where the probed images changed: the voxels between consecutive frames, each light
/// volume between frames two apart (the frames that write it). Where the light changed most,
/// and the voxels around it, say which blocks it is.
fn log_changes(probed: &[(u64, String, Vec<u8>, [u32; 3], u32)]) {
    let find = |frame: u64, name: &str| probed.iter().find(|p| p.0 == frame && p.1 == name);
    let Some(first) = probed.iter().map(|p| p.0).min() else { return };
    let voxels = |frame: u64| find(frame, "imgVoxelMask");
    let id_at = |v: &(u64, String, Vec<u8>, [u32; 3], u32), x: i64, y: i64, z: i64| -> Option<u32> {
        let [w, h, d] = v.3;
        if x < 0 || y < 0 || z < 0 || x >= w as i64 || y >= h as i64 || z >= d as i64 {
            return None;
        }
        let i = ((z as usize * h as usize + y as usize) * w as usize + x as usize) * v.4 as usize;
        Some(u32::from(v.2[i]) | u32::from(*v.2.get(i + 1).unwrap_or(&0)) << 8)
    };
    let mut line = String::from("floodfill changes:");
    if let (Some(a), Some(b)) = (voxels(first), voxels(first + 1)) {
        let n = a.2.chunks_exact(a.4 as usize).zip(b.2.chunks_exact(b.4 as usize)).filter(|(x, y)| x != y).count();
        line += &format!(" voxels changed between frames {n};");
    }
    let names: std::collections::BTreeSet<&str> =
        probed.iter().map(|p| p.1.as_str()).filter(|n| n.starts_with("imgLpv")).collect();
    for name in names {
        for f in [first, first + 1] {
            let (Some(a), Some(b)) = (find(f, name), find(f + 2, name)) else { continue };
            let bpt = a.4 as usize;
            let [w, h, _] = a.3;
            let mut changed = 0usize;
            let mut worst: Vec<(u32, usize)> = Vec::new();
            for (k, (x, y)) in a.2.chunks_exact(bpt).zip(b.2.chunks_exact(bpt)).enumerate() {
                let delta = x.iter().zip(y).map(|(p, q)| u32::from(p.abs_diff(*q))).max().unwrap_or(0);
                if delta > 1 {
                    changed += 1;
                    worst.push((delta, k));
                }
            }
            worst.sort_by(|p, q| q.0.cmp(&p.0));
            worst.truncate(4);
            line += &format!(" {name} frames {f}/{}: {changed} voxels changed by more than 1", f + 2);
            for (delta, k) in worst {
                let (x, y, z) = ((k % w as usize) as i64, ((k / w as usize) % h as usize) as i64, (k / (w * h) as usize) as i64);
                let around: Vec<u32> = voxels(f)
                    .map(|v| {
                        [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
                            .iter()
                            .filter_map(|(dx, dy, dz)| {
                                // The voxel volume may be larger than the light volume; both
                                // are centred on the camera.
                                let off = |vw: u32, lw: u32| (i64::from(vw) - i64::from(lw)) / 2;
                                id_at(v, x + dx + off(v.3[0], a.3[0]), y + dy + off(v.3[1], a.3[1]), z + dz + off(v.3[2], a.3[2]))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let c = |v: i64, n: u32| v - i64::from(n / 2);
                line += &format!(
                    " [{},{},{} from the camera, by {delta}; voxel and neighbours {around:?}]",
                    c(x, a.3[0]),
                    c(y, a.3[1]),
                    c(z, a.3[2])
                );
            }
            line.push(';');
        }
    }
    log::info!("{line}");
}

/// Images and storage buffers no stage of a pass writes become read-only in every stage, and
/// those one writes take the same access in every stage (one bind group layout entry serves
/// both; wgpu wants each stage to declare exactly its access). JCL's terrain writes the
/// held-light image from its vertex stage and reads it from the fragment stage.
pub(super) fn unify_storage_access(modules: [&mut naga::Module; 2]) {
    use naga::{AddressSpace, ImageClass, StorageAccess, TypeInner};
    use std::collections::HashMap;
    let mut access: HashMap<(u32, u32), (bool, StorageAccess)> = HashMap::new();
    for m in modules.iter() {
        let Ok(info) = naga::valid::Validator::new(naga::valid::ValidationFlags::empty(), naga::valid::Capabilities::all())
            .validate(m)
        else {
            return;
        };
        for (h, g) in m.global_variables.iter() {
            let Some(b) = &g.binding else { continue };
            let declared = match (&g.space, &m.types[g.ty].inner) {
                (AddressSpace::Storage { access }, _) => *access,
                (_, TypeInner::Image { class: ImageClass::Storage { access, .. }, .. }) => *access,
                _ => continue,
            };
            let written = (0..m.entry_points.len()).any(|i| {
                info.get_entry_point(i)[h].intersects(naga::valid::GlobalUse::WRITE | naga::valid::GlobalUse::ATOMIC)
            });
            let e = access.entry((b.group, b.binding)).or_insert((false, StorageAccess::empty()));
            e.0 |= written;
            e.1 |= declared;
        }
    }
    for m in modules {
        let mut retype = Vec::new();
        for (h, g) in m.global_variables.iter_mut() {
            let Some(b) = &g.binding else { continue };
            let Some(&(written, union)) = access.get(&(b.group, b.binding)) else { continue };
            let want = if written { union | StorageAccess::LOAD } else { StorageAccess::LOAD };
            if let AddressSpace::Storage { access } = &mut g.space {
                *access = want;
            } else {
                retype.push((h, want));
            }
        }
        for (h, want) in retype {
            let ty = m.global_variables[h].ty;
            let TypeInner::Image { dim, arrayed, class: ImageClass::Storage { format, access } } = m.types[ty].inner else {
                continue;
            };
            if access == want {
                continue;
            }
            let name = m.types[ty].name.clone();
            m.global_variables[h].ty = m.types.insert(
                naga::Type {
                    name,
                    inner: TypeInner::Image { dim, arrayed, class: ImageClass::Storage { format, access: want } },
                },
                naga::Span::UNDEFINED,
            );
        }
    }
}

impl PackPost {
    /// `draw_hand` with the held items' translucent quads kept for `iris_draw_hand_translucent`
    /// when the pack has `gbuffers_hand_water`. Whether it drew (else Horizon's own draws).
    pub(super) fn iris_draw_hand(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        frame: &super::super::Frame,
    ) -> bool {
        let Some(st) = state(self) else { return false };
        if st.borrow().hand_translucent.is_none() {
            return false;
        }
        let (opaque, translucent): (Vec<&[super::super::Vertex]>, Vec<&[super::super::Vertex]>) =
            frame.hand.chunks_exact(4).partition(|q| !translucent_hand_quad(q));
        // What the hand holds, in the log when it changes (which quads drew translucent).
        {
            thread_local! { static LAST: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) }; }
            let counts = (opaque.len(), translucent.len());
            if LAST.with(|l| l.replace(counts)) != counts && counts != (0, 0) {
                let layers: std::collections::BTreeSet<u32> = frame.hand.iter().map(|v| v.layer).collect();
                log::info!(
                    "held items: {} quads with gbuffers_hand, {} translucent with gbuffers_hand_water (overlays {layers:?})",
                    counts.0, counts.1
                );
            }
        }
        let opaque: Vec<super::super::Vertex> = opaque.concat();
        st.borrow_mut().hand_translucent_quads = translucent.concat();
        let Some((mut pass, mut geometry)) = self.hand.take() else { return true };
        self.iris_hand_pass(device, queue, encoder, size, frame, &opaque, &mut pass, &mut geometry, "HAND_SOLID");
        self.hand = Some((pass, geometry));
        true
    }

    /// The held items' translucent quads, after translucent terrain.
    pub(super) fn iris_draw_hand_translucent(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        frame: &super::super::Frame,
    ) {
        let Some(st) = state(self) else { return };
        let (taken, quads) = {
            let mut s = st.borrow_mut();
            (s.hand_translucent.take(), std::mem::take(&mut s.hand_translucent_quads))
        };
        let Some((mut pass, mut geometry)) = taken else { return };
        self.iris_hand_pass(device, queue, encoder, size, frame, &quads, &mut pass, &mut geometry, "HAND_TRANSLUCENT");
        st.borrow_mut().hand_translucent = Some((pass, geometry));
    }

    /// One hand pass over `quads` (Horizon's `draw_hand`).
    #[allow(clippy::too_many_arguments)]
    fn iris_hand_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        frame: &super::super::Frame,
        quads: &[super::super::Vertex],
        pass: &mut Pass,
        geometry: &mut super::super::pack_hand::PackHand,
        stage: &str,
    ) {
        let projection = super::super::pack_hand::projection(
            frame.hand_fov_degrees,
            size[0] as f32 / size[1].max(1) as f32,
            frame.depth_far,
            frame.bob,
        );
        geometry.prepare(device, queue, quads, projection);
        if geometry.quads == 0 {
            return;
        }
        let group = self.bind(device, queue, pass, size);
        {
            let mut rp = self.begin(encoder, pass);
            rp.set_pipeline(&pass.pipeline);
            rp.set_bind_group(0, &group, &[]);
            rp.set_bind_group(1, &geometry.group, &[]);
            if pass.draw {
                self.draws.bind(&mut rp, stage);
            }
            rp.set_vertex_buffer(0, geometry.vertices());
            rp.set_index_buffer(geometry.indices.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..geometry.quads * 6, 0, 0..1);
        }
        self.mips_valid &= !pass.written;
    }
}

/// The hit block's outline, drawn by the mod after the pack's deferred programs: Horizon drew
/// it into the pack's colour before them, under the material the pack had for the block behind
/// it, so the pack lit and reflected it as that block (a torch in hand shone off it). Iris
/// draws it with the pack's line program, which packs leave unlit.
pub(super) struct Outline {
    pipeline: wgpu::RenderPipeline,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    edges: wgpu::Buffer,
    capacity: u64,
    layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
}

const OUTLINE_WGSL: &str = "
struct U { view_proj: mat4x4<f32>, screen: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> edges: array<vec4<f32>>;
@vertex fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> {
    // Pulled a little toward the camera, as the game draws outlines, so the block's own faces
    // don't hide them.
    let shrink = 1.0 - 1.0 / 256.0;
    let a = u.view_proj * vec4(edges[2u * i].xyz * shrink, 1.0);
    let b = u.view_proj * vec4(edges[2u * i + 1u].xyz * shrink, 1.0);
    let p = select(a, b, v >= 2u);
    let d = normalize((b.xy / b.w - a.xy / a.w) * u.screen.xy + vec2(1e-6, 0.0));
    var o = vec2(-d.y, d.x) * u.screen.z / u.screen.xy;
    if (v % 2u) != 0u { o = -o; }
    // GL's clip depth to wgpu's, as the pack's programs write it.
    return vec4(p.xy + o * p.w, (p.z + p.w) * 0.5, p.w);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4(0.0, 0.0, 0.0, 0.4); }
";

impl Outline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, depth: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Iris Features outline"),
            source: wgpu::ShaderSource::Wgsl(OUTLINE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Iris Features outline"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Iris Features outline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Iris Features outline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
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
                    // The alpha too: packs keep translucent colour in a buffer of its own and
                    // blend it by its alpha later.
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Iris Features outline"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (edges, group) = Self::edges(device, &layout, &uniform, 64);
        Self { pipeline, group, uniform, edges, capacity: 64, layout, format }
    }

    fn edges(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        count: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let edges = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Iris Features outline edges"),
            size: count * 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Iris Features outline"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: edges.as_entire_binding() },
            ],
        });
        (edges, group)
    }
}

impl PackPost {
    /// The hit block's outline into the buffer translucent terrain draws its colour to.
    pub(super) fn iris_draw_outline(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &super::super::Frame,
    ) {
        if frame.outline.is_empty() {
            return;
        }
        let Some(st) = state(self) else { return };
        let Some(targets) = self.targets.as_ref() else { return };
        let buffer = self
            .water
            .as_ref()
            .and_then(|w| w.writes.iter().find(|(l, _)| *l == 0).map(|w| w.1))
            .unwrap_or(0);
        let Some(color) = targets.color[buffer].as_ref() else { return };
        let view = &color[usize::from(self.main & 1 << buffer != 0)];
        let format = self.formats[buffer];
        let depth_format = targets.depth_texture.format();
        let matrix = |name: &str| match super::pack_uniforms::index(name).and_then(|i| self.uniforms.values.get(i)) {
            Some(super::pack_uniforms::Value::F(f)) => glam::Mat4::from_cols_array(f),
            _ => glam::Mat4::IDENTITY,
        };
        let view_proj = matrix("gbufferProjection") * matrix("gbufferModelView");
        let size = targets.size;
        let mut s = st.borrow_mut();
        if s.outline.as_ref().is_none_or(|o| o.format != format) {
            s.outline = Some(Outline::new(device, format, depth_format));
        }
        let o = s.outline.as_mut().expect("made");
        let camera = frame.camera;
        let points: Vec<[f32; 4]> = frame
            .outline
            .iter()
            .flat_map(|e| {
                [
                    [(e[0] - camera.x) as f32, (e[1] - camera.y) as f32, (e[2] - camera.z) as f32, 1.0],
                    [(e[3] - camera.x) as f32, (e[4] - camera.y) as f32, (e[5] - camera.z) as f32, 1.0],
                ]
            })
            .collect();
        let count = frame.outline.len() as u64;
        if count > o.capacity {
            o.capacity = count.next_power_of_two();
            (o.edges, o.group) = Outline::edges(device, &o.layout, &o.uniform, o.capacity);
        }
        queue.write_buffer(&o.edges, 0, bytemuck::cast_slice(&points));
        // Line width in pixels: the game's 2.5 at 1080 lines, scaled with the render size.
        let width = (size[1] as f32 / 1080.0 * 2.5).max(1.0) * 0.5;
        let mut u: Vec<f32> = view_proj.to_cols_array().to_vec();
        u.extend([size[0] as f32 * 0.5, size[1] as f32 * 0.5, width, 0.0]);
        queue.write_buffer(&o.uniform, 0, bytemuck::cast_slice(&u));
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Iris Features outline"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rp.set_pipeline(&o.pipeline);
        rp.set_bind_group(0, &o.group, &[]);
        rp.draw(0..4, 0..count as u32);
    }
}

/// naga's storage format for a texture format images use.
fn storage_format(f: wgpu::TextureFormat) -> Option<naga::StorageFormat> {
    use naga::StorageFormat as S;
    use wgpu::TextureFormat as T;
    Some(match f {
        T::R8Unorm => S::R8Unorm,
        T::R8Snorm => S::R8Snorm,
        T::R8Uint => S::R8Uint,
        T::R8Sint => S::R8Sint,
        T::R16Uint => S::R16Uint,
        T::R16Sint => S::R16Sint,
        T::R16Float => S::R16Float,
        T::R16Unorm => S::R16Unorm,
        T::Rg8Unorm => S::Rg8Unorm,
        T::Rg8Uint => S::Rg8Uint,
        T::R32Uint => S::R32Uint,
        T::R32Sint => S::R32Sint,
        T::R32Float => S::R32Float,
        T::Rg16Uint => S::Rg16Uint,
        T::Rg16Sint => S::Rg16Sint,
        T::Rg16Float => S::Rg16Float,
        T::Rg16Unorm => S::Rg16Unorm,
        T::Rgba8Unorm => S::Rgba8Unorm,
        T::Rgba8Snorm => S::Rgba8Snorm,
        T::Rgba8Uint => S::Rgba8Uint,
        T::Rgba8Sint => S::Rgba8Sint,
        T::Rgb10a2Unorm => S::Rgb10a2Unorm,
        T::Rg11b10Ufloat => S::Rg11b10Ufloat,
        T::Rg32Uint => S::Rg32Uint,
        T::Rg32Sint => S::Rg32Sint,
        T::Rg32Float => S::Rg32Float,
        T::Rgba16Uint => S::Rgba16Uint,
        T::Rgba16Sint => S::Rgba16Sint,
        T::Rgba16Float => S::Rgba16Float,
        T::Rgba16Unorm => S::Rgba16Unorm,
        T::Rgba32Uint => S::Rgba32Uint,
        T::Rgba32Sint => S::Rgba32Sint,
        T::Rgba32Float => S::Rgba32Float,
        _ => return None,
    })
}

/// Each storage image the modules declare takes the format of the pack's image of its name: a
/// pack may declare an image without one (`writeonly uniform image3D img;`), and the frontend
/// gives those a placeholder.
pub(super) fn declared_image_formats(module: &mut naga::Module) {
    use naga::{ImageClass, TypeInner};
    let images: Vec<(String, wgpu::TextureFormat)> =
        with_build(|b| b.images.iter().map(|i| (i.0.clone(), i.2)).collect()).unwrap_or_default();
    let mut retype = Vec::new();
    for (h, g) in module.global_variables.iter() {
        let Some(name) = &g.name else { continue };
        let TypeInner::Image { class: ImageClass::Storage { format, .. }, .. } = module.types[g.ty].inner else { continue };
        let Some(want) = images.iter().find(|i| &i.0 == name).and_then(|i| storage_format(i.1)) else { continue };
        if want != format {
            retype.push((h, want));
        }
    }
    for (h, want) in retype {
        let ty = module.global_variables[h].ty;
        let TypeInner::Image { dim, arrayed, class: ImageClass::Storage { mut access, .. } } = module.types[ty].inner else { continue };
        // Atomic stores (`atomic_vertex_stores`) need a 32-bit integer format: on another, plain
        // stores again.
        if access.contains(naga::StorageAccess::ATOMIC)
            && !matches!(want, naga::StorageFormat::R32Uint | naga::StorageFormat::R32Sint)
        {
            access.remove(naga::StorageAccess::ATOMIC);
            unatomic_stores(module, h);
        }
        let name = module.types[ty].name.clone();
        module.global_variables[h].ty = module.types.insert(
            naga::Type { name, inner: TypeInner::Image { dim, arrayed, class: ImageClass::Storage { format: want, access } } },
            naga::Span::UNDEFINED,
        );
    }
}

/// `atomic_vertex_stores` undone for `image`: its atomics become the stores they were.
fn unatomic_stores(module: &mut naga::Module, image: naga::Handle<naga::GlobalVariable>) {
    use naga::{Expression, Statement};
    fn fix(block: &mut naga::Block, exprs: &naga::Arena<Expression>, image: naga::Handle<naga::GlobalVariable>) {
        for s in block.iter_mut() {
            match s {
                Statement::ImageAtomic { image: i, coordinate, array_index, value, .. }
                    if matches!(exprs[*i], Expression::GlobalVariable(g) if g == image) =>
                {
                    if let Expression::AccessIndex { base, index: 0 } = exprs[*value] {
                        *s = Statement::ImageStore { image: *i, coordinate: *coordinate, array_index: *array_index, value: base };
                    }
                }
                Statement::Block(b) => fix(b, exprs, image),
                Statement::If { accept, reject, .. } => {
                    fix(accept, exprs, image);
                    fix(reject, exprs, image);
                }
                Statement::Switch { cases, .. } => cases.iter_mut().for_each(|c| fix(&mut c.body, exprs, image)),
                Statement::Loop { body, continuing, .. } => {
                    fix(body, exprs, image);
                    fix(continuing, exprs, image);
                }
                _ => {}
            }
        }
    }
    for (_, f) in module.functions.iter_mut() {
        fix(&mut f.body, &f.expressions, image);
    }
    for ep in module.entry_points.iter_mut() {
        fix(&mut ep.function.body, &ep.function.expressions, image);
    }
}

/// `PassIris::images` from here up: colour buffer `k - COLOR_IMAGE` as an image (`colorimgN`).
pub(super) const COLOR_IMAGE: usize = 1 << 20;

thread_local! {
    // The colour buffers' formats while a pack's passes are built, those that can be images.
    static COLOR_FORMATS: std::cell::Cell<[Option<wgpu::TextureFormat>; BUFFERS]> = const { std::cell::Cell::new([None; BUFFERS]) };
}

/// Whether buffers of `format` can be bound as storage images on this device.
pub(super) fn storage_capable(format: wgpu::TextureFormat, features: wgpu::Features) -> bool {
    format.guaranteed_format_features(features).allowed_usages.contains(wgpu::TextureUsages::STORAGE_BINDING)
}

/// The colour buffers' formats, for the passes about to be built.
pub(super) fn set_color_formats(formats: [wgpu::TextureFormat; BUFFERS], features: wgpu::Features) {
    COLOR_FORMATS.with(|c| c.set(formats.map(|f| storage_capable(f, features).then_some(f))));
}

/// Colour buffer `n`'s format, when it can be an image.
pub(super) fn color_format(n: usize) -> Option<wgpu::TextureFormat> {
    COLOR_FORMATS.with(|c| c.get()[n])
}

/// The colour buffers a pass binds as images.
pub(super) fn color_images(pass: &PassIris) -> u16 {
    pass.images.iter().filter_map(|(_, k)| k.checked_sub(COLOR_IMAGE)).fold(0, |m, n| m | 1 << n)
}

impl PackPost {
    /// Colour buffer `n`'s side programs read now.
    pub(super) fn color_view(&self, n: usize) -> Option<&wgpu::TextureView> {
        let t = self.targets.as_ref()?;
        t.color.get(n)?.as_ref().map(|c| &c[usize::from(self.main & 1 << n != 0)])
    }
}

/// While a pack runs, a GPU validation error goes to the log instead of ending the game (wgpu's
/// default): a pack's program the mod can't yet give everything it needs then draws wrongly or
/// not at all, and the player can switch packs. Each message is logged once.
pub(super) fn log_gpu_errors(device: &wgpu::Device) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
        // The reasons wgpu nests under "Validation Error", each.
        let text = match &e {
            wgpu::Error::Validation { description, .. } | wgpu::Error::Internal { description, .. } => {
                description.split_whitespace().collect::<Vec<_>>().join(" ")
            }
            other => other.to_string(),
        };
        let Ok(mut seen) = SEEN.lock() else { return };
        if seen.len() < 64 && !seen.contains(&text) {
            log::error!("GPU error under the shader pack (the game goes on): {text}");
            seen.push(text);
        }
    }));
}
