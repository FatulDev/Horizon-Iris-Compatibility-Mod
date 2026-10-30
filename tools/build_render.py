# Render and game side of the bundle (exec'd by build.py, which defines the helpers).

RE = 'crates/horizon-client-render/src/'
GA = 'crates/horizon-client/src/'

def unprefix_iris(s):
    # Inside the module, `iris::x` is just `x`.
    return re.sub(r'(?<![\w:])iris::', '', s)

# --- Uniforms -------------------------------------------------------------------------------
# FrameUniforms::update with the view bob in the model-view matrix, as Iris has it.
upd = deindent(take(pristine(RE + 'gfx/pack_uniforms.rs'), r'^    pub fn update\('))
upd = sub1(upd, 'pub fn update(&mut self, frame: &Frame, inputs: &PackInputs, size: [u32; 2]) {',
           'pub(crate) fn update(u: &mut FrameUniforms, frame: &Frame, inputs: &PackInputs, size: [u32; 2]) {')
upd = re.sub(r'\bself\.', 'u.', upd)
upd = sub1(upd, '    let view = crate::camera::view_matrix(frame.yaw, frame.pitch);',
           '    // Iris moves the view bob and the nausea spin from the projection into the model-view\n'
           '    // matrix: packs take gbufferProjection for a plain perspective (reading its diagonal),\n'
           '    // and with the bob in it their clouds, sky and lighting swayed with each step.\n'
           '    let view = frame.bob * frame.spin * crate::camera::view_matrix(frame.yaw, frame.pitch);')
upd = re.sub(r'\)\s*\*\s*frame\.bob\s*\*\s*frame\.spin;', ');', upd, count=1)
assert 'frame.bob * frame.spin * crate' in upd and upd.count('frame.bob') == 1, upd[:1500]
module('render-uniforms', RE + 'gfx/pack_uniforms.rs', 'iris_uniforms',
       own('render_uniforms.rs') + '\n' + upd, 'pub')
hook('update', 'horizon_client_render::gfx::pack_uniforms::FrameUniforms::update', '''\
fn iris_update(u: &mut FrameUniforms, frame: &Frame, inputs: &PackInputs, size: [u32; 2]) -> ::core::ops::ControlFlow<()> {
    iris_uniforms::update(u, frame, inputs, size);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('u', '&mut FrameUniforms', '&mut *self'), binding('frame', '&Frame', 'frame'),
      binding('inputs', '&PackInputs', 'inputs'), binding('size', '[u32; 2]', 'size')])
hook('eval-custom', 'horizon_client_render::gfx::pack_uniforms::FrameUniforms::eval_custom', '''\
fn iris_eval_custom(u: &mut FrameUniforms, custom: &mut horizon_shaders::Custom) -> ::core::ops::ControlFlow<()> {
    iris_uniforms::eval_custom(u, custom);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('u', '&mut FrameUniforms', '&mut *self'), binding('custom', '&mut horizon_shaders::Custom', '&mut *custom')])

hook('hand-projection', 'horizon_client_render::gfx::pack_hand::projection', '''\
fn iris_hand_projection(hud_fov: f32, aspect: f32, far: f32) -> ::core::ops::ControlFlow<Mat4> {
    // The view bob is in gbufferModelView now (Iris's convention), which the hand's programs
    // multiply by as well: in its projection too it would sway twice.
    ::core::ops::ControlFlow::Break(
        Mat4::from_scale(glam::Vec3::new(1.0, 1.0, HAND_DEPTH))
            * Mat4::perspective_rh_gl(hud_fov.to_radians(), aspect, 0.05, far),
    )
}
''', [binding('hud_fov', 'f32', 'hud_fov'), binding('aspect', 'f32', 'aspect'), binding('far', 'f32', 'far')])

hook('flat-draws', 'horizon_client_render::gfx::flat::wanted', '''\
fn iris_flat_draws(value: bool, adapter: &wgpu::Adapter, device: &wgpu::Device) -> bool {
    // A shader pack's terrain programs read `mc_Entity`, the face's normal and tangent from
    // flat draws only (per-section draws give -1 and an up normal): Horizon turns them on
    // only on Metal; the mod wherever the device can (`HORIZON_FLAT_DRAWS=0` still turns
    // them off).
    value || (std::env::var("HORIZON_FLAT_DRAWS").as_deref() != Ok("0") && supported(adapter, device))
}
''', [binding('value', 'bool', '$value'), binding('adapter', '&wgpu::Adapter', 'adapter'),
      binding('device', '&wgpu::Device', 'device')], kind='RETURN')

# --- Device features ----------------------------------------------------------------------------
hook('device-features', 'horizon_client_render::gfx::cull::Cull::features', '''\
fn iris_features(value: wgpu::Features, adapter: &wgpu::Adapter) -> wgpu::Features {
    // Images cleared each frame and written from vertex stages; compute programs' `barrier()`
    // (naga lowers it with subgroup barriers); image atomics, for voxels written in an order of
    // their own (`atomic_vertex_stores`).
    horizon_shaders::iris::set_image_atomics(adapter.features().contains(wgpu::Features::TEXTURE_ATOMIC));
    value
        | (adapter.features()
            & (wgpu::Features::CLEAR_TEXTURE
                | wgpu::Features::VERTEX_WRITABLE_STORAGE
                | wgpu::Features::SUBGROUP
                | wgpu::Features::SUBGROUP_BARRIER
                | wgpu::Features::TEXTURE_ATOMIC))
}
''', [binding('value', 'wgpu::Features', '$value'), binding('adapter', '&wgpu::Adapter', 'adapter')], kind='RETURN')

# --- The pack's passes ------------------------------------------------------------------------
pp_ref = ref(RE + 'gfx/pack_post.rs')
ir = strip_tests(ref(RE + 'gfx/pack_post/iris.rs'))
# The PackPost methods that kept the resources in a field: rewritten in render_iris_pp.rs.
ir, _ = cut(ir, r'^impl PackPost \{\n    /// The bind group entries')
ir, _ = cut(ir, r'^#\[allow\(dead_code\)\]\nfn _uses_entry')
ir, _ = cut(ir, r'^impl PackPost \{\n    /// Every pass')
ir = sub1(ir, 'if f[h].contains(naga::valid::GlobalUse::WRITE) {',
          'if f[h].intersects(naga::valid::GlobalUse::WRITE | naga::valid::GlobalUse::ATOMIC) {')
ir = sub1(ir, '''            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let made = checked(device, || {''', '''            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        // Vertex stages store into 32-bit integer images atomically (`atomic_vertex_stores`).
        let atomic = matches!(self.format, wgpu::TextureFormat::R32Uint | wgpu::TextureFormat::R32Sint)
            && horizon_shaders::iris::image_atomics();
        let usage = if atomic { usage | wgpu::TextureUsages::STORAGE_ATOMIC } else { usage };
        let made = checked(device, || {''')
# Raw custom textures of other formats than RGBA8: one to four channels of bytes, 16-bit
# unsigned or half floats, or floats (Noble's cloud noise is RGB8); three channels gain an
# opaque alpha.
ir = sub1(ir, """            let per = match (raw.internal.as_str(), raw.pixel_type.as_str()) {
                ("RGBA8" | "RGBA", "UNSIGNED_BYTE") => 4,
                (i, t) => return Err(format!("raw {i} {t} is not supported")),
            };
            let n = (raw.size[0] * raw.size[1] * raw.size[2]) as usize * per;
            if bytes.len() < n {
                return Err(format!("{} bytes, {n} expected", bytes.len()));
            }""", """            let Some((fmt, channels, size_of)) = raw_layout(&raw.internal, &raw.pixel_type) else {
                return Err(format!("raw {} {} is not supported", raw.internal, raw.pixel_type));
            };
            format = fmt;
            let texels = (raw.size[0] * raw.size[1] * raw.size[2]) as usize;
            let n = texels * channels * size_of;
            if bytes.len() < n {
                return Err(format!("{} bytes, {n} expected", bytes.len()));
            }
            let data = widen_rgb(&bytes[..n], channels, size_of, &raw.pixel_type);
            let per = data.len() / texels.max(1);""")
ir = sub1(ir, "            (d, v, raw.size, bytes[..n].to_vec(), per as u32)", "            (d, v, raw.size, data, per as u32)")
ir = sub1(ir, """    let (dimension, view_dimension, size, texels, bytes_per_texel) = match &decl.raw {""",
          """    let mut format = wgpu::TextureFormat::Rgba8Unorm;
    let (dimension, view_dimension, size, texels, bytes_per_texel) = match &decl.raw {""")
ir = sub1(ir, """            dimension,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,""", """            dimension,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,""")
ir += open(os.path.join(ROOT, 'src', 'render_raw_textures.rs')).read()
ir = sub1(ir, 'use horizon_shaders::{Group, Interface, WorkGroups};\n',
          'use horizon_shaders::iris::WorkGroups;\nuse horizon_shaders::{Group, Interface};\n')
ir = sub1(ir, 'use super::{BUFFERS, Binding, PackPost, SamplerKind, Source, checked, entry, source};\n', 'use super::*;\n')
# Storage uses come from the modules (the translator's interface has no field for them).
ir = sub1(ir, '''    name: &str,
    interface: &Interface,
    modules: &[(wgpu::ShaderStages, &naga::Module)],''', '''    name: &str,
    storage: &(Vec<ImageUse>, Vec<u32>),
    modules: &[(wgpu::ShaderStages, &naga::Module)],''')
ir = sub1(ir, '    for u in &interface.images {', '    for u in &storage.0 {')
ir = sub1(ir, '    for &index in &interface.buffers {', '    for &index in &storage.1 {')
s0 = ir.index('        if let Some(q) = qualifier_format(&u.format)')
s1 = ir.index('        let (stages, writes) = uses(modules, u.binding);')
ir = ir[:s0] + ir[s1:]
a0 = ir.index('        let access = match u.access.as_str() {')
a1 = ir.index('};', a0) + 3
ir = ir[:a0] + '        let access = u.access;\n' + ir[a1:]
ir = sub1(ir, '    let mut iris = storage_entries(&c.name, interface, &modules, &mut entries, notes)?;',
          '    let uses = storage_uses(&modules);\n    let mut iris = storage_entries(&c.name, &uses, &modules, &mut entries, notes)?;')
ir, _ = cut(ir, r'^pub\(super\) fn qualifier_format\(')
ir = sub1(ir, 'pub(super) struct Iris {\n    pub resources: Resources,', 'pub(super) struct Iris {\n    #[allow(dead_code)]\n    pub resources: Resources,')
# Iris's precedence of `texture.<stage>.colortexN`: it replaces buffer N in that stage's programs
# until one of them has written (flipped) the buffer; after that they read the buffer.
ir = sub1(ir, '    pub features: wgpu::Features,\n}', '    pub features: wgpu::Features,\n    /// (stage, order, buffers written) of every post program.\n    pub writes: Vec<(&\'static str, u32, u16)>,\n}')
ir = sub1(ir, '            features,\n        }\n    }', '            features,\n            writes: Vec::new(),\n        }\n    }')
ir = sub1(ir, """        let horizon_has = s.name != "noisetex"
            && source(&s.name, group.shadow(), shadow.is_some(), watershadow) != Source::Dummy;""", """        let horizon_source = source(&s.name, group.shadow(), shadow.is_some(), watershadow);
        let overridable = match horizon_source {
            Source::Color(n) => !group.shadow() && flipped & 1 << n == 0,
            _ => false,
        };
        let horizon_has = s.name != "noisetex" && horizon_source != Source::Dummy && !overridable;""")
ir = sub1(ir, '    let stage = stage_of(group);\n', '    let stage = stage_of(group);\n    let flipped = flipped_before(stage, program_order(group, name));\n')
ir = sub1(ir, 'c: horizon_shaders::ComputePass,', 'c: horizon_shaders::iris::ComputePass,')
ir = sub1(ir, 'post: &mut horizon_shaders::IrisPost,', 'post: &mut horizon_shaders::iris::IrisPost,')
ir = ir.replace('super::pack_uniforms::iris_index(', 'super::pack_uniforms::iris_uniforms::iris_index(')

copies = []
# PackPost::new → PackPost::iris_new, with the mod's resources.
new = take(pp_ref, r'^    pub fn new\(')
new = sub1(new, 'pub fn new(', 'pub(crate) fn iris_new(')
new = sub1(new, 'iris::set_build(Some(iris.context(features)));', 'iris::set_build(Some(iris.context(features)));\n        iris::record_writes(&post.programs);')
new = sub1(new, 'let mut iris_post = post.iris;', 'let mut iris_post = horizon_shaders::iris::take_pending();')
new = sub1(new, 'custom.forget_smoothing();', 'horizon_shaders::iris::forget_smoothing(&mut custom);')
new = sub1(new, '.flat_map(|p| p.iris.textures.iter().copied())', '.flat_map(|p| built_pass(&p.ubo).textures)')
new = sub1(new, '        Self {\n            notes,\n', '        let pp = Self {\n            notes,\n')
new = sub1(new, '            iris: Some(iris),\n        }\n    }', '        };\n        register(&pp, iris);\n        pp\n    }')
new = new.replace('pack_uniforms::iris_index(', 'pack_uniforms::iris_uniforms::iris_index(')
# Translucent particles with `gbuffers_particles_translucent`, after deferred under `mixed`.
new = sub1(new, '        let programs = [\n            post.entities,', '''        let particles_translucent = iris_post.particles_translucent.take();
        let particles_translucent_late = post.particles_late;
        let particles_late = if particles_translucent.is_some() {
            iris_post.particles_opaque_late
        } else {
            post.particles_late
        };
        let programs = [
            post.entities,''')
new = sub1(new, 'let late = [false, false, post.translucent_late, post.particles_late];',
           'let late = [false, false, post.translucent_late, particles_late];')
new = sub1(new, '''        let shadow_entities = shadow_programs.map(|p| {''', '''        let particles_translucent = particles_translucent
            .zip(entity_geometry.as_ref())
            .and_then(|(p, g)| build(p, Some(g)))
            .filter(|_| entity_passes[super::pack_entities::PARTICLE].is_some());
        let shadow_entities = shadow_programs.map(|p| {''')
# Rain and snow write depth only with `rainDepth=true`, as in Iris: the composite passes would
# otherwise take each streak for a surface right in front of the camera.
new = sub1(new, '''        let weather_geometry = world.weather.then(|| Geometry {
            buffers: &entity_buffers,
            table: None,
            depth_write: true,''', '''        let weather_geometry = world.weather.then(|| Geometry {
            buffers: &entity_buffers,
            table: None,
            depth_write: iris_post.rain_depth,''')
new = sub1(new, '        let pp = Self {\n', '        let mut pp = Self {\n')
new = sub1(new, '        register(&pp, iris);\n', '''        register(&pp, iris);
        if let Some(p) = &particles_translucent {
            pp.notes.push(format!("translucent particles draw with {}", p.name));
        }
        set_particles_translucent(&pp, particles_translucent, particles_translucent_late);
''')
copies.append(new)
# bind, run_shadow, run.
bind = take(pp_ref, r'^    fn bind\(')
bind = sub1(bind, '    fn bind(', '    pub(super) fn iris_bind(')
bind = sub1(bind, '        let targets = self.targets.as_ref().expect("resized");\n', '')
bind = sub1(bind, '        let group = pass.groups.entry(key)', '''        let st = state(self);
        let guard = st.as_ref().map(|s| s.borrow());
        let iris = guard.as_deref().map(|s| &s.iris);
        let pi = guard
            .as_deref()
            .and_then(|s| s.passes.get(&pass.ubo))
            .cloned()
            .unwrap_or_default();
        let group = pass.groups.entry(key)''')
bind = sub1(bind, '''            self.sampler_entries(
                pass.group,
                &pass.samplers,
                &pass.iris.textures,''', '''            self.sampler_entries(
                iris,
                pass.group,
                &pass.samplers,
                &pi.textures,''')
bind = sub1(bind, 'self.iris_entries(&pass.iris, &mut entries);', 'iris_entries(iris, &pi, &mut entries);')
copies.append(bind)
rs = take(pp_ref, r'^    fn run_shadow\(')
rs = sub1(rs, '    fn run_shadow(', '    pub(super) fn iris_run_shadow(')
copies.append(rs)
run = take(pp_ref, r'^    pub fn run\(')
run = sub1(run, '    pub fn run(', '    pub(super) fn iris_run(')
run = sub1(run, '''        let notes = &mut self.notes;
        if self.iris.as_mut().is_some_and(|i| i.resize(device, size, notes)) {
            self.clear_all_groups();
        }
''', '''        self.iris_resize(device, size);
        // Horizon drops the post and terrain passes' bind groups with the old targets; the
        // others (entities, particles, sky, hand...) may sample them too.
        if resized {
            self.clear_all_groups();
        }
''')
run = sub1(run, '        self.resize(device, size);\n',
           '        let resized = self.targets.as_ref().is_none_or(|t| t.size != size);\n        self.resize(device, size);\n')
copies.append(run)
impl = 'impl PackPost {\n' + re.sub(r'(?<![\w:])super::', 'super::super::', '\n'.join(copies)) + '}\n'
# build_pass.
bp = take(pp_ref, r'^fn build_pass\(')
bp = sub1(bp, '''    iris::read_only_where_unwritten(&mut p.program.vertex);''', '''    let uses = storage_uses(&[
        (wgpu::ShaderStages::VERTEX, &p.program.vertex),
        (wgpu::ShaderStages::FRAGMENT, &p.program.fragment),
    ]);
    iris::read_only_where_unwritten(&mut p.program.vertex);''')
bp = sub1(bp, '''        &p.name,
        &p.program.interface,
        &modules,''', '''        &p.name,
        &uses,
        &modules,''')
bp = sub1(bp, '        iris: pass_iris,\n', '')
m = re.search(r'    Some\(Pass \{\n        ubo: (device\.create_buffer\(&wgpu::BufferDescriptor \{.*?\n        \}\)),\n', bp, re.S)
bp = bp[:m.start()] + '    let ubo = ' + m.group(1).replace('\n    ', '\n') + ';\n    record_pass(&ubo, pass_iris);\n    Some(Pass {\n        ubo,\n' + bp[m.end():]
bp = re.sub(r'(?<![\w:])super::', 'super::super::', bp)
bp = bp.replace('pack_uniforms::iris_index(', 'pack_uniforms::iris_uniforms::iris_index(')
render_iris = (ir.rstrip() + '\n\n' + own('render_iris_pp.rs') + '\n// Horizon\'s, with the mod\'s resources.\n'
               + unprefix_iris(bp) + '\n' + unprefix_iris(impl))
module('render-pack-post', RE + 'gfx/pack_post.rs', 'iris', render_iris)

hook('bind', 'horizon_client_render::gfx::pack_post::PackPost::bind', '''\
fn iris_bind(
    pp: &PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &mut Pass,
    out: [u32; 2],
) -> ::core::ops::ControlFlow<wgpu::BindGroup> {
    ::core::ops::ControlFlow::Break(pp.iris_bind(device, queue, pass, out))
}
''', [binding('pp', '&PackPost', 'self'), binding('device', '&wgpu::Device', 'device'),
      binding('queue', '&wgpu::Queue', 'queue'), binding('pass', '&mut Pass', '&mut *pass'),
      binding('out', '[u32; 2]', 'out')])
hook('run-shadow', 'horizon_client_render::gfx::pack_post::PackPost::run_shadow', '''\
fn iris_run_shadow(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    terrain: &mut TerrainDraw<'_>,
) -> ::core::ops::ControlFlow<()> {
    pp.iris_run_shadow(device, queue, encoder, terrain);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('pp', '&mut PackPost', '&mut *self'), binding('device', '&wgpu::Device', 'device'),
      binding('queue', '&wgpu::Queue', 'queue'), binding('encoder', '&mut wgpu::CommandEncoder', '&mut *encoder'),
      binding('terrain', "&mut TerrainDraw<'_>", '&mut *terrain')])
hook('run', 'horizon_client_render::gfx::pack_post::PackPost::run', '''\
fn iris_run(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    world: &wgpu::TextureView,
    world_depth: &wgpu::TextureView,
    screen: &wgpu::TextureView,
    frame: &super::Frame,
    inputs: &PackInputs,
    size: [u32; 2],
    window: [u32; 2],
    pages: &[(u32, wgpu::Buffer)],
    terrain: &mut TerrainDraw<'_>,
) -> ::core::ops::ControlFlow<()> {
    pp.iris_run(device, queue, encoder, world, world_depth, screen, frame, inputs, size, window, pages, terrain);
    pp.probe_floodfill(device, encoder);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('pp', '&mut PackPost', '&mut *self'), binding('device', '&wgpu::Device', 'device'),
      binding('queue', '&wgpu::Queue', 'queue'), binding('encoder', '&mut wgpu::CommandEncoder', '&mut *encoder'),
      binding('world', '&wgpu::TextureView', 'world'), binding('world_depth', '&wgpu::TextureView', 'world_depth'),
      binding('screen', '&wgpu::TextureView', 'screen'), binding('frame', '&super::Frame', 'frame'),
      binding('inputs', '&PackInputs', 'inputs'), binding('size', '[u32; 2]', 'size'),
      binding('window', '[u32; 2]', 'window'), binding('pages', '&[(u32, wgpu::Buffer)]', 'pages'),
      binding('terrain', "&mut TerrainDraw<'_>", '&mut *terrain')])

# --- Translucent particles: their own program (`gbuffers_particles_translucent`) -----------------
module('render-pack-entities', RE + 'gfx/pack_entities.rs', 'iris_entities', own('render_iris_entities.rs'), 'pub')
hook('prepare-entities', 'horizon_client_render::gfx::pack_post::PackPost::prepare_entities', '''\
fn iris_prepare_entities(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    entities: &[super::Vertex],
    spans: &[super::pack_draw::EntitySpan],
    rest: &mut Vec<super::Vertex>,
) -> ::core::ops::ControlFlow<()> {
    pp.iris_prepare_entities(device, queue, entities, spans, rest);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('pp', '&mut PackPost', '&mut *self'), binding('device', '&wgpu::Device', 'device'),
      binding('queue', '&wgpu::Queue', 'queue'), binding('entities', '&[super::Vertex]', 'entities'),
      binding('spans', '&[super::pack_draw::EntitySpan]', 'spans'),
      binding('rest', '&mut Vec<super::Vertex>', '&mut *rest')])
hook('draw-entities', 'horizon_client_render::gfx::pack_post::PackPost::draw_entities', '''\
fn iris_draw_entities(
    pp: &mut PackPost,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    size: [u32; 2],
    late: bool,
) -> ::core::ops::ControlFlow<()> {
    pp.iris_draw_entities(device, queue, encoder, size, late);
    ::core::ops::ControlFlow::Break(())
}
''', [binding('pp', '&mut PackPost', '&mut *self'), binding('device', '&wgpu::Device', 'device'),
      binding('queue', '&wgpu::Queue', 'queue'), binding('encoder', '&mut wgpu::CommandEncoder', '&mut *encoder'),
      binding('size', '[u32; 2]', 'size'), binding('late', 'bool', 'late')])

# --- `at_midBlock` (Iris), and `mc_Entity` by sprite and glow ----------------------------------
# Eclipse's floodfill voxelises terrain at vertex + at_midBlock/64; without it every block landed
# on whichever voxel its corners rounded to, which changed as the camera moved. The id table
# (`client_block_ids.rs`) gives ids by sprite layer and HDR tier, and which faces place their
# block: faces whose sprite other blocks share with other ids are moved out of the voxel volume
# when the block's other faces place it (else one block's faces wrote two ids, and lights
# flickered with the order the GPU drew them in).
ti = take(pristine(RE + 'gfx/pack_atlas.rs'), r'^    pub fn terrain_inputs\(')
ti = sub1(ti, '    pub fn terrain_inputs(', '    pub(crate) fn iris_terrain_inputs(')
ti = sub1(ti, '            ("at_tangent", "hz_tangent"),', '            ("at_tangent", "hz_tangent"),\n            ("at_midBlock", mid_block),')
ti = sub1(ti, '        if flat {\n            body.push_str(FACE);\n        }',
          '        if flat {\n            body.push_str(FACE);\n            body.push_str(IRIS_IDS);\n        }')
ti = sub1(ti, '"vec4(float(hz_block_ids[2u * hz_layer]), float(hz_block_ids[2u * hz_layer + 1u]), 0.0, 0.0)"',
          '"vec4(float(hz_id), float(hz_block_ids[hz_base + 1u]), 0.0, 0.0)"')
ti = sub1(ti, '        let attributes = [', """        // Flat draws have the quad's corners: the block is the one its centre lies in, just
        // inside the face; a face that does not place its block points far outside the voxels.
        // The vertex path has none: zero, as before.
        let mid_block = if flat {
            "(hz_voxel ? vec4((floor((hz_p0 + hz_p2) * 0.5 - hz_Normal * 0.01) + 0.5 - hz_v_pos) * 64.0, 0.0) : vec4(1.0e8, 1.0e8, 1.0e8, 0.0))"
        } else {
            "vec4(0.0)"
        };
        let attributes = [""")
IRIS_IDS = r'''
/// GLSL after `FACE`: `hz_id`, the face's `mc_Entity`, and `hz_voxel`, whether it places its
/// block in voxels (`client_block_ids.rs`'s table: 16 redstone dust ids by power, then 8 words
/// per layer: Horizon's id and fluid flag, an id per HDR tier, the flags). Redstone dust is
/// tinted by its power: the brightest corner's red gives it back.
const IRIS_IDS: &str = "    uint hz_base = 16u + 8u * hz_layer;\n    \
    uint hz_tier = min(255u - uint(hz_v_color.a * 255.0 + 0.5), 4u);\n    \
    int hz_id = hz_block_ids[hz_base + 2u + hz_tier];\n    \
    int hz_flags = hz_block_ids[hz_base + 7u];\n    \
    bool hz_voxel = ((hz_flags >> int(hz_tier)) & 1) != 0;\n    \
    if ((hz_flags & 32) != 0) {\n        \
        uint hz_red = max(max(hz_vertices[hz_q + 5u] & 255u, hz_vertices[hz_q + 11u] & 255u), \
max(hz_vertices[hz_q + 17u] & 255u, hz_vertices[hz_q + 23u] & 255u));\n        \
        float hz_r = float(hz_red) / 255.0;\n        \
        int hz_power = hz_r < 0.37 ? 0 : clamp(int(round((hz_r - 0.4) / 0.6 * 15.0)), 1, 15);\n        \
        hz_id = hz_block_ids[uint(hz_power)];\n        \
        hz_voxel = hz_voxel && abs(hz_Normal.y) > 0.5;\n    \
    }\n";
'''
module('render-pack-atlas', RE + 'gfx/pack_atlas.rs', 'iris_atlas',
       "// `PackAtlas::terrain_inputs` with Iris's `at_midBlock`, and `mc_Entity` by sprite and glow.\nuse super::*;\n"
       + IRIS_IDS + "\nimpl PackAtlas {\n" + ti + "}\n")
hook('terrain-inputs', 'horizon_client_render::gfx::pack_atlas::PackAtlas::terrain_inputs', """\
fn iris_terrain_inputs(atlas: &PackAtlas, flat: bool) -> ::core::ops::ControlFlow<horizon_shaders::GbuffersInputs> {
    ::core::ops::ControlFlow::Break(atlas.iris_terrain_inputs(flat))
}
""", [binding('atlas', '&PackAtlas', 'self'), binding('flat', 'bool', 'flat')])

# --- Rain and snow passes kept across clear frames ----------------------------------------------
# Horizon's `draw_weather` takes the rain and snow passes out, then returns before putting them
# back on a frame without weather: after the first clear frame the pack had no weather passes
# left, and Horizon drew its own rain into the pack's buffers instead (rain showing block
# textures, the scene behind it unlit). A frame without weather now leaves them in place.
hook('draw-weather', 'horizon_client_render::gfx::pack_post::PackPost::draw_weather', '''\
fn iris_draw_weather(pp: &PackPost) -> ::core::ops::ControlFlow<()> {
    if pp.weather_draw.is_none() {
        ::core::ops::ControlFlow::Break(())
    } else {
        ::core::ops::ControlFlow::Continue(())
    }
}
''', [binding('pp', '&PackPost', 'self')])

# `draw_clouds` has the same flaw: a frame without clouds dropped a pack's `gbuffers_clouds` pass.
hook('draw-clouds', 'horizon_client_render::gfx::pack_post::PackPost::draw_clouds', '''\
fn iris_draw_clouds(pp: &PackPost) -> ::core::ops::ControlFlow<()> {
    if pp.cloud_draw.is_none() {
        ::core::ops::ControlFlow::Break(())
    } else {
        ::core::ops::ControlFlow::Continue(())
    }
}
''', [binding('pp', '&PackPost', 'self')])

# --- The renderer: loading a pack with the mod's resources ---------------------------------------
ssp = deindent(take(pristine(RE + 'gfx.rs'), r'^    pub fn set_shader_pack\('))
ssp = sub1(ssp, 'pub fn set_shader_pack(\n    &mut self,', 'pub fn set_shader_pack(\n    r: &mut Renderer,')
ssp = re.sub(r'\bself\b(?=\s*\.)', 'r', ssp)
ssp = sub1(ssp, 'pack_post::PackPost::new(', 'pack_post::PackPost::iris_new(')
ssp = sub1(ssp, '    if pack.is_none() {\n', '    if pack.is_none() {\n        pack_post::iris::forget();\n')
module('render-bridge', RE + 'gfx.rs', 'iris_bridge',
       '// `Renderer::set_shader_pack`, building the pack with the mod\'s resources.\nuse super::*;\n\n' + ssp
       + '''
/// Floodfill diagnostics in the log (Options > Mods > Iris Features).
pub fn set_floodfill_probe(on: bool) {
    pack_post::iris::PROBE_FLOODFILL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the floodfill diagnostics are on.
pub fn floodfill_probe() -> bool {
    pack_post::iris::PROBE_FLOODFILL.load(std::sync::atomic::Ordering::Relaxed)
}
''', 'pub')
hook('drop-shader-packs', 'horizon_client_render::gfx::Renderer::drop_shader_packs', '''\
fn iris_forget() {
    pack_post::iris::forget();
}
''', [], cancellable=False)

# --- The game: the pack's options in `.properties`, the player for Iris's uniforms --------------
spi = take(ref(GA + 'game/shader_packs.rs'), r'^    pub\(super\) fn shader_pack_inputs\(')
spi = sub1(spi, 'pub(super) fn shader_pack_inputs(', 'pub(super) fn iris_shader_pack_inputs(')
spi = re.sub(r'\bsuper::', 'super::super::', spi)
spi = sub1(spi, '                match Pack::open(&path) {\n',
           '                // The values the player applied in the shader options (Iris\'s file).\n'
           '                super::super::native_hooks::iris_options_menu::iris_load_values(name);\n'
           '                match Pack::open(&path) {\n')
spi = sub1(spi, '                        let macros = Macros::standard();\n',
           '                        let mut macros = Macros::standard();\n'
           '                        horizon_shaders::iris::define_iris(&mut macros);\n')
spi = sub1(spi, '''                        // `.properties` files see the pack's options at their defaults, as in
                        // Iris (Iris Features); programs read them from their own sources.''',
           '''                        // `.properties` files see the pack's options at the player's values,
                        // as in Iris; programs read them from their own sources.''')
spi = sub1(spi, 'let properties = Properties::parse(&text, &options);',
           'let properties = horizon_shaders::iris::parse_properties(&text, &options);')
spi = sub1(spi, 'renderer.set_shader_pack(dimension, Some(post), &table)',
           'horizon_client::gfx::iris_bridge::set_shader_pack(renderer, dimension, Some(post), &table)')
spi = spi.replace('horizon_client::gfx::pack_uniforms::IrisPlayer {', 'horizon_client::gfx::pack_uniforms::iris_uniforms::IrisPlayer {')
spi = sub1(spi, '        PackInputs {\n            day_time:',
           '        horizon_client::gfx::pack_uniforms::iris_uniforms::set_player(iris);\n        PackInputs {\n            day_time:')
spi = sub1(spi, '            held_light,\n            iris,\n        }', '            held_light,\n        }')
spi = sub1(spi, '''                        let table = block_id_table(&self.pack, &ids);
                        log::info!(
                            "shader pack {name}: mc_Entity for {} of {} block sprites in {} ms",
                            table.chunks(2).filter(|p| p[0] != -1).count(),
                            table.len() / 2,
                            started_ids.elapsed().as_millis()
                        );
''', """                        let mut table = block_id_table(&self.pack, &ids);
                        // Iris gives a fluid's surface the id of its block (`minecraft:water` for
                        // water): Eclipse draws water as water only with it. Each fluid sprite
                        // takes its block's id, whatever else shares the sprite.
                        for (block, sprites) in [
                            ("minecraft:water", ["water_still", "water_flow", "water_overlay"]),
                            ("minecraft:lava", ["lava_still", "lava_flow", ""]),
                        ] {
                            let Some(id) = ids.id(block, |_| None) else { continue };
                            for sprite in sprites.iter().filter(|s| !s.is_empty()) {
                                let key = format!("minecraft:block/{sprite}");
                                let Some(&l) = self.pack.render.sprite_layers.get(&key) else {
                                    log::info!("shader pack {name}: no sprite {key}");
                                    continue;
                                };
                                let at = l as usize * 2;
                                if let Some(slot) = table.get_mut(at) {
                                    *slot = id;
                                }
                                if let Some(slot) = table.get_mut(at + 1) {
                                    *slot = 1;
                                }
                            }
                        }
                        log::info!(
                            "shader pack {name}: mc_Entity for {} of {} block sprites in {} ms",
                            table.chunks(2).filter(|p| p[0] != -1).count(),
                            table.len() / 2,
                            started_ids.elapsed().as_millis()
                        );
                        // Ids by sprite and glow, and which faces place their block in voxels.
                        let (table, stats) = iris_block_id_table(&self.pack, &ids, &table);
                        log::info!(
                            "shader pack {name}: {} sprite uses shared by blocks of other ids, {} of them \\
                             left out of voxels (their blocks' other faces place them), {} redstone \\
                             dust sprites lit by power ({} ms in all)",
                            stats.mixed,
                            stats.left_out.len(),
                            stats.wire_layers,
                            started_ids.elapsed().as_millis()
                        );
                        log::debug!("shader pack {name}: left out of voxels: {}", stats.left_out.join(", "));
""")
module('client-shader-packs', GA + 'game/shader_packs.rs', 'iris_shader_packs',
       '// `shader_pack_inputs` with the pack\'s options in its `.properties`, the macros Iris defines,\n'
       '// the mod\'s resources, and the player for Iris\'s uniforms.\nuse super::*;\n\nimpl Game<\'_> {\n' + spi + '}\n'
       + own('block_id_slots.rs') + own('client_block_ids.rs'))
hook('shader-pack-inputs', "horizon_client::game::shader_packs::Game<'_>::shader_pack_inputs", '''\
fn iris_shader_pack_inputs(
    game: &mut crate::game::Game<'_>,
    camera: glam::DVec3,
    partial: f32,
) -> ::core::ops::ControlFlow<PackInputs> {
    ::core::ops::ControlFlow::Break(game.iris_shader_pack_inputs(camera, partial))
}
''', [binding('game', "&mut crate::game::Game<'_>", '&mut *self'), binding('camera', 'glam::DVec3', 'camera'),
      binding('partial', 'f32', 'partial')])

# --- Shader options: Iris Features' page in Options > Mods ----------------------------------------
module('client-options-menu', GA + 'game/native_hooks.rs', 'iris_options_menu',
       own('client_options_menu.rs').replace('{VERSION}', VERSION), 'pub')
hook('native-mods', "horizon_client::game::native_hooks::Game<'_>::native_mods", '''\
fn iris_native_mods(game: &mut crate::game::Game<'_>, mods: &mut Vec<NativeMod>) {
    game.iris_mods(mods);
}
''', [binding('game', "&mut crate::game::Game<'_>", '&mut *self'), binding('mods', '&mut Vec<NativeMod>', '&mut *mods')],
     cancellable=False)
hook('native-settings', "horizon_client::game::native_hooks::Game<'_>::native_settings", '''\
fn iris_native_settings(game: &mut crate::game::Game<'_>, mod_id: &str, buttons: &mut Vec<String>) {
    if mod_id == iris_options_menu::MOD_ID {
        game.iris_settings(buttons);
    }
}
''', [binding('game', "&mut crate::game::Game<'_>", '&mut *self'), binding('mod_id', '&str', 'mod_id'),
      binding('buttons', '&mut Vec<String>', '&mut *buttons')], cancellable=False)
hook('native-settings-click', "horizon_client::game::native_hooks::Game<'_>::native_settings_click", '''\
fn iris_native_settings_click(game: &mut crate::game::Game<'_>, mod_id: &str, button: usize) {
    if mod_id == iris_options_menu::MOD_ID {
        game.iris_settings_click(button);
    }
}
''', [binding('game', "&mut crate::game::Game<'_>", '&mut *self'), binding('mod_id', '&str', 'mod_id'),
      binding('button', 'usize', 'button')], cancellable=False)
