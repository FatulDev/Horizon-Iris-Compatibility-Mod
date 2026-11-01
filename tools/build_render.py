# The renderer and the game: executed by build.py.
GA = 'crates/horizon-client/src/'

module('render-iris', RE + 'gfx.rs', 'iris', dev(RE + 'gfx/iris.rs').replace('//!', '//'), 'pub')

replace('uniform-index', RE + 'gfx/pack_uniforms.rs', r'^pub fn index\(', 'horizon_client_render::gfx::pack_uniforms::index')
replace('eval-custom', RE + 'gfx/pack_uniforms.rs', r'^    pub fn eval_custom\(', 'horizon_client_render::gfx::pack_uniforms::FrameUniforms::eval_custom', owner='FrameUniforms')
replace('bindings', RE + 'gfx/pack_post.rs', r'^fn bindings\(', 'horizon_client_render::gfx::pack_post::bindings')
replace('pack-makeable', RE + 'gfx/pack_images.rs', r'^pub\(super\) fn makeable\(', 'horizon_client_render::gfx::pack_images::makeable')
replace('resize-images', RE + 'gfx/pack_images.rs', r'^pub\(super\) fn resize_images\(', 'horizon_client_render::gfx::pack_images::resize_images')
replace('terrain-inputs', RE + 'gfx/pack_atlas.rs', r'^    pub fn terrain_inputs\(', 'horizon_client_render::gfx::pack_atlas::PackAtlas::terrain_inputs', owner='PackAtlas')
replace('lod-emit', RE + 'lod/mesh.rs', r'^/// While a shader pack.*\nfn emit\(|^fn emit\(', 'horizon_client_render::lod::mesh::emit')

# Which buffers the passes built so far write, and the pack's custom uniform count (Iris's own
# uniforms come after them).
hook('pack-new', 'horizon_client_render::gfx::pack_post::PackPost::new', '''\
fn iris_pack_new(device: &wgpu::Device, post: &horizon_shaders::Post) {
    crate::gfx::iris::set_custom_count(post.custom.uniforms().count());
    // A pack's GPU errors are logged, each once, instead of ending the game (wgpu's default).
    static HANDLER: std::sync::Once = std::sync::Once::new();
    HANDLER.call_once(|| {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(std::collections::BTreeSet::<String>::new()));
        device.on_uncaptured_error(std::sync::Arc::new(move |e| {
            let text = e.to_string();
            if seen.lock().map(|mut s| s.insert(text.clone())).unwrap_or(false) {
                log::error!("GPU error while a shader pack runs: {text}");
            }
        }));
    });
}
''', [binding('device', '&wgpu::Device', 'device'), binding('post', '&horizon_shaders::Post', '&post')], cancellable=False)
hook('pack-macros', 'horizon_client_render::gfx::Renderer::define_pack_macros', '''\
fn iris_pack_macros(macros: &mut horizon_shaders::Macros) {
''' + IRIS_MACROS + '''}
''', [binding('macros', '&mut horizon_shaders::Macros', '&mut *macros')], cancellable=False)
hook('pack-terrain-on', 'horizon_client_render::gfx::Renderer::set_shader_pack', '''\
fn iris_pack_terrain_on(pack: &Option<horizon_shaders::Post>) {
    horizon_shaders::iris::set_pack_terrain(pack.is_some());
}
''', [binding('pack', '&Option<horizon_shaders::Post>', '&pack')], cancellable=False)
hook('pack-terrain-off', 'horizon_client_render::gfx::Renderer::drop_shader_packs', '''\
fn iris_pack_terrain_off() {
    horizon_shaders::iris::set_pack_terrain(false);
}
''', [], cancellable=False)

hook('device-features', 'horizon_client_render::gfx::cull::Cull::features', '''\
fn iris_features(value: wgpu::Features, adapter: &wgpu::Adapter) -> wgpu::Features {
    // Compute programs' `barrier()` (naga lowers it with subgroup barriers); image atomics, for
    // voxels a vertex stage writes in no order of its own.
    horizon_shaders::iris::set_image_atomics(adapter.features().contains(wgpu::Features::TEXTURE_ATOMIC));
    value | (adapter.features() & (wgpu::Features::SUBGROUP | wgpu::Features::SUBGROUP_BARRIER | wgpu::Features::TEXTURE_ATOMIC))
}
''', [binding('value', 'wgpu::Features', '$value'), binding('adapter', '&wgpu::Adapter', 'adapter')], kind='RETURN')

# The player for Iris's own uniforms, each frame.
hook('iris-player', "horizon_client::game::shader_packs::Game<'_>::held_for_pack", '''\
fn iris_player(game: &mut crate::game::Game<'_>) {
    let invisible = game.potions.effects.get("invisibility").is_some();
    if let Some(p) = game.player.as_mut() {
        horizon_client::gfx::iris::set_player(horizon_client::gfx::iris::IrisPlayer {
            health: p.health.health(),
            max_health: p.attribute("minecraft:max_health").unwrap_or(20.0) as f32,
            food: p.health.food.food as f32,
            air: p.air as f32,
            max_air: 300.0,
            armor: 0.0,
            burning: p.fire_ticks > 0,
            hurt: p.health.hurt.cooldown > 0,
            invisible,
            on_ground: p.ground.on_ground,
            sneaking: p.shift_key_down,
            sprinting: p.sprinting,
            spectator: p.spectator,
        });
    }
}
''', [binding('game', "&mut crate::game::Game<'_>", '&mut *self')], cancellable=False)

# Shader options: a profile's values as the pack writes them.
replace('next-profile', RE + 'shader_options.rs', r'^    pub fn next_profile\(', 'horizon_client_render::shader_options::ShaderOptions::next_profile', owner='ShaderOptions')

# The block outline drawn by the pack's frame, after translucent terrain, with its camera.
module('render-iris-extra', RE + 'gfx/pack_post.rs', 'iris_extra', dev(RE + 'gfx/pack_post/iris_extra.rs'), 'pub(crate)')
hook('outline-frame', 'horizon_client_render::gfx::pack_post::PackPost::run', '''\
fn iris_outline_frame(frame: &super::Frame) {
    iris_extra::set_outline(frame);
}
''', [binding('frame', '&super::Frame', 'frame')], cancellable=False)
hook('outline-draw', 'horizon_client_render::gfx::pack_post::PackPost::draw_translucent', '''\
fn iris_outline_draw(value: (), pp: &mut PackPost, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
    iris_extra::draw_outline(pp, device, encoder);
    value
}
''', [binding('value', '()', '$value'), binding('pp', '&mut PackPost', '&mut *self'), binding('device', '&wgpu::Device', 'device'),
      binding('encoder', '&mut wgpu::CommandEncoder', '&mut *encoder')], kind='RETURN')
hook('horizon-outline', 'horizon_client_render::gfx::Renderer::draw_outline', '''\
fn iris_horizon_outline() -> ::core::ops::ControlFlow<()> {
    // A shader pack draws the outline itself (`pack_post::iris_extra`).
    if horizon_shaders::iris::pack_terrain() {
        ::core::ops::ControlFlow::Break(())
    } else {
        ::core::ops::ControlFlow::Continue(())
    }
}
''', [])
