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
fn iris_pack_new(post: &horizon_shaders::Post) {
    crate::gfx::iris::set_custom_count(post.custom.uniforms().count());
}
''', [binding('post', '&horizon_shaders::Post', '&post')], cancellable=False)
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
