//! Iris Features: render state the mod's copies of Horizon's functions share: Iris's own
//! uniforms (the player), and which buffers the passes built so far write.
use super::pack_uniforms::Value;

/// Entity programs of the mod's own after `pack_entities`' (`pack_post::iris_extra` draws them):
/// block-breaking cracks (`gbuffers_damagedblock`) and translucent particles
/// (`gbuffers_particles_translucent`).
pub const DAMAGED: usize = super::pack_entities::PROGRAMS;
pub const PARTICLE_TRANSLUCENT: usize = super::pack_entities::PROGRAMS + 1;

thread_local! {
    /// The blends `build_pass` gives its blended outputs, in order, where the pack sets a
    /// buffer's own (`blend.<program>.<buffer>`); `None` keeps the program's.
    static BLENDS: std::cell::RefCell<std::collections::VecDeque<Option<wgpu::BlendState>>> =
        const { std::cell::RefCell::new(std::collections::VecDeque::new()) };
}

/// At the start of `build_pass`: the per-buffer blends of `p`'s outputs that take its blend
/// (those `pack_post::blend_state` is asked for, in the same order).
pub fn queue_buffer_blends(
    p: &horizon_shaders::PostProgram,
    formats: &[wgpu::TextureFormat; horizon_shaders::BUFFERS],
    features: wgpu::Features,
) {
    let mut queue = std::collections::VecDeque::new();
    if p.blend.is_some()
        && p.group != horizon_shaders::Group::Final
        && let Some(program) = &p.program
    {
        for &location in &program.interface.outputs {
            let Some(&buffer) = p.buffers.get(location as usize) else {
                continue;
            };
            let blendable = formats[buffer as usize]
                .guaranteed_format_features(features)
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE);
            if blendable {
                queue.push_back(
                    horizon_shaders::iris::buffer_blend(&p.name, buffer as u8)
                        .map(|b| b.map_or(wgpu::BlendState::REPLACE, blend_state)),
                );
            }
        }
    }
    BLENDS.with(|b| *b.borrow_mut() = queue);
}

/// In `pack_post::blend_state`: the next output's own blend, when the pack sets one.
pub fn next_buffer_blend() -> Option<wgpu::BlendState> {
    BLENDS.with(|b| b.borrow_mut().pop_front().flatten())
}

/// `pack_post::blend_state`'s mapping (that function hands out the queued blends, so it isn't
/// called from here).
fn blend_state(b: horizon_shaders::Blend) -> wgpu::BlendState {
    use horizon_shaders::Factor as F;
    use wgpu::BlendFactor as W;
    let f = |f| match f {
        F::Zero => W::Zero,
        F::One => W::One,
        F::SrcColor => W::Src,
        F::OneMinusSrcColor => W::OneMinusSrc,
        F::DstColor => W::Dst,
        F::OneMinusDstColor => W::OneMinusDst,
        F::SrcAlpha => W::SrcAlpha,
        F::OneMinusSrcAlpha => W::OneMinusSrcAlpha,
        F::DstAlpha => W::DstAlpha,
        F::OneMinusDstAlpha => W::OneMinusDstAlpha,
        F::SrcAlphaSaturate => W::SrcAlphaSaturated,
    };
    let component = |src, dst| wgpu::BlendComponent {
        src_factor: f(src),
        dst_factor: f(dst),
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState { color: component(b[0], b[1]), alpha: component(b[2], b[3]) }
}

static CRUMBLING: std::sync::Mutex<[f32; 4]> = std::sync::Mutex::new([0.0; 4]);
/// The frame's destroy-stage sprite rect (`Frame::crumbling_sprite`): its size, which every
/// stage's sprite shares.
pub fn set_crumbling(rect: [f32; 4]) {
    *CRUMBLING.lock().unwrap_or_else(|e| e.into_inner()) = rect;
}

/// Crack quads (`breaking::geometry`) as a pack's program reads them: Horizon's carry a decal
/// UV that repeats over the block and their stage sprite's origin in the light bytes
/// (`breaking::stage_light`), which entity.wgsl wraps per pixel. A pack samples the atlas at
/// the UV as given, so each quad's UVs become atlas UVs inside its stage's sprite here (they
/// read the whole atlas), and the light full: the pass is unlit.
pub fn crack_uvs(quads: &mut [crate::gfx::Vertex]) {
    let [_, _, w, h] = *CRUMBLING.lock().unwrap_or_else(|e| e.into_inner());
    for quad in quads.chunks_exact_mut(4) {
        let light = quad[0].light;
        let origin = [
            (light & 0xffff) as f32 / 65535.0,
            (light >> 16) as f32 / 65535.0,
        ];
        // The tile the quad's corners lie in: the lowest corner's, so a face's 0 and 1 edges
        // land on the sprite's two sides.
        let low = |k: usize| (quad.iter().map(|v| v.uv[k]).fold(f32::INFINITY, f32::min) + 1e-4).floor();
        let base = [low(0), low(1)];
        for v in quad.iter_mut() {
            v.uv = [
                origin[0] + (v.uv[0] - base[0]).clamp(0.0, 1.0) * w,
                origin[1] + (v.uv[1] - base[1]).clamp(0.0, 1.0) * h,
            ];
            v.light = 240 | 240 << 8;
        }
    }
}

/// The local player as Iris's own uniforms describe it.
#[derive(Clone, Copy, Debug)]
pub struct IrisPlayer {
    pub health: f32,
    pub max_health: f32,
    pub food: f32,
    pub air: f32,
    pub max_air: f32,
    pub armor: f32,
    pub burning: bool,
    pub hurt: bool,
    pub invisible: bool,
    pub on_ground: bool,
    pub sneaking: bool,
    pub sprinting: bool,
    pub spectator: bool,
}

impl IrisPlayer {
    /// A healthy player, as a world without one reads.
    pub const HEALTHY: Self = Self {
        health: 20.0,
        max_health: 20.0,
        food: 20.0,
        air: 300.0,
        max_air: 300.0,
        armor: 0.0,
        burning: false,
        hurt: false,
        invisible: false,
        on_ground: true,
        sneaking: false,
        sprinting: false,
        spectator: false,
    };
}

static PLAYER: std::sync::Mutex<IrisPlayer> = std::sync::Mutex::new(IrisPlayer::HEALTHY);

/// The player for the next frames' uniforms.
pub fn set_player(player: IrisPlayer) {
    *PLAYER.lock().unwrap_or_else(|e| e.into_inner()) = player;
}

/// Iris's own uniforms: after Horizon's and the pack's custom ones.
pub const NAMES: [&str; 16] = [
    "currentPlayerHealth",
    "maxPlayerHealth",
    "currentPlayerHunger",
    "maxPlayerHunger",
    "currentPlayerAir",
    "maxPlayerAir",
    "currentPlayerArmor",
    "isSpectator",
    "lightningBoltPosition",
    "is_burning",
    "is_hurt",
    "is_invisible",
    "is_on_ground",
    "is_sneaking",
    "is_sprinting",
    "cloudHeight",
];

/// The position of `name` among [`NAMES`].
pub fn index(name: &str) -> Option<usize> {
    NAMES.iter().position(|n| *n == name)
}

/// Components of `NAMES[k]`.
pub fn components(k: usize) -> usize {
    if NAMES[k] == "lightningBoltPosition" { 4 } else { 1 }
}

/// The values of [`NAMES`], as Iris gives them: health, hunger and air from 0 to 1 with their
/// maximums apart, booleans as 0 or 1, no lightning bolt (w 0).
pub fn values() -> Vec<Value> {
    let p = *PLAYER.lock().unwrap_or_else(|e| e.into_inner());
    let f = |v: &[f32]| {
        let mut a = [0.0; 16];
        a[..v.len()].copy_from_slice(v);
        Value::F(a)
    };
    let flag = |b: bool| Value::I([i32::from(b), 0, 0]);
    let part = |v: f32, max: f32| if max > 0.0 { (v / max).clamp(0.0, 1.0) } else { 0.0 };
    vec![
        f(&[part(p.health, p.max_health)]),
        f(&[p.max_health]),
        f(&[part(p.food, 20.0)]),
        f(&[20.0]),
        f(&[part(p.air, p.max_air)]),
        f(&[p.max_air]),
        f(&[p.armor]),
        flag(p.spectator),
        f(&[0.0, 0.0, 0.0, 0.0]),
        flag(p.burning),
        flag(p.hurt),
        flag(p.invisible),
        flag(p.on_ground),
        flag(p.sneaking),
        flag(p.sprinting),
        // The clouds' altitude, as Horizon draws them (`clouds.rs`): read as 0, Complementary
        // and its kin put their clouds at the floor of the world, seen edge-on as a grey band
        // at the horizon, and none in the sky.
        f(&[192.33]),
    ]
}

static CUSTOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// How many custom uniforms the pack being built has (Iris's own come after them).
pub fn set_custom_count(n: usize) {
    CUSTOM.store(n, std::sync::atomic::Ordering::Relaxed);
}
pub fn custom_count() -> usize {
    CUSTOM.load(std::sync::atomic::Ordering::Relaxed)
}


/// `mc_Entity`'s table for flat terrain draws, which know a face's sprite layer and how it glows
/// (the HDR code its colour's alpha carries, `mesher::emitting`), not its block. Horizon gives a
/// layer the id most of its blocks have; where blocks of other ids share a sprite (a furnace's
/// sides are the lit furnace's too) one block's faces then carried two ids, and a pack that
/// voxelises terrain (Eclipse's floodfill) kept whichever face the GPU drew last: lights
/// flickered, an unlit furnace or pumpkin shone.
///
/// The table, in `i32`s: 16 ids of `redstone_wire` by power, then [`ID_STRIDE`] per sprite
/// layer: Horizon's (id, fluid) pair, an id for each glow code (0 to [`GLOW_CODES`] - 1), and
/// flags: bit `c` set when faces of code `c` place their block in voxels (`at_midBlock`),
/// [`IRIS_WIRE`] when the layer is redstone dust tinted by its power.
pub const GLOW_CODES: usize = 9;
pub const ID_STRIDE: usize = 2 + GLOW_CODES + 1;
pub const IRIS_WIRE: i32 = 1 << 16;

/// The glow code of HDR tier `tier` (`mesher::LOW`... plus `WHOLE`), as `mesher::emitting`
/// writes it: 15 minus it is the colour alpha's low nibble.
pub fn glow_code(tier: u8) -> u8 {
    let level = (tier & !crate::mesher::WHOLE).min(crate::mesher::VERY_BRIGHT);
    if level > 0 && tier & crate::mesher::WHOLE != 0 { level + 4 } else { level }
}

/// A block state as [`id_table`] sees it.
pub struct IdState<'a> {
    /// `minecraft:furnace`.
    pub name: &'a str,
    /// Its id in `block.properties`, -1 without one.
    pub id: i32,
    /// `redstone_wire`'s power.
    pub wire_power: Option<usize>,
    /// Its faces: (sprite layer, glow code, tinted).
    pub faces: Vec<(u32, u8, bool)>,
}

/// The table for `states` over Horizon's (id, fluid) pairs by layer (`pairs`); `layer_name`
/// gives a layer's sprite (`block/furnace_side`). Also how many (layer, code) uses blocks of
/// different ids share, and how many of those are left out of voxels.
pub fn id_table<'n>(
    states: &[IdState<'_>],
    layer_name: impl Fn(u32) -> Option<&'n str>,
    pairs: &[i32],
) -> (Vec<i32>, usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    let layers = pairs.len() / 2;
    let mut table = vec![-1; 16 + ID_STRIDE * layers];
    for l in 0..layers {
        let at = 16 + ID_STRIDE * l;
        table[at] = pairs[2 * l];
        table[at + 1] = pairs[2 * l + 1];
        table[at + 2..at + 2 + GLOW_CODES].fill(pairs[2 * l]);
        table[at + 2 + GLOW_CODES] = (1 << GLOW_CODES) - 1;
    }
    let (mut mixed, mut left_out) = (0, 0);
    let fluid = |l: u32| pairs.get(2 * l as usize + 1) == Some(&1);
    // (layer, code) -> id -> (a block named like the sprite, the blocks).
    type Votes<'s> = BTreeMap<i32, (bool, BTreeSet<&'s str>)>;
    let mut slots: BTreeMap<(u32, u8), Votes<'_>> = BTreeMap::new();
    let mut uses: Vec<BTreeSet<(u32, u8)>> = Vec::with_capacity(states.len());
    let mut users: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    // Per layer: (every face on it is redstone dust's, every one tinted).
    let mut wire: BTreeMap<u32, (bool, bool)> = BTreeMap::new();
    for (k, s) in states.iter().enumerate() {
        let block = s.name.strip_prefix("minecraft:").unwrap_or(s.name);
        if let Some(power) = s.wire_power
            && power < 16
            && table[power] == -1
        {
            table[power] = s.id;
        }
        let mut used = BTreeSet::new();
        for &(layer, code, tinted) in &s.faces {
            if layer as usize >= layers || fluid(layer) {
                continue;
            }
            let code = code.min(GLOW_CODES as u8 - 1);
            let named = layer_name(layer).and_then(|n| n.strip_prefix("block/")) == Some(block);
            let v = slots.entry((layer, code)).or_default().entry(s.id).or_default();
            v.0 |= named;
            v.1.insert(s.name);
            if used.insert((layer, code)) {
                users.entry((layer, code)).or_default().push(k);
            }
            let w = wire.entry(layer).or_insert((true, true));
            w.0 &= s.wire_power.is_some();
            w.1 &= tinted;
        }
        uses.push(used);
    }
    let wire_layer = |l: u32| wire.get(&l).is_some_and(|&(w, t)| w && t);
    // A use places its blocks rightly when they all have its id, or when it is tinted dust.
    let placed =
        |slot: &(u32, u8)| wire_layer(slot.0) || slots.get(slot).is_some_and(|v| v.len() == 1);
    for (&(layer, code), votes) in &slots {
        let at = 16 + ID_STRIDE * layer as usize;
        if let Some(best) = votes
            .iter()
            .max_by_key(|(id, (named, blocks))| (*named, blocks.len(), -**id))
            .map(|(id, _)| *id)
        {
            table[at + 2 + code as usize] = best;
        }
        if votes.len() > 1 && !wire_layer(layer) {
            mixed += 1;
            // Left out when each of its blocks has another face that places it; else kept with
            // the commonest id (a stair of a block's sprite is better as that block than absent).
            let of = users.get(&(layer, code)).map_or(&[][..], Vec::as_slice);
            if of
                .iter()
                .all(|&k| uses[k].iter().any(|s| *s != (layer, code) && placed(s)))
            {
                table[at + 2 + GLOW_CODES] &= !(1 << code);
                left_out += 1;
            }
        }
    }
    for &layer in wire.keys().filter(|l| wire_layer(**l)) {
        table[16 + ID_STRIDE * layer as usize + 2 + GLOW_CODES] |= IRIS_WIRE;
    }
    (table, mixed, left_out)
}

/// GLSL after `pack_atlas`'s `FACE`: `hz_id`, the face's `mc_Entity`, and `hz_voxel`, whether it
/// places its block in voxels ([`id_table`]'s table). Redstone dust is tinted by its power: the
/// brightest corner's red gives it back.
pub const IDS_GLSL: &str = "    uint hz_base = 16u + 12u * hz_layer;\n    \
    uint hz_glow = min(15u - (uint(hz_v_color.a * 255.0 + 0.5) & 15u), 8u);\n    \
    int hz_id = hz_block_ids[hz_base + 2u + hz_glow];\n    \
    int hz_flags = hz_block_ids[hz_base + 11u];\n    \
    bool hz_voxel = ((hz_flags >> int(hz_glow)) & 1) != 0;\n    \
    if ((hz_flags & 65536) != 0) {\n        \
        uint hz_red = max(max(hz_vertices[hz_q + 5u] & 255u, hz_vertices[hz_q + 11u] & 255u), \
max(hz_vertices[hz_q + 17u] & 255u, hz_vertices[hz_q + 23u] & 255u));\n        \
        float hz_rf = float(hz_red) / 255.0;\n        \
        int hz_power = hz_rf < 0.37 ? 0 : clamp(int(round((hz_rf - 0.4) / 0.6 * 15.0)), 1, 15);\n        \
        hz_id = hz_block_ids[uint(hz_power)];\n        \
        hz_voxel = hz_voxel && abs(hz_Normal.y) > 0.5;\n    \
    }\n";

#[cfg(test)]
mod id_tests {
    use super::*;

    #[test]
    fn crack_uvs_land_inside_their_stage_sprite() {
        set_crumbling([0.0, 0.0, 0.25, 0.5]);
        let v = |u: f32, w: f32| crate::gfx::Vertex {
            uv: [u, w],
            light: crate::breaking::stage_light([0.5, 0.25]),
            ..bytemuck::Zeroable::zeroed()
        };
        // A full face's decal UVs, -1 to 0 (`breaking::decal_uv` negates).
        let mut quad = [v(-1.0, -1.0), v(-1.0, 0.0), v(0.0, 0.0), v(0.0, -1.0)];
        crack_uvs(&mut quad);
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(close(quad[0].uv[0], 0.5) && close(quad[0].uv[1], 0.25));
        assert!(close(quad[2].uv[0], 0.75) && close(quad[2].uv[1], 0.75));
        set_crumbling([0.0; 4]);
    }

    #[test]
    fn cracks_sample_their_stage_sprite() {
        set_crumbling([0.0, 0.0, 0.25, 0.5]);
        let v = |uv: [f32; 2]| crate::gfx::Vertex {
            uv,
            light: (0.5f32 * 65535.0) as u32 | (0.25f32 * 65535.0).round() as u32 >> 0 << 16,
            ..bytemuck::Zeroable::zeroed()
        };
        let mut q = [v([-1.0, 0.0]), v([0.0, 0.0]), v([0.0, -1.0]), v([-1.0, -1.0])];
        crack_uvs(&mut q);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(near(q[0].uv[0], 0.5) && near(q[0].uv[1], 0.75));
        assert!(near(q[2].uv[0], 0.75) && near(q[2].uv[1], 0.25));
        assert_eq!(q[0].light, 240 | 240 << 8);
    }

    #[test]
    fn shared_sprites_take_ids_by_glow_and_leave_voxels_when_placed_elsewhere() {
        // Layer 0 furnace_side (unlit 10, lit 11 glowing), layer 1 furnace_front_on (lit only),
        // layer 2 furnace_front (unlit only).
        let unlit = IdState { name: "minecraft:furnace", id: 10, wire_power: None, faces: vec![(0, 0, false), (2, 0, false)] };
        let lit = IdState { name: "minecraft:lit_furnace", id: 11, wire_power: None, faces: vec![(0, 3, false), (1, 3, false)] };
        let pairs = [10, -1, 11, -1, 10, -1];
        let (t, mixed, left) = id_table(&[unlit, lit], |_| None, &pairs);
        let at = |l: usize| 16 + ID_STRIDE * l;
        assert_eq!(t[at(0) + 2], 10);
        assert_eq!(t[at(0) + 2 + 3], 11);
        assert_eq!((mixed, left), (0, 0));
        // The same glow: one shared use, left out since each block's front places it.
        let a = IdState { name: "minecraft:a", id: 1, wire_power: None, faces: vec![(0, 0, false), (1, 0, false)] };
        let b = IdState { name: "minecraft:b", id: 2, wire_power: None, faces: vec![(0, 0, false), (2, 0, false)] };
        let (t, mixed, left) = id_table(&[a, b], |_| None, &pairs);
        assert_eq!((mixed, left), (1, 1));
        assert_eq!(t[at(0) + 2 + GLOW_CODES] & 1, 0);
        assert_eq!(t[at(1) + 2 + GLOW_CODES] & 1, 1);
    }
}
