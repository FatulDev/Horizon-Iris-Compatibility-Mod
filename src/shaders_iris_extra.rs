// What the other crates use of the mod's shader side.

pub use crate::post::iris_post::{ComputePass, IrisPost};
pub use crate::translate::iris_translate::{ComputeProgram, WorkGroups, translate_compute};

thread_local! {
    // The compute programs and resources of the dimension `load_post` loaded last, until the
    // renderer takes them with the programs (`Post` has no field for them).
    static PENDING: std::cell::RefCell<Option<IrisPost>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn set_pending(post: IrisPost) {
    PENDING.with(|p| *p.borrow_mut() = Some(post));
}

/// The compute programs and resources of the dimension `load_post` loaded last (empty when there
/// are none, or they were taken).
pub fn take_pending() -> IrisPost {
    PENDING.with(|p| p.borrow_mut().take()).unwrap_or_default()
}

/// `Properties::parse` with custom uniform expressions reading the macros' values, as Iris's
/// preprocessor gives them.
pub fn parse_properties(text: &str, macros: &Macros) -> Properties {
    crate::properties::iris_properties::parse(text, macros)
}

/// Forgets what the custom uniforms' `smooth` calls have seen.
pub fn forget_smoothing(custom: &mut crate::Custom) {
    crate::custom::iris_custom::forget_smoothing(custom);
}

/// Whether the device has image atomics (`TEXTURE_ATOMIC`): set by the renderer when it asks
/// for its features, read when programs are translated.
static IMAGE_ATOMICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_image_atomics(on: bool) {
    IMAGE_ATOMICS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn image_atomics() -> bool {
    IMAGE_ATOMICS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `layer.solid`, `layer.cutout`, `layer.cutout_mipped` and `layer.translucent` of the running
/// pack's `block.properties` (Iris 4.2): blocks it draws on another layer than the game. Each
/// block name (`minecraft:glass`) with its layer: 0 solid, 1 cutout, 2 translucent.
static LAYERS: std::sync::RwLock<Vec<(String, u8)>> = std::sync::RwLock::new(Vec::new());

/// Reads the layer overrides of `block.properties` (called as the pack's ids are read).
pub fn set_layers(properties: &crate::Properties) {
    let mut out = Vec::new();
    for e in &properties.entries {
        let layer = match e.key.as_str() {
            "layer.solid" => 0,
            "layer.cutout" | "layer.cutout_mipped" => 1,
            "layer.translucent" => 2,
            _ => continue,
        };
        for word in e.value.split_whitespace() {
            let name = word.split(':').filter(|p| !p.contains('=')).collect::<Vec<_>>();
            let name = match name[..] {
                [n] => format!("minecraft:{n}"),
                [ns, n, ..] => format!("{ns}:{n}"),
                _ => continue,
            };
            out.push((name, layer));
        }
    }
    if let Ok(mut l) = LAYERS.write() {
        *l = out;
    }
}

/// The layer the pack puts block `name` (`minecraft:glass`) on, when it moves it.
pub fn block_layer(name: &str) -> Option<u8> {
    LAYERS.read().ok()?.iter().rev().find(|(n, _)| n == name).map(|l| l.1)
}

/// The layer of a held block's sprite (`block/glass`, `block/glass_pane_top`): the layer of the
/// block of that name, or of the pane or block whose texture it is.
pub fn sprite_layer(sprite: &str) -> Option<u8> {
    let name = sprite.strip_prefix("minecraft:").unwrap_or(sprite);
    let base = name.strip_prefix("block/")?;
    let layers = LAYERS.read().ok()?;
    if layers.is_empty() {
        return None;
    }
    let candidates = [
        base.to_owned(),
        base.trim_end_matches("_top").to_owned(),
        format!("{}_pane", base.trim_end_matches("_top").trim_end_matches("_pane")),
    ];
    layers
        .iter()
        .rev()
        .find(|(n, _)| candidates.iter().any(|c| n.strip_prefix("minecraft:") == Some(c.as_str())))
        .map(|l| l.1)
}

/// Whether a pack draws the terrain (set while one runs): distant terrain then meshes each
/// block's face on its own instead of merging runs of them.
static PACK_TERRAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_pack_terrain(on: bool) {
    PACK_TERRAIN.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn pack_terrain() -> bool {
    PACK_TERRAIN.load(std::sync::atomic::Ordering::Relaxed)
}
