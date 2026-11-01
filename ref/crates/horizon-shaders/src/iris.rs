//! Iris Features: state the mod's copies of Horizon's functions share.
use std::sync::atomic::{AtomicBool, Ordering};

static IMAGE_ATOMICS: AtomicBool = AtomicBool::new(false);
/// Whether the device has image atomics (`TEXTURE_ATOMIC`).
pub fn set_image_atomics(on: bool) {
    IMAGE_ATOMICS.store(on, Ordering::Relaxed);
}
pub fn image_atomics() -> bool {
    IMAGE_ATOMICS.load(Ordering::Relaxed)
}

static PACK_TERRAIN: AtomicBool = AtomicBool::new(false);
/// While a shader pack's terrain programs draw (distant terrain is meshed a block a face).
pub fn set_pack_terrain(on: bool) {
    PACK_TERRAIN.store(on, Ordering::Relaxed);
}
pub fn pack_terrain() -> bool {
    PACK_TERRAIN.load(Ordering::Relaxed)
}

static ONE_D: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
/// `image.<name>` declared with one size: a 1D image (kept as `width x 1`).
pub fn set_one_d(name: &str, on: bool) {
    let mut l = ONE_D.lock().unwrap_or_else(|e| e.into_inner());
    l.retain(|n| n != name);
    if on {
        l.push(name.to_owned());
    }
}
pub fn is_one_d(name: &str) -> bool {
    ONE_D.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|n| n == name)
}

static WRITTEN_BEFORE: std::sync::Mutex<Vec<(String, u16)>> = std::sync::Mutex::new(Vec::new());
/// For each composite-style program, the buffers the programs before it write: as in Iris,
/// `texture.<stage>.colortexN` stands in for buffer N only until a program has written it.
pub fn set_written_before(table: Vec<(String, u16)>) {
    *WRITTEN_BEFORE.lock().unwrap_or_else(|e| e.into_inner()) = table;
}
pub fn written_before(program: &str) -> u16 {
    WRITTEN_BEFORE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(n, _)| n == program)
        .map_or(0, |e| e.1)
}
