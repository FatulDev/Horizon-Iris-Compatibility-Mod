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

/// Programs Horizon has no place for, loaded with the pack (`gbuffers_hand_water`, the weather
/// program again for drawing without depth), for the renderer to take.
static EXTRA: std::sync::Mutex<Vec<(String, crate::PostProgram)>> = std::sync::Mutex::new(Vec::new());
pub fn set_extra(programs: Vec<(String, crate::PostProgram)>) {
    *LOADED.lock().unwrap_or_else(|e| e.into_inner()) =
        programs.iter().map(|(n, _)| n.clone()).collect();
    *EXTRA.lock().unwrap_or_else(|e| e.into_inner()) = programs;
}

static LOADED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
/// Whether the pack being run has the extra program `name` (taken or not).
pub fn extra_loaded(name: &str) -> bool {
    LOADED.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|n| n == name)
}
/// Takes the program loaded as `name`.
pub fn take_extra(name: &str) -> Option<crate::PostProgram> {
    let mut l = EXTRA.lock().unwrap_or_else(|e| e.into_inner());
    let at = l.iter().position(|(n, _)| n == name)?;
    Some(l.remove(at).1)
}

static RAIN_DEPTH: AtomicBool = AtomicBool::new(false);
/// `rain.depth`: whether rain and snow write depth (Iris: only when `true`).
pub fn set_rain_depth(on: bool) {
    RAIN_DEPTH.store(on, Ordering::Relaxed);
}
pub fn rain_depth() -> bool {
    RAIN_DEPTH.load(Ordering::Relaxed)
}

static BUFFER_BLENDS: std::sync::Mutex<Vec<(String, u8, Option<crate::Blend>)>> =
    std::sync::Mutex::new(Vec::new());
/// The pack's `blend.<program>.<buffer>` keys: (program, buffer, its blend or `None` for off).
pub fn set_buffer_blends(table: Vec<(String, u8, Option<crate::Blend>)>) {
    *BUFFER_BLENDS.lock().unwrap_or_else(|e| e.into_inner()) = table;
}
/// Buffer `buffer`'s own blend in program `program`: `None` when the pack sets none, else the
/// blend (`Some(None)`: off).
pub fn buffer_blend(program: &str, buffer: u8) -> Option<Option<crate::Blend>> {
    BUFFER_BLENDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(p, b, _)| p == program && *b == buffer)
        .map(|e| e.2)
}
