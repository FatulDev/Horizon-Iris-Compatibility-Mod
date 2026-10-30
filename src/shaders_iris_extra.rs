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
