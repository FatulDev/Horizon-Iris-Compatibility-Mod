//! Iris Features: render state the mod's copies of Horizon's functions share: Iris's own
//! uniforms (the player), and which buffers the passes built so far write.
use super::pack_uniforms::Value;

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
pub const NAMES: [&str; 15] = [
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

