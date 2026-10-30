// The uniforms only Iris provides (its "exclusive" uniforms): the player's health, hunger, air,
// state and look, after the frame's and the pack's custom ones.
use super::*;

/// The local player as Iris's own uniforms describe it.
#[derive(Clone, Copy, Debug)]
pub struct IrisPlayer {
    pub health: f32,
    pub max_health: f32,
    pub food: f32,
    pub air: f32,
    pub max_air: f32,
    pub burning: bool,
    pub hurt: bool,
    pub invisible: bool,
    pub on_ground: bool,
    pub sneaking: bool,
    pub sprinting: bool,
    pub spectator: bool,
    pub first_person: bool,
    /// Where the player looks (a unit vector).
    pub look: [f32; 3],
}

impl Default for IrisPlayer {
    /// A healthy player in first person, as a world without one reads.
    fn default() -> Self {
        Self {
            health: 20.0,
            max_health: 20.0,
            food: 20.0,
            air: 300.0,
            max_air: 300.0,
            burning: false,
            hurt: false,
            invisible: false,
            on_ground: true,
            sneaking: false,
            sprinting: false,
            spectator: false,
            first_person: true,
            look: [0.0, 0.0, 1.0],
        }
    }
}

thread_local! {
    // This frame's player, set by the game before the frame (`PackInputs` has no field for it).
    static PLAYER: std::cell::Cell<Option<IrisPlayer>> = const { std::cell::Cell::new(None) };
}

/// Sets the player the next frames' uniforms describe.
pub fn set_player(player: IrisPlayer) {
    PLAYER.with(|p| p.set(Some(player)));
}

/// The player the uniforms describe.
pub fn player() -> IrisPlayer {
    PLAYER.with(|p| p.get()).unwrap_or_default()
}

/// Uniforms only Iris provides, after the frame's and the pack's custom ones: their names and
/// GLSL types.
pub const IRIS_NAMES: [(&str, &str); 20] = [
    ("currentPlayerHealth", "float"),
    ("maxPlayerHealth", "float"),
    ("currentPlayerHunger", "float"),
    ("maxPlayerHunger", "float"),
    ("currentPlayerAir", "float"),
    ("maxPlayerAir", "float"),
    ("currentPlayerArmor", "float"),
    ("firstPersonCamera", "bool"),
    ("isSpectator", "bool"),
    ("playerLookVector", "vec3"),
    ("relativeEyePosition", "vec3"),
    ("lightningBoltPosition", "vec4"),
    ("is_burning", "bool"),
    ("is_hurt", "bool"),
    ("is_invisible", "bool"),
    ("is_on_ground", "bool"),
    ("is_sneaking", "bool"),
    ("is_sprinting", "bool"),
    ("isRightHanded", "bool"),
    ("alphaTestRef", "float"),
];

/// The position of `name` in [`IRIS_NAMES`].
pub fn iris_index(name: &str) -> Option<usize> {
    IRIS_NAMES.iter().position(|n| n.0 == name)
}

/// The values of [`IRIS_NAMES`] for `p`, as Iris gives them: health, hunger and air from 0 to 1
/// with their maximums apart, booleans as 0 or 1.
pub fn iris_values(p: &IrisPlayer) -> Vec<Value> {
    let flag = |b: bool| Value::i(&[i32::from(b)]);
    let part = |v: f32, max: f32| if max > 0.0 { (v / max).clamp(0.0, 1.0) } else { 0.0 };
    vec![
        Value::f(&[part(p.health, p.max_health)]),
        Value::f(&[p.max_health]),
        Value::f(&[part(p.food, 20.0)]),
        Value::f(&[20.0]),
        Value::f(&[part(p.air, p.max_air)]),
        Value::f(&[p.max_air]),
        Value::f(&[0.0]),
        flag(p.first_person),
        flag(p.spectator),
        Value::f(&p.look),
        // The camera is at the eyes in first person.
        Value::f(&[0.0, 0.0, 0.0]),
        // No bolt: w is 0.
        Value::f(&[0.0, 0.0, 0.0, 0.0]),
        flag(p.burning),
        flag(p.hurt),
        flag(p.invisible),
        flag(p.on_ground),
        flag(p.sneaking),
        flag(p.sprinting),
        flag(true),
        // The alpha below which cutout textures (leaves, plants, glass panes) are discarded: Iris's
        // 0.1 for the programs that test alpha; opaque texels are 1.0 anyway.
        Value::f(&[0.1]),
    ]
}

/// `FrameUniforms::eval_custom`, with Iris's uniforms readable by the custom ones and put after
/// them.
pub(crate) fn eval_custom(u: &mut FrameUniforms, custom: &mut horizon_shaders::Custom) {
    let iris = iris_values(&player());
    u.values.truncate(NAMES.len());
    let values = &u.values;
    let builtin = |name: &str| {
        let (v, n) = match index(name) {
            Some(i) => (values.get(i)?, components(name)),
            None => {
                let k = iris_index(name)?;
                let n = match IRIS_NAMES[k].1 {
                    "vec3" => 3,
                    "vec4" => 4,
                    _ => 1,
                };
                (iris.get(k)?, n)
            }
        };
        Some(match *v {
            // In Iris a custom uniform's `m.i.j` is GLSL's `m[i][j]` (column i, row j: packs write
            // `gbufferModelViewInverse.0.0 * x + gbufferModelViewInverse.1.0 * y + ...` for a
            // matrix times a vector); Horizon reads row i, column j. A matrix goes in transposed,
            // so both agree; else the sun's world direction turned with the camera.
            Value::F(f) if n == 16 => {
                let t: [f64; 16] = std::array::from_fn(|k| f64::from(f[(k % 4) * 4 + k / 4]));
                horizon_shaders::Val::vector(&t)
            }
            Value::F(f) => horizon_shaders::Val::vector(&f.map(f64::from)[..n]),
            Value::I(i) => horizon_shaders::Val::vector(&i.map(f64::from)[..n.min(3)]),
        })
    };
    let dt = match index("frameTime").and_then(|i| values.get(i)) {
        Some(Value::F(f)) => f64::from(f[0]),
        _ => 0.0,
    };
    // Iris's `smooth(x, up, down)` takes half-lives in tenths of a second; Horizon's reads them as
    // seconds. `dt` reaches nothing else, so ten times the frame time is Iris's smoothing (a
    // water exit, the sun's path, biome fades: ten times too slow otherwise).
    let out = custom.eval(dt * 10.0, &builtin);
    u.values
        .extend(out.iter().map(|v| Value::f(&v.v.map(|x| x as f32)[..v.n])));
    u.values.extend(iris.iter().copied());
}
