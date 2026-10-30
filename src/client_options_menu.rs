// Iris Features' page in Options > Mods: the running shader pack's own options, as Iris's shader
// pack settings screen has them (its screens, profiles, and each option's values, with the pack's
// names for them), as a list of buttons. A click moves an option to its next value (Shift: the
// one before); changes wait for Apply, which writes Iris's `shaderpacks/<pack>.txt` and loads the
// pack again.
use super::*;
use horizon_shaders::iris_options::{self, Item, ShaderOptions};
use std::cell::RefCell;
use std::collections::BTreeMap;

pub(crate) const MOD_ID: &str = "iris-features";
const TITLE: &str = "Iris Features {VERSION}";

#[derive(Clone)]
enum Action {
    None,
    Back,
    Open(String),
    Cycle(String),
    Profile,
    Apply,
    Discard,
    Reset,
    Probe,
}

#[derive(Default)]
struct Menu {
    /// The pack the page shows (its file name), and its options.
    pack: String,
    model: Option<ShaderOptions>,
    /// Why there are none (asked again each time the page is shown).
    problem: Option<String>,
    /// The screens opened from the main one.
    path: Vec<String>,
    /// Values picked and not applied yet.
    pending: BTreeMap<String, String>,
    /// What each button of the page does.
    actions: Vec<Action>,
}

thread_local! {
    static MENU: RefCell<Menu> = RefCell::new(Menu::default());
}

/// `shaderpacks/<pack>.txt`: the values the player applied to `pack`, as Iris keeps them.
pub(crate) fn iris_values_path(pack: &str) -> std::path::PathBuf {
    Game::shader_packs_dir().join(format!("{pack}.txt"))
}

/// The values the player applied to `pack`, from its file, for the pack's next reading.
pub(crate) fn iris_load_values(pack: &str) {
    let text = fs::read_to_string(iris_values_path(pack)).unwrap_or_default();
    iris_options::set_values(pack, iris_options::parse_values(&text));
}

impl Game<'_> {
    pub(crate) fn iris_mods(&mut self, mods: &mut Vec<NativeMod>) {
        mods.push(NativeMod {
            id: MOD_ID.to_owned(),
            title: TITLE.to_owned(),
            description: "Runs shader packs made for Iris (compute programs, images, storage \
                          buffers, custom textures). Its page holds the running pack's options."
                .to_owned(),
            keys: Vec::new(),
        });
    }

    /// The pack Horizon runs (`HORIZON_SHADER_PACK` first, as `shader_pack_inputs` has it).
    fn iris_picked_pack(&self) -> String {
        match std::env::var("HORIZON_SHADER_PACK") {
            Ok(p) => p,
            Err(_) => self.settings.horizon.shader_pack.clone(),
        }
    }

    pub(crate) fn iris_settings(&mut self, buttons: &mut Vec<String>) {
        let picked = self.iris_picked_pack();
        let locale = self.settings.language.clone();
        let on = self.native_on(MOD_ID);
        MENU.with(|m| {
            let mut m = m.borrow_mut();
            if m.pack != picked {
                *m = Menu {
                    pack: picked.clone(),
                    ..Menu::default()
                };
            }
            if m.model.is_none() {
                load(&mut m, &locale);
            }
            let (labels, actions) = page(&m, on);
            m.actions = actions;
            buttons.extend(labels);
        });
    }

    pub(crate) fn iris_settings_click(&mut self, button: usize) {
        let back = self.physical_keys.contains(&KeyCode::ShiftLeft)
            || self.physical_keys.contains(&KeyCode::ShiftRight);
        let apply = MENU.with(|m| {
            let mut m = m.borrow_mut();
            let Some(action) = m.actions.get(button).cloned() else {
                return false;
            };
            click(&mut m, action, back)
        });
        if apply {
            self.iris_apply();
        }
    }

    /// Keeps the picked values in the pack's file and loads the pack again with them.
    fn iris_apply(&mut self) {
        let written = MENU.with(|m| {
            let mut guard = m.borrow_mut();
            let m = &mut *guard;
            let Some(model) = &m.model else { return None };
            let mut values = iris_options::values(&m.pack);
            values.extend(std::mem::take(&mut m.pending));
            // As Iris writes it: only what differs from the pack's defaults.
            values.retain(|k, v| model.options.get(k).is_none_or(|o| o.default != *v));
            let path = iris_values_path(&m.pack);
            let result = fs::write(&path, iris_options::values_text(&values));
            iris_options::set_values(&m.pack, values);
            // Its screens may follow the options: read again.
            m.model = None;
            Some((path, result))
        });
        let Some((path, result)) = written else { return };
        match result {
            Ok(()) => log::info!("shader options saved to {}", path.display()),
            Err(e) => log::warn!("shader options: could not write {}: {e}", path.display()),
        }
        self.pack_ids = None;
        if let Some(renderer) = &mut self.renderer {
            renderer.drop_shader_packs();
        }
    }
}

/// The options of `m.pack`, at the values applied to it.
fn load(m: &mut Menu, locale: &str) {
    m.problem = None;
    if m.pack.is_empty() {
        m.problem = Some("No shader pack is in use (Options > Horizon Settings > Shader Packs).".into());
        return;
    }
    let path = Game::shader_packs_dir().join(&m.pack);
    if horizon_shader_api::package::is_pack(&path) {
        m.problem = Some(format!("{} is one of Horizon's own packs: no Iris options.", m.pack));
        return;
    }
    let pack = match horizon_shaders::Pack::open(&path) {
        Ok(p) => p,
        Err(e) => {
            m.problem = Some(format!("{}: {e}", m.pack));
            return;
        }
    };
    iris_load_values(&m.pack);
    let mut macros = horizon_shaders::Macros::standard();
    horizon_shaders::iris::define_iris(&mut macros);
    let macros = horizon_shaders::iris::properties_macros(&pack, &macros);
    let model = ShaderOptions::load(&pack, locale, &macros);
    if model.options.is_empty() {
        m.problem = Some(format!("{} has no options.", m.pack));
        return;
    }
    // A screen the options no longer lead to is closed.
    let open = m.path.iter().take_while(|s| model.screens.contains_key(*s)).count();
    m.path.truncate(open);
    m.model = Some(model);
}

/// An option's value: picked, else applied, else its default.
fn value_of(m: &Menu, applied: &BTreeMap<String, String>, name: &str) -> String {
    let default = || {
        let model = m.model.as_ref();
        model.and_then(|o| o.options.get(name)).map_or_else(String::new, |o| o.default.clone())
    };
    m.pending
        .get(name)
        .or_else(|| applied.get(name))
        .cloned()
        .unwrap_or_else(default)
}

/// The page's labels and what each does.
fn page(m: &Menu, on: bool) -> (Vec<String>, Vec<Action>) {
    let mut rows: Vec<(String, Action)> = Vec::new();
    if !on {
        rows.push(("§7Turned off: the launcher's next build leaves Iris Features out.".into(), Action::None));
    }
    let Some(o) = &m.model else {
        let problem = m.problem.clone().unwrap_or_default();
        rows.push((format!("§7{problem}"), Action::None));
        return rows.into_iter().unzip();
    };
    let applied = iris_options::values(&m.pack);
    let screen = m.path.last().cloned().unwrap_or_default();
    let mut title = format!("§l{}§r", m.pack.trim_end_matches(".zip"));
    for s in &m.path {
        title += &format!(" §7>§r {}", o.screen_label(s));
    }
    rows.push((title, Action::None));
    if !m.path.is_empty() {
        rows.push(("< Back".into(), Action::Back));
    }
    let value = |n: &str| value_of(m, &applied, n);
    for item in o.screens.get(&screen).into_iter().flatten() {
        match item {
            Item::Option(n) => {
                let v = value(n);
                let mark = if m.pending.contains_key(n) { "§e* §r" } else { "" };
                let label = format!("{mark}{}§r: {}", o.option_label(n), o.value_label(n, &v));
                rows.push((label, Action::Cycle(n.clone())));
            }
            Item::Screen(s) => rows.push((format!("{}§r >", o.screen_label(s)), Action::Open(s.clone()))),
            Item::Profile if !o.profiles.is_empty() => {
                let current = o.current_profile(|k| value(k));
                let name = current.map_or("§7Custom".into(), |p| o.profile_label(&o.profiles[p].0));
                rows.push((format!("Profile: {name}"), Action::Profile));
            }
            Item::Profile | Item::Empty => {}
        }
    }
    if !m.pending.is_empty() {
        let n = m.pending.len();
        let s = if n == 1 { "" } else { "s" };
        rows.push((format!("§aApply {n} change{s}§r (loads the pack again)"), Action::Apply));
        rows.push(("Discard changes".into(), Action::Discard));
    }
    if m.path.is_empty() {
        rows.push(("Reset every option to the pack's default".into(), Action::Reset));
        let probe = if horizon_client::gfx::iris_bridge::floodfill_probe() { "§aOn" } else { "§cOff" };
        rows.push((format!("Floodfill diagnostics in the log: {probe}"), Action::Probe));
    }
    rows.push(("§7Click: next value. Shift+click: the one before.".into(), Action::None));
    rows.into_iter().unzip()
}

/// Does `action`; true when the changes are to be applied.
fn click(m: &mut Menu, action: Action, back: bool) -> bool {
    let applied = iris_options::values(&m.pack);
    match action {
        Action::None => {}
        Action::Back => _ = m.path.pop(),
        Action::Open(s) => m.path.push(s),
        Action::Cycle(name) => {
            let Some(o) = m.model.as_ref().and_then(|o| o.options.get(&name)) else {
                return false;
            };
            let current = value_of(m, &applied, &name);
            let n = o.values.len();
            if n == 0 {
                return false;
            }
            let next = match o.values.iter().position(|v| *v == current) {
                Some(i) if back => (i + n - 1) % n,
                Some(i) => (i + 1) % n,
                None => 0,
            };
            let value = o.values[next].clone();
            set(m, &applied, name, value);
        }
        Action::Profile => {
            let Some(o) = m.model.as_ref().filter(|o| !o.profiles.is_empty()) else {
                return false;
            };
            let n = o.profiles.len();
            let current = o.current_profile(|k| value_of(m, &applied, k));
            let next = match current {
                Some(i) if back => (i + n - 1) % n,
                Some(i) => (i + 1) % n,
                None if back => n - 1,
                None => 0,
            };
            let set_values = o.profiles[next].1.clone();
            for (k, v) in set_values {
                set(m, &applied, k, v);
            }
        }
        Action::Apply => return true,
        Action::Probe => {
            let on = horizon_client::gfx::iris_bridge::floodfill_probe();
            horizon_client::gfx::iris_bridge::set_floodfill_probe(!on);
        }
        Action::Discard => m.pending.clear(),
        Action::Reset => {
            let Some(o) = m.model.as_ref() else { return false };
            let defaults: Vec<(String, String)> =
                o.options.values().map(|o| (o.name.clone(), o.default.clone())).collect();
            for (k, v) in defaults {
                set(m, &applied, k, v);
            }
        }
    }
    false
}

/// Option `name` picked at `value` (no change pending when that is the applied value).
fn set(m: &mut Menu, applied: &BTreeMap<String, String>, name: String, value: String) {
    let default = m.model.as_ref().and_then(|o| o.options.get(&name)).map(|o| o.default.clone());
    let now = applied.get(&name).cloned().or(default).unwrap_or_default();
    if value == now {
        m.pending.remove(&name);
    } else {
        m.pending.insert(name, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_shaders::iris_options::{Kind, ShaderOption};

    fn menu() -> Menu {
        let option = |name: &str, kind, default: &str, values: &[&str]| {
            let values = values.iter().map(|v| v.to_string()).collect();
            let o = ShaderOption { name: name.into(), kind, default: default.into(), values };
            (name.to_owned(), o)
        };
        let pairs = |p: &[(&str, &str)]| p.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        let model = ShaderOptions {
            options: [option("A", Kind::Switch, "true", &["true", "false"]), option("B", Kind::Value, "1", &["1", "2", "3"])]
                .into_iter()
                .collect(),
            screens: [
                (String::new(), vec![Item::Profile, Item::Option("A".into()), Item::Empty, Item::Screen("S".into())]),
                ("S".to_owned(), vec![Item::Option("B".into())]),
            ]
            .into_iter()
            .collect(),
            profiles: vec![
                ("LOW".into(), pairs(&[("A", "false"), ("B", "1")])),
                ("HIGH".into(), pairs(&[("A", "true"), ("B", "3")])),
            ],
            lang: Default::default(),
        };
        Menu { pack: "menu-test.zip".into(), model: Some(model), ..Menu::default() }
    }

    fn labels(m: &mut Menu) -> Vec<String> {
        let (labels, actions) = page(m, true);
        m.actions = actions;
        labels
    }

    fn press(m: &mut Menu, label: &str, back: bool) -> bool {
        let i = labels(m).iter().position(|l| l.contains(label)).unwrap_or_else(|| panic!("no {label}"));
        let action = m.actions[i].clone();
        click(m, action, back)
    }

    #[test]
    fn cycles_options_and_profiles_until_applied() {
        let mut m = menu();
        let page0 = labels(&mut m);
        assert!(page0.contains(&"Profile: §7Custom".to_owned()), "{page0:?}");
        assert!(page0.iter().any(|l| l == "A§r: §aOn"), "{page0:?}");
        assert!(!page0.iter().any(|l| l.contains("Apply")));
        press(&mut m, "A§r", false);
        assert_eq!(m.pending.get("A").map(String::as_str), Some("false"));
        assert!(labels(&mut m).iter().any(|l| l.contains("Apply 1 change")));
        // Back to its applied value: nothing pending.
        press(&mut m, "A§r", true);
        assert!(m.pending.is_empty());
        press(&mut m, "Profile", false);
        assert!(labels(&mut m).contains(&"Profile: LOW".to_owned()));
        assert_eq!(m.pending.len(), 1, "B=1 is its value already");
        press(&mut m, "S§r >", false);
        assert_eq!(m.path, ["S"]);
        press(&mut m, "B§r", true);
        assert_eq!(m.pending.get("B").map(String::as_str), Some("3"));
        press(&mut m, "< Back", false);
        assert!(m.path.is_empty());
        assert!(labels(&mut m).contains(&"Profile: §7Custom".to_owned()));
        press(&mut m, "Discard", false);
        assert!(m.pending.is_empty());
        assert!(press(&mut m, "Profile", true) == false && m.pending.contains_key("B"));
        assert!(press(&mut m, "Apply", false));
    }
}
