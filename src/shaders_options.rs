// A shader pack's own options, as Iris reads them: what the pack declares in its sources, the
// screens, profiles and names its `shaders.properties` and `lang/` give them, and the player's
// values (`shaderpacks/<pack>.txt`, Iris's file), which reach the sources as they are read
// (`read_text`, in place of `Pack::read_text`).
use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;

use crate::{Lang, Macros, Pack};

/// Source files: options are declared in these, and in files of other names they include.
const SOURCES: [&str; 8] = ["vsh", "fsh", "gsh", "csh", "tcs", "tes", "glsl", "inc"];

/// The player's values, by pack name (`Eclipse.zip`).
static VALUES: RwLock<BTreeMap<String, BTreeMap<String, String>>> = RwLock::new(BTreeMap::new());

/// Pack `pack` is read with `values` from now on (empty: its defaults).
pub fn set_values(pack: &str, values: BTreeMap<String, String>) {
    let mut all = VALUES.write().unwrap_or_else(|e| e.into_inner());
    all.insert(pack.to_owned(), values);
}

/// The values pack `pack` is read with.
pub fn values(pack: &str) -> BTreeMap<String, String> {
    let all = VALUES.read().unwrap_or_else(|e| e.into_inner());
    all.get(pack).cloned().unwrap_or_default()
}

/// `NAME=value` lines (Iris's `shaderpacks/<pack>.txt`); `#` starts a comment.
pub fn parse_values(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .filter(|(k, v)| !k.is_empty() && plain(v))
        .collect()
}

/// The text of `shaderpacks/<pack>.txt` for `values`.
pub fn values_text(values: &BTreeMap<String, String>) -> String {
    values.iter().map(|(k, v)| format!("{k}={v}\n")).collect()
}

/// A value that can stand for a token of GLSL: a number, `true`/`false` or a macro's name.
fn plain(v: &str) -> bool {
    !v.is_empty() && v.chars().all(|c| c.is_ascii_alphanumeric() || "_.+-".contains(c))
}

/// `Pack::read_text` with the player's values written into the option declarations.
pub fn read_text(pack: &Pack, path: &str) -> Option<String> {
    let bytes = pack.read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let data = [".properties", ".lang", ".json", ".txt", ".mcmeta"];
    if data.iter().any(|e| path.ends_with(e)) {
        return Some(text.to_owned());
    }
    let values = {
        let all = VALUES.read().unwrap_or_else(|e| e.into_inner());
        match all.get(pack.name()) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => return Some(text.to_owned()),
        }
    };
    let mut out = String::with_capacity(text.len() + 64);
    for line in text.split_inclusive('\n') {
        let body = line.trim_end_matches(['\n', '\r']);
        match rewrite(body, &values) {
            Some(new) => {
                out.push_str(&new);
                out.push_str(&line[body.len()..]);
            }
            None => out.push_str(line),
        }
    }
    Some(out)
}

/// `line` declaring an option `values` gives a value, with that value; `None` keeps the line.
fn rewrite(line: &str, values: &BTreeMap<String, String>) -> Option<String> {
    let decl = declaration(line)?;
    let value = values.get(&decl.name)?;
    if *value == decl.default || !plain(value) {
        return None;
    }
    match decl.kind {
        Kind::Switch => {
            let on = match value.as_str() {
                "true" => true,
                "false" => false,
                _ => return None,
            };
            let indent = &line[..line.len() - line.trim_start().len()];
            let comment = decl.comment.map(|c| format!(" //{c}")).unwrap_or_default();
            let slashes = if on { "" } else { "//" };
            Some(format!("{indent}{slashes}#define {}{comment}", decl.name))
        }
        Kind::Value | Kind::Const => {
            if decl.boolean && value != "true" && value != "false" {
                return None;
            }
            // The declaration's value token, and nothing else, changes.
            let start = decl.value_at;
            let end = start + decl.default.len();
            Some(format!("{}{}{}", &line[..start], value, &line[end..]))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `#define NAME` on, `//#define NAME` off.
    Switch,
    /// `#define NAME value // [a b c]`.
    Value,
    /// `const int NAME = value; // [a b c]`.
    Const,
}

struct Declaration {
    name: String,
    kind: Kind,
    default: String,
    /// Where the value starts in the line (`Value` and `Const`).
    value_at: usize,
    values: Vec<String>,
    /// A `const bool`.
    boolean: bool,
    /// The comment after `//`, without the slashes.
    comment: Option<String>,
}

/// The values listed in `comment` (`[a b c]`, anywhere in it).
fn listed(comment: Option<&str>) -> Vec<String> {
    let Some(c) = comment else { return Vec::new() };
    let Some(a) = c.find('[') else { return Vec::new() };
    let Some(b) = c[a..].find(']') else { return Vec::new() };
    c[a + 1..a + b].split_whitespace().map(str::to_owned).collect()
}

fn ident_end(s: &str) -> usize {
    s.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(s.len())
}

/// The option a source line declares, if it declares one.
fn declaration(line: &str) -> Option<Declaration> {
    let trimmed = line.trim_start();
    let (commented, rest) = match trimmed.strip_prefix("//") {
        Some(r) => (true, r.trim_start()),
        None => (false, trimmed),
    };
    if let Some(after) = rest.strip_prefix('#') {
        let words = after.trim_start().strip_prefix("define")?;
        if !words.starts_with(char::is_whitespace) {
            return None;
        }
        let words = words.trim_start();
        let name = &words[..ident_end(words)];
        if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
            return None;
        }
        let tail = words[name.len()..].trim_start();
        let (code, comment) = match tail.find("//") {
            Some(i) => (tail[..i].trim_end(), Some(&tail[i + 2..])),
            None => (tail.trim_end(), None),
        };
        if code.is_empty() {
            return Some(Declaration {
                name: name.to_owned(),
                kind: Kind::Switch,
                default: (!commented).to_string(),
                value_at: 0,
                values: vec!["true".into(), "false".into()],
                boolean: true,
                comment: comment.map(str::to_owned),
            });
        }
        // `#define NAME value // [list]`: one token, then a comment with the list.
        if commented || code.contains(char::is_whitespace) || words[name.len()..].starts_with('(') {
            return None;
        }
        let values = listed(comment);
        if values.is_empty() {
            return None;
        }
        return Some(Declaration {
            name: name.to_owned(),
            kind: Kind::Value,
            default: code.to_owned(),
            value_at: line.len() - tail.len(),
            values,
            boolean: false,
            comment: comment.map(str::to_owned),
        });
    }
    if commented {
        return None;
    }
    // `const <type> NAME = value; // [list]`.
    let after = rest.strip_prefix("const")?;
    if !after.starts_with(char::is_whitespace) {
        return None;
    }
    let after = after.trim_start();
    let ty = &after[..ident_end(after)];
    if !matches!(ty, "int" | "float" | "bool") {
        return None;
    }
    let after = after[ty.len()..].trim_start();
    let name = &after[..ident_end(after)];
    if name.is_empty() {
        return None;
    }
    let value_part = after[name.len()..].trim_start().strip_prefix('=')?.trim_start();
    let semi = value_part.find(';')?;
    let value = value_part[..semi].trim_end();
    if value.is_empty() || value.contains(char::is_whitespace) {
        return None;
    }
    let tail = &value_part[semi + 1..];
    let comment = tail.find("//").map(|i| &tail[i + 2..]);
    let boolean = ty == "bool";
    let values = if boolean {
        vec!["true".into(), "false".into()]
    } else {
        listed(comment)
    };
    if values.is_empty() {
        return None;
    }
    Some(Declaration {
        name: name.to_owned(),
        kind: Kind::Const,
        default: value.to_owned(),
        value_at: line.len() - value_part.len(),
        values,
        boolean,
        comment: comment.map(str::to_owned),
    })
}

/// One option of a pack.
#[derive(Clone, Debug)]
pub struct ShaderOption {
    pub name: String,
    pub kind: Kind,
    pub default: String,
    /// The values it cycles through (`true`, `false` for a switch).
    pub values: Vec<String>,
}

impl ShaderOption {
    pub fn is_boolean(&self) -> bool {
        self.values == ["true", "false"]
    }
}

/// An entry of an options screen.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Option(String),
    Screen(String),
    Profile,
    Empty,
}

/// A pack's options, screens, profiles and their names.
#[derive(Debug, Default)]
pub struct ShaderOptions {
    pub options: BTreeMap<String, ShaderOption>,
    /// Screens by name; the main one is "".
    pub screens: BTreeMap<String, Vec<Item>>,
    /// Profiles in the order the pack gives them, each with the values it sets.
    pub profiles: Vec<(String, Vec<(String, String)>)>,
    pub lang: Lang,
}

impl ShaderOptions {
    /// `pack`'s options, with their defaults as the pack ships them; `shaders.properties` is read
    /// with `macros` (the options at the values the pack is read with).
    pub fn load(pack: &Pack, locale: &str, macros: &Macros) -> Self {
        let mut options: BTreeMap<String, ShaderOption> = BTreeMap::new();
        let mut ambiguous = BTreeSet::new();
        let mut tested = BTreeSet::new();
        for text in sources(pack, false) {
            for line in text.lines() {
                note_tests(line, &mut tested);
                let Some(d) = declaration(line) else { continue };
                match options.get(&d.name) {
                    // Declared again with another default or kind: Iris leaves it out.
                    Some(o) if o.kind != d.kind || o.default != d.default => {
                        ambiguous.insert(d.name);
                    }
                    Some(_) => {}
                    None => {
                        let o = ShaderOption {
                            name: d.name.clone(),
                            kind: d.kind,
                            default: d.default,
                            values: d.values,
                        };
                        options.insert(d.name, o);
                    }
                }
            }
        }
        // A define without a value is an option where something tests it, as in Iris.
        options.retain(|n, o| !ambiguous.contains(n) && (o.kind != Kind::Switch || tested.contains(n)));
        let text = pack.read_text("shaders.properties").unwrap_or_default();
        let properties = crate::iris::parse_properties(&text, macros);
        let mut screens: BTreeMap<String, Vec<Item>> = BTreeMap::new();
        let mut raw_profiles: Vec<(String, String)> = Vec::new();
        for e in &properties.entries {
            let key = e.key.trim();
            if let Some(name) = key.strip_prefix("profile.") {
                raw_profiles.retain(|(n, _)| n != name);
                raw_profiles.push((name.to_owned(), e.value.clone()));
                continue;
            }
            let name = match key {
                "screen" => "",
                k => match k.strip_prefix("screen.") {
                    Some(n) if !n.ends_with(".columns") && n != "columns" => n,
                    _ => continue,
                },
            };
            let items = e
                .value
                .split_whitespace()
                .filter_map(|t| match t {
                    "<empty>" => Some(Item::Empty),
                    "<profile>" => Some(Item::Profile),
                    "*" => Some(Item::Option("*".into())),
                    t if t.len() > 2 && t.starts_with('[') && t.ends_with(']') => {
                        Some(Item::Screen(t[1..t.len() - 1].to_owned()))
                    }
                    t if options.contains_key(t) => Some(Item::Option(t.to_owned())),
                    _ => None,
                })
                .collect();
            screens.insert(name.to_owned(), items);
        }
        let placed: BTreeSet<String> = screens
            .values()
            .flatten()
            .filter_map(|i| match i {
                Item::Option(o) if o != "*" => Some(o.clone()),
                _ => None,
            })
            .collect();
        // `*`: the options no screen names; no screens at all: every option on one.
        let rest: Vec<Item> = options
            .keys()
            .filter(|n| !placed.contains(*n))
            .map(|n| Item::Option(n.clone()))
            .collect();
        if screens.is_empty() {
            screens.insert(String::new(), rest);
        } else {
            for items in screens.values_mut() {
                if let Some(at) = items.iter().position(|i| *i == Item::Option("*".into())) {
                    items.splice(at..=at, rest.iter().cloned());
                }
            }
        }
        // A screen named but not given is left out.
        let names: BTreeSet<String> = screens.keys().cloned().collect();
        for items in screens.values_mut() {
            items.retain(|i| !matches!(i, Item::Screen(s) if !names.contains(s)));
        }
        let raw: BTreeMap<&str, &str> = raw_profiles.iter().map(|(n, v)| (n.as_str(), v.as_str())).collect();
        let profiles = raw_profiles
            .iter()
            .map(|(name, _)| {
                let mut set = Vec::new();
                expand(name, &raw, &options, &mut set, 0);
                (name.clone(), set)
            })
            .filter(|(_, set)| !set.is_empty())
            .collect();
        Self {
            options,
            screens,
            profiles,
            lang: Lang::load(pack, locale),
        }
    }

    /// The name the pack gives option `name`.
    pub fn option_label(&self, name: &str) -> String {
        self.lang
            .get(&format!("option.{name}"))
            .map_or_else(|| name.to_owned(), str::to_owned)
    }

    /// The name the pack gives `value` of option `name`.
    pub fn value_label(&self, name: &str, value: &str) -> String {
        if let Some(l) = self.lang.get(&format!("value.{name}.{value}")) {
            return l.to_owned();
        }
        if self.options.get(name).is_some_and(ShaderOption::is_boolean) {
            return if value == "true" { "§aOn".into() } else { "§cOff".into() };
        }
        let prefix = self.lang.get(&format!("prefix.{name}")).unwrap_or("");
        let suffix = self.lang.get(&format!("suffix.{name}")).unwrap_or("");
        format!("{prefix}{value}{suffix}")
    }

    /// The name the pack gives screen `name` (the main screen: "").
    pub fn screen_label(&self, name: &str) -> String {
        if name.is_empty() {
            return "Shader Options".into();
        }
        self.lang
            .get(&format!("screen.{name}"))
            .map_or_else(|| name.replace('_', " "), str::to_owned)
    }

    /// The name the pack gives profile `name`.
    pub fn profile_label(&self, name: &str) -> String {
        self.lang
            .get(&format!("profile.{name}"))
            .map_or_else(|| name.replace('_', " "), str::to_owned)
    }

    /// The first profile whose values all match `value`'s.
    pub fn current_profile(&self, value: impl Fn(&str) -> String) -> Option<usize> {
        self.profiles
            .iter()
            .position(|(_, set)| set.iter().all(|(k, v)| value(k) == *v))
    }
}

/// Profile `name`'s values into `set`, following the profiles it names.
fn expand(
    name: &str,
    raw: &BTreeMap<&str, &str>,
    options: &BTreeMap<String, ShaderOption>,
    set: &mut Vec<(String, String)>,
    depth: u32,
) {
    let Some(text) = raw.get(name) else { return };
    if depth > 8 {
        return;
    }
    for token in text.split_whitespace() {
        if let Some(other) = token.strip_prefix("profile.") {
            expand(other, raw, options, set, depth + 1);
            continue;
        }
        let (key, value) = if let Some(k) = token.strip_prefix('!') {
            (k, "false")
        } else if let Some((k, v)) = token.split_once([':', '=']) {
            (k, v)
        } else {
            (token, "true")
        };
        if !options.contains_key(key) || !plain(value) {
            continue;
        }
        set.retain(|(k, _)| k != key);
        set.push((key.to_owned(), value.to_owned()));
    }
}

/// The names `#ifdef`, `#ifndef` and `defined` test in `line`.
fn note_tests(line: &str, tested: &mut BTreeSet<String>) {
    let Some(body) = line.trim_start().strip_prefix('#') else {
        return;
    };
    let body = body.trim_start();
    let ident = |s: &str| -> String {
        let s = s.trim_start().trim_start_matches('(').trim_start();
        s[..ident_end(s)].to_owned()
    };
    for key in ["ifdef", "ifndef"] {
        if let Some(rest) = body.strip_prefix(key) {
            tested.insert(ident(rest));
        }
    }
    let mut rest = body;
    while let Some(i) = rest.find("defined") {
        rest = &rest[i + "defined".len()..];
        let name = ident(rest);
        if !name.is_empty() {
            tested.insert(name);
        }
    }
}

/// The text of every source file of `pack` and of the files they include: as the pack ships
/// them, or with the player's values (`current`).
fn sources(pack: &Pack, current: bool) -> Vec<String> {
    let files = pack.all_files();
    let source = |path: &str| SOURCES.contains(&path.rsplit_once('.').map_or("", |(_, e)| e));
    let text = |p: &str| {
        if current {
            return read_text(pack, p);
        }
        let b = pack.read(p)?;
        let t = String::from_utf8_lossy(&b);
        Some(t.strip_prefix('\u{feff}').unwrap_or(&t).to_owned())
    };
    let mut texts: Vec<String> = files.iter().filter(|p| source(p)).filter_map(|p| text(p)).collect();
    let included: BTreeSet<String> = texts
        .iter()
        .flat_map(|t| t.lines())
        .filter_map(|l| l.trim().strip_prefix("#include"))
        .map(|r| r.trim().trim_matches(|c| c == '"' || c == '<' || c == '>'))
        .map(|r| r.rsplit('/').next().unwrap_or(r).to_owned())
        .collect();
    let extra: Vec<String> = files
        .iter()
        .filter(|p| !source(p) && included.contains(p.rsplit('/').next().unwrap_or(p)))
        .filter_map(|p| text(p))
        .collect();
    texts.extend(extra);
    texts
}

/// The value of each value option (`#define NAME value // [a b c]`, and `const` options) as the
/// pack is read now, for its `.properties` files; a name given two values is left out.
pub fn value_options(pack: &Pack) -> BTreeMap<String, String> {
    let mut values: BTreeMap<String, Option<String>> = BTreeMap::new();
    for text in sources(pack, true) {
        for d in text.lines().filter_map(declaration) {
            if d.kind == Kind::Switch {
                continue;
            }
            match values.get_mut(&d.name) {
                Some(seen) if seen.as_deref() != Some(d.default.as_str()) => *seen = None,
                Some(_) => {}
                None => _ = values.insert(d.name, Some(d.default)),
            }
        }
    }
    values.into_iter().filter_map(|(k, v)| Some((k, v?))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn declarations_and_rewrites() {
        let v = map(&[
            ("AO", "false"),
            ("BLOOM", "true"),
            ("LEVEL", "3"),
            ("SCALE", "0.5"),
            ("shadowMapResolution", "1024"),
            ("sunPathRotation", "-20"),
            ("BAD", "1; discard"),
        ]);
        assert_eq!(rewrite("  #define AO // ambient", &v).unwrap(), "  //#define AO // ambient");
        assert_eq!(rewrite("// #define BLOOM", &v).unwrap(), "#define BLOOM");
        assert_eq!(rewrite("#define LEVEL 2 // [1 2 3]", &v).unwrap(), "#define LEVEL 3 // [1 2 3]");
        assert_eq!(
            rewrite("#define SCALE 0.75  // resolution. [0.5 0.75 1.0]", &v).unwrap(),
            "#define SCALE 0.5  // resolution. [0.5 0.75 1.0]"
        );
        assert_eq!(
            rewrite("const int shadowMapResolution = 2048; // [1024 2048]", &v).unwrap(),
            "const int shadowMapResolution = 1024; // [1024 2048]"
        );
        assert_eq!(
            rewrite("const float\tsunPathRotation\t= -35;\t//[-40 -35 -20]", &v).unwrap(),
            "const float\tsunPathRotation\t= -20;\t//[-40 -35 -20]"
        );
        assert!(rewrite("#define LEVEL 2", &v).is_none());
        assert!(rewrite("#define OTHER 1 // [1 2]", &v).is_none());
        assert!(rewrite("#define BAD 1 // [1 2]", &v).is_none());
        assert!(rewrite("const uint LEVEL = 2u; // [", &v).is_none());
        assert!(declaration("#define MIX(a, b) mix(a, b)").is_none());
        assert!(declaration("#define F(x)").is_none());
    }

    #[test]
    fn values_file() {
        let v = parse_values("# c\nA=1\n B = true \nC=\nD=1;x\n");
        assert_eq!(v, map(&[("A", "1"), ("B", "true")]));
        assert_eq!(values_text(&v), "A=1\nB=true\n");
    }

    #[test]
    fn profiles_expand() {
        let options: BTreeMap<String, ShaderOption> = ["A", "B", "C"]
            .iter()
            .map(|n| {
                let o = ShaderOption { name: n.to_string(), kind: Kind::Value, default: "1".into(), values: vec![] };
                (n.to_string(), o)
            })
            .collect();
        let raw: BTreeMap<&str, &str> = [("LOW", "A:1 !B"), ("HIGH", "profile.LOW A:3 C X:2")].into_iter().collect();
        let mut set = Vec::new();
        expand("HIGH", &raw, &options, &mut set, 0);
        assert_eq!(
            set,
            vec![("B".into(), "false".into()), ("A".into(), "3".into()), ("C".into(), "true".into())]
        );
    }
}
