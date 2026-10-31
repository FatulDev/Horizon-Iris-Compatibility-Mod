//! The first pass over a stage's tokens: what it declares at global scope, which of its
//! functions take samplers, which outputs and built-ins it uses.

use std::collections::{BTreeSet, HashMap};
use std::fmt::Write;

use naga::ShaderStage;

use super::Error;
use super::rewrite::attribute_value;
use crate::Preprocessed;
use crate::macros::lex;

/// A varying: its name and its declaration with `@` where the name goes (`flat ivec2 @[2];`).
pub(super) struct Var {
    pub(super) name: String,
    pub(super) decl: String,
    /// Of a struct type: the pack's declaration stays as a private global, and the members
    /// travel in varyings of ours (`struct_parts`).
    pub(super) structure: bool,
}

pub(super) struct Stage<'p> {
    pub(super) kind: ShaderStage,
    pub(super) pp: &'p Preprocessed,
    pub(super) tokens: Vec<&'p str>,
    /// Token ranges to blank out (declarations moved into the header).
    pub(super) removed: Vec<std::ops::Range<usize>>,
    /// Name, type with its array suffix, token of the type.
    pub(super) uniforms: Vec<(String, String, usize)>,
    pub(super) samplers: Vec<(String, String)>,
    pub(super) ins: Vec<Var>,
    pub(super) outs: Vec<Var>,
    /// Vertex inputs, now private globals, and the value our `main` gives each.
    /// Vertex inputs: name, our composite value, type.
    pub(super) attributes: Vec<(String, String, String)>,
    /// A gbuffers-style program: the matrices are the frame's, not the quad's.
    pub(super) gbuffers: bool,
    /// The pack's fragment outputs, declared again with their locations.
    pub(super) frag_outs: Vec<(u32, String)>,
    /// The pack's functions that take samplers: which parameters are samplers.
    pub(super) sampler_functions: HashMap<String, Vec<bool>>,
    /// The pack's structs: each member's type and name with its array suffix.
    pub(super) structs: HashMap<String, Vec<(String, String)>>,
    pub(super) outputs: BTreeSet<u32>,
    pub(super) frag_coord: bool,
    /// Tokens written differently, by index.
    pub(super) replaced: HashMap<usize, String>,
    /// Globals whose initializer runs in our `main` (`g = hz_init_g();`), in order.
    pub(super) inits: Vec<String>,
    pub(super) errors: Vec<Error>,
    /// Images (Iris Features): name, type (`uimage3D`), format qualifier (`r32ui`), access
    /// (`readonly `, `writeonly ` or empty).
    pub(super) images: Vec<(String, String, String, String)>,
    /// Storage-buffer blocks (Iris Features): the pack's binding of each.
    pub(super) buffers: Vec<u32>,
    /// `bool` members of storage-buffer blocks, stored as `uint` (a bool is not host-shareable).
    pub(super) bool_members: Vec<String>,
    /// Token ranges of storage-buffer blocks (their declarations are not accesses).
    pub(super) block_ranges: Vec<std::ops::Range<usize>>,
    /// Text written before / after a token, which is otherwise rewritten as usual.
    pub(super) before: HashMap<usize, String>,
    pub(super) after: HashMap<usize, String>,
    /// Function definitions: the `{` opening the body and the return type.
    pub(super) bodies: Vec<(usize, String)>,
    /// `(` of sampling calls on a shadow sampler whose result is swizzled (`texture(s, p).x`):
    /// the call is wrapped in `vec4(...)`, as naga takes no swizzle of a scalar.
    pub(super) shadow_swizzles: std::collections::HashSet<usize>,
    /// Interface block instance names (`} data_out;`): `data_out.` goes.
    pub(super) instances: Vec<String>,
    /// Globals whose constant initializer calls a function naga can't evaluate.
    pub(super) runtime_consts: Vec<String>,
}

/// Types a value converts to by constructor (scalars, vectors, matrices).
pub(super) const CONVERTIBLE: [&str; 25] = [
    "float", "int", "uint", "bool", "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "uvec2",
    "uvec3", "uvec4", "bvec2", "bvec3", "bvec4", "mat2", "mat3", "mat4", "mat2x2", "mat3x3",
    "mat4x4", "mat2x3", "mat3x4", "mat4x3",
];

/// Built-ins naga can't evaluate in a constant expression.
pub(super) const NOT_CONSTANT: [&str; 12] = [
    "mix", "smoothstep", "step", "pow", "exp", "exp2", "log", "log2", "sqrt", "inversesqrt",
    "normalize", "length",
];

/// Memory qualifiers of images and buffers: kept as access, or dropped (`restrict`...).
pub(super) const MEMORY: [&str; 5] = ["readonly", "writeonly", "restrict", "coherent", "volatile"];

pub(super) const STORAGE: [&str; 5] = ["uniform", "in", "out", "attribute", "varying"];
pub(super) const QUALIFIERS: [&str; 11] = [
    "flat",
    "smooth",
    "noperspective",
    "centroid",
    "invariant",
    "highp",
    "mediump",
    "lowp",
    "precise",
    "sample",
    "const",
];
/// The compatibility profile's uniform state (with its `Inverse`/`Transpose` forms).
pub(super) const UNIFORM_STATE: [&str; 5] = [
    "gl_ModelViewMatrix",
    "gl_ProjectionMatrix",
    "gl_ModelViewProjectionMatrix",
    "gl_NormalMatrix",
    "gl_TextureMatrix",
];
/// The compatibility profile's vertex inputs.
pub(super) const VERTEX_INPUTS: [&str; 6] = [
    "gl_Vertex",
    "gl_Color",
    "gl_SecondaryColor",
    "gl_Normal",
    "gl_FogCoord",
    "gl_MultiTexCoord",
];
/// Words that became keywords after GLSL 1.x and that old packs use as names.
pub(super) const LATER_KEYWORDS: [&str; 17] = [
    "sample",
    "buffer",
    "shared",
    "filter",
    "input",
    "output",
    "common",
    "partition",
    "active",
    "resource",
    "patch",
    "subroutine",
    "coherent",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
];

/// Vulkan GLSL's texture and sampler types, which naga's frontend reads as types, renamed.
const VULKAN_TYPES: [(&str, &str); 9] = [
    ("sampler", "hz_kw_sampler"),
    ("samplerShadow", "hz_kw_samplerShadow"),
    ("texture1D", "hz_kw_texture1D"),
    ("texture2D", "hz_kw_texture2D"),
    ("texture3D", "hz_kw_texture3D"),
    ("textureCube", "hz_kw_textureCube"),
    ("texture2DArray", "hz_kw_texture2DArray"),
    ("texture2DMS", "hz_kw_texture2DMS"),
    ("textureCubeArray", "hz_kw_textureCubeArray"),
];

pub(super) fn significant(t: &str) -> bool {
    !t.trim().is_empty()
}

/// `sampler2D`, `isampler3D`, `usampler2D`, `sampler2DShadow`...
pub(super) fn is_sampler_type(t: &str) -> bool {
    t.trim_start_matches(['i', 'u']).starts_with("sampler")
}

pub(super) fn is_ident(t: &str) -> bool {
    t.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
}

impl<'p> Stage<'p> {
    pub(super) fn scan(pp: &'p Preprocessed, kind: ShaderStage) -> Self {
        let mut s = Stage {
            kind,
            pp,
            tokens: lex(&pp.code),
            removed: Vec::new(),
            uniforms: Vec::new(),
            samplers: Vec::new(),
            ins: Vec::new(),
            outs: Vec::new(),
            attributes: Vec::new(),
            frag_outs: Vec::new(),
            sampler_functions: HashMap::new(),
            structs: HashMap::new(),
            outputs: BTreeSet::new(),
            frag_coord: false,
            gbuffers: false,
            replaced: HashMap::new(),
            inits: Vec::new(),
            errors: Vec::new(),
            images: Vec::new(),
            buffers: Vec::new(),
            bool_members: Vec::new(),
            block_ranges: Vec::new(),
            before: HashMap::new(),
            after: HashMap::new(),
            bodies: Vec::new(),
            shadow_swizzles: std::collections::HashSet::new(),
            instances: Vec::new(),
            runtime_consts: Vec::new(),
        };
        s.rename_vulkan_types();
        let (mut start, mut depth, mut parens) = (0, 0usize, 0usize);
        for i in 0..s.tokens.len() {
            match s.tokens[i] {
                "(" => parens += 1,
                ")" => parens = parens.saturating_sub(1),
                "{" if depth == 0 && parens == 0 => {
                    s.global(start, i, true);
                    s.structure(start, i);
                    depth = 1;
                }
                "{" => depth += 1,
                "}" if depth > 0 => {
                    depth -= 1;
                    if depth == 0 {
                        start = i + 1;
                    }
                }
                ";" if depth == 0 && parens == 0 => {
                    s.global(start, i, false);
                    start = i + 1;
                }
                "gl_FragData" | "gl_FragColor" if kind == ShaderStage::Fragment => {
                    let n = if s.tokens[i] == "gl_FragColor" {
                        Some(0)
                    } else {
                        s.index_after(i).map(|(n, _)| n)
                    };
                    match n {
                        Some(n) => _ = s.outputs.insert(n),
                        None => s
                            .errors
                            .push(s.error(i, "`gl_FragData` needs a constant index")),
                    }
                }
                "gl_FragCoord" => s.frag_coord = true,
                "gl_TexCoord" => {
                    if let Some((n, _)) = s.index_after(i) {
                        s.builtin_varying(format!("hz_TexCoord{n}"));
                    }
                }
                "gl_FrontColor" => s.builtin_varying("hz_FrontColor".into()),
                "gl_Color" if kind == ShaderStage::Fragment => {
                    s.builtin_varying("hz_FrontColor".into())
                }
                _ => {}
            }
        }
        s.bool_member_accesses();
        s.instance_prefixes();
        s.returns();
        s.vector_modulo();
        s.shadow_swizzle_calls();
        let types = s.declared_types();
        s.relational_conversions(&types);
        s.matrix_swizzle_assignments(&types);
        s
    }

    /// Names that desktop GLSL leaves free but Vulkan GLSL takes for its separate texture and
    /// sampler types (`sampler2D sampler`): renamed, unless called (`texture2D(...)`).
    fn rename_vulkan_types(&mut self) {
        for i in 0..self.tokens.len() {
            let Some(&(_, renamed)) = VULKAN_TYPES.iter().find(|(t, _)| *t == self.tokens[i])
            else {
                continue;
            };
            let called = self.tokens[i + 1..]
                .iter()
                .find(|t| significant(t))
                .is_some_and(|&t| t == "(");
            if !called {
                self.tokens[i] = renamed;
            }
        }
    }

    pub(super) fn builtin_varying(&mut self, name: String) {
        let list = if self.kind == ShaderStage::Vertex {
            &mut self.outs
        } else {
            &mut self.ins
        };
        if !list.iter().any(|v| v.name == name) {
            list.push(Var {
                name,
                decl: "vec4 @;".into(),
                structure: false,
            });
        }
    }

    /// `[ n ]` right after token `i`: the number and the index of the `]`.
    pub(super) fn index_after(&self, i: usize) -> Option<(u32, usize)> {
        let mut it = (i + 1..self.tokens.len()).filter(|&j| significant(self.tokens[j]));
        let open = it.next()?;
        let n = it.next()?;
        let close = it.next()?;
        (self.tokens[open] == "[" && self.tokens[close] == "]")
            .then(|| self.tokens[n].parse().ok().map(|n| (n, close)))
            .flatten()
    }

    pub(super) fn error(&self, token: usize, message: &str) -> Error {
        let line: usize = self.tokens[..token.min(self.tokens.len())]
            .iter()
            .map(|t| t.matches('\n').count())
            .sum();
        let o = self.pp.lines.get(line).copied().unwrap_or_default();
        Error {
            file: self
                .pp
                .files
                .get(o.file as usize)
                .cloned()
                .unwrap_or_default(),
            line: o.line,
            message: message.to_owned(),
        }
    }

    /// A statement at global scope, tokens `from..to` (without its `;`, or before a `{`).
    pub(super) fn global(&mut self, from: usize, to: usize, body: bool) {
        let words: Vec<usize> = (from..to)
            .filter(|&i| significant(self.tokens[i]))
            .collect();
        // A prototype: a `(` before any `=`, and no storage qualifier.
        let paren = words
            .iter()
            .position(|&i| matches!(self.tokens[i], "(" | "="));
        let prototype = paren.is_some_and(|p| self.tokens[words[p]] == "(")
            && !words.iter().any(|&i| STORAGE.contains(&self.tokens[i]));
        if body && words.iter().any(|&i| self.tokens[i] == "buffer") {
            return self.buffer_block(&words, to);
        }
        if body
            && words
                .iter()
                .any(|&i| matches!(self.tokens[i], "in" | "out" | "varying"))
            && !words.iter().any(|&i| self.tokens[i] == "(")
        {
            return self.interface_block(&words, from, to);
        }
        if body {
            self.body_of(&words, to);
        }
        if body || prototype {
            return self.function(&words);
        }
        let mut k = 0;
        let mut storage = None;
        let mut location = None;
        let mut interpolation = String::new();
        let mut format = String::new();
        let mut access = String::new();
        while let Some(&i) = words.get(k) {
            let w = self.tokens[i];
            if w == "layout" {
                // layout(location = n, ...), or an image's format: layout(rgba16f)
                let mut j = k + 1;
                while let Some(&t) = words.get(j) {
                    j += 1;
                    let word = self.tokens[t];
                    if word == "location" {
                        location = words
                            .get(j + 1)
                            .and_then(|&n| self.tokens[n].parse::<u32>().ok());
                    }
                    let assigned = words.get(j).is_some_and(|&n| self.tokens[n] == "=");
                    let after_eq = j >= 2 && words.get(j - 2).is_some_and(|&n| self.tokens[n] == "=");
                    if is_ident(word) && !assigned && !after_eq && word != "layout" {
                        format = word.to_owned();
                    }
                    if word == ")" {
                        break;
                    }
                }
                k = j;
                continue;
            }
            if MEMORY.contains(&w) {
                if matches!(w, "readonly" | "writeonly") {
                    access = format!("{w} ");
                }
                self.removed.push(i..i + 1);
            } else if STORAGE.contains(&w) {
                storage = Some(w);
            } else if QUALIFIERS.contains(&w) {
                if matches!(
                    w,
                    "flat" | "smooth" | "noperspective" | "centroid" | "sample"
                ) {
                    write!(interpolation, "{w} ").unwrap();
                }
            } else {
                break;
            }
            k += 1;
        }
        let Some(storage) = storage else {
            return self.initializer(&words, to);
        };
        let Some(&ty_at) = words.get(k) else { return };
        let ty = self.tokens[ty_at].to_owned();
        if words[k..].iter().any(|&i| self.tokens[i] == "{") {
            return; // An interface block: left as the pack wrote it.
        }
        // Declarators: name [array] [= init], separated by commas.
        let mut declarators: Vec<(String, String)> = Vec::new();
        let mut j = k + 1;
        while let Some(&n) = words.get(j) {
            let name = self.tokens[n].to_owned();
            let mut array = String::new();
            j += 1;
            let mut nesting = 0;
            while let Some(&t) = words.get(j) {
                match self.tokens[t] {
                    "," if nesting == 0 => break,
                    "(" | "[" => nesting += 1,
                    ")" | "]" => nesting -= 1,
                    _ => {}
                }
                if self.tokens[t] == "=" && nesting == 0 {
                    // An initializer: uniforms are set by the game, so it is dropped.
                    while words.get(j).is_some_and(|&t| self.tokens[t] != ",") {
                        j += 1;
                    }
                    break;
                }
                array.push_str(self.tokens[t]);
                j += 1;
            }
            j += 1;
            declarators.push((name, array));
        }
        let fragment = self.kind == ShaderStage::Fragment;
        let input = matches!(storage, "in" | "attribute") || (storage == "varying" && fragment);
        self.removed.push(from..to + 1);
        for (name, array) in declarators {
            let decl = format!("{interpolation}{ty} @{array};");
            match storage {
                "uniform" if is_sampler_type(&ty) => self.samplers.push((name, ty.clone())),
                "uniform" if ty.contains("image") => {
                    if format.is_empty() && !access.starts_with("writeonly") {
                        self.errors.push(self.error(
                            ty_at,
                            &format!("`{name}`: an image without a format qualifier"),
                        ));
                    }
                    if !self.images.iter().any(|i| i.0 == name) {
                        self.images
                            .push((name, ty.clone(), format.clone(), access.clone()));
                    }
                }
                "uniform" => self.uniforms.push((name, format!("{ty}{array}"), ty_at)),
                _ if input && !fragment => {
                    let value = attribute_value(&name, &ty);
                    self.attributes.push((name, value, ty.clone()));
                    self.removed.pop();
                    // A private global of ours now: `in`/`attribute` and qualifiers go.
                    for &w in &words[..k] {
                        self.removed.push(w..w + 1);
                    }
                }
                _ if self.structs.contains_key(&ty) => {
                    self.removed.pop();
                    for &w in &words[..k] {
                        self.removed.push(w..w + 1);
                    }
                    let var = Var {
                        name,
                        decl,
                        structure: true,
                    };
                    if input {
                        self.ins.push(var);
                    } else {
                        self.outs.push(var);
                    }
                }
                _ if input => self.ins.push(Var {
                    name,
                    decl,
                    structure: false,
                }),
                _ if fragment => {
                    let n = location.unwrap_or(self.outputs.len() as u32);
                    self.outputs.insert(n);
                    self.frag_outs
                        .push((n, format!("layout(location = {n}) out {ty} {name}{array};")));
                }
                _ => self.outs.push(Var {
                    name,
                    decl,
                    structure: false,
                }),
            }
        }
    }

    /// A global `T name = a ? b : c;` or, without `const`, any initialized global: naga only
    /// evaluates constant initializers at global scope (a conditional is not one; GL drivers
    /// also take uniforms and other globals there), so each initializer becomes a function of
    /// ours on the same line (`T a; T hz_init_a() { return T(...); } T b; ...`), called by our
    /// `main` before the pack's, in file order. The return converts as GLSL's initialization
    /// does (`vec2 v = ivec2(...)`).
    pub(super) fn initializer(&mut self, words: &[usize], to: usize) {
        let text = |i: usize| self.tokens[i];
        let Some(eq) = words.iter().position(|&i| text(i) == "=") else {
            return;
        };
        // Declarators, split at top-level commas: (first word, `=` if any, end).
        let mut declarators = Vec::new();
        let (mut begin, mut nesting, mut assign) = (0, 0i32, None);
        for (n, &i) in words.iter().enumerate() {
            match text(i) {
                "(" | "[" | "{" => nesting += 1,
                ")" | "]" | "}" => nesting -= 1,
                "=" if nesting == 0 && assign.is_none() => assign = Some(n),
                "," if nesting == 0 => {
                    declarators.push((begin, assign.take(), n));
                    begin = n + 1;
                }
                _ => {}
            }
        }
        declarators.push((begin, assign, words.len()));
        let constant = words[..eq].iter().any(|&i| text(i) == "const");
        let conditional =
            |d: &(usize, Option<usize>, usize)| words[d.0..d.2].iter().any(|&i| text(i) == "?");
        // naga's constant evaluator lacks some built-ins (`mix`...): such a constant, and one
        // that reads such a constant, is initialized at run time instead.
        let runtime = words[eq + 1..].iter().enumerate().any(|(n, &i)| {
            let t = text(i);
            let called = words.get(eq + 2 + n).is_some_and(|&j| text(j) == "(");
            (called && NOT_CONSTANT.contains(&t)) || self.runtime_consts.iter().any(|c| c == t)
        });
        if constant && !declarators.iter().any(conditional) && !runtime {
            return;
        }
        // The first declarator's name ends the type.
        let first_end = declarators[0].1.unwrap_or(declarators[0].2);
        let Some(name_at) = words[..first_end].iter().rposition(|&i| is_ident(text(i))) else {
            return;
        };
        // The type, with a `vec3[]`'s count written in.
        let mut ty = String::new();
        for &i in &words[..name_at] {
            match text(i) {
                "const" => {}
                "]" if ty.ends_with('[') => match self.array_length(i) {
                    Some(n) => write!(ty, "{n}]").unwrap(),
                    None => return,
                },
                t if ty.is_empty() || t == "[" => ty.push_str(t),
                // Qualifiers before the type (`highp`...): the type is the last word.
                t => ty = t.to_owned(),
            }
        }
        if ty.is_empty() {
            return;
        }
        let mut replaced = Vec::new();
        let mut names = Vec::new();
        for (n, &(begin, assign, end)) in declarators.iter().enumerate() {
            let name_at = if n == 0 { name_at } else { begin };
            let name = text(words[name_at]).to_owned();
            if !is_ident(&name) {
                return;
            }
            let stop = assign.unwrap_or(end);
            let mut array: String = words[name_at + 1..stop].iter().map(|&i| text(i)).collect();
            if array == "[]" {
                // `T x[] = T[](a, b)`: the count is written in, as the rewrite does for the name.
                match assign.and_then(|a| self.array_length(words[a - 1])) {
                    Some(n) => array = format!("[{n}]"),
                    None => return,
                }
            }
            let next = if n + 1 < declarators.len() {
                format!(" {ty}")
            } else {
                String::new()
            };
            let end_token = words.get(end).copied().unwrap_or(to);
            match assign {
                Some(a) => {
                    // Scalars, vectors and matrices convert; arrays and structs must match.
                    let scalar = array.is_empty()
                        && !ty.contains('[')
                        && (matches!(ty.as_str(), "float" | "int" | "uint" | "bool")
                            || ty.contains("vec")
                            || ty.starts_with("mat"));
                    let (open, close) = if scalar {
                        (format!("{ty}("), ")")
                    } else {
                        (String::new(), "")
                    };
                    replaced.push((
                        words[a],
                        format!("; {ty}{array} hz_init_{name}() {{ return {open}"),
                    ));
                    replaced.push((end_token, format!("{close}; }}{next}")));
                    names.push(name);
                }
                None if n + 1 < declarators.len() => replaced.push((end_token, format!(";{next}"))),
                None => {}
            }
        }
        for &i in &words[..name_at] {
            if text(i) == "const" {
                self.removed.push(i..i + 1);
            }
        }
        self.replaced.extend(replaced);
        if constant {
            self.runtime_consts.extend(names.iter().cloned());
        }
        self.inits.extend(names);
    }

    /// `struct Name {` at token `open`, whose words start at `from`: its members, to `}`.
    fn structure(&mut self, from: usize, open: usize) {
        let words: Vec<&str> = self.tokens[from..open]
            .iter()
            .copied()
            .filter(|t| significant(t))
            .collect();
        let [.., "struct", name] = words[..] else {
            return;
        };
        let mut members = Vec::new();
        let mut decl: Vec<&str> = Vec::new();
        for &t in &self.tokens[open + 1..] {
            match t {
                "}" => break,
                ";" => {
                    // `T a, b[2]`, after any precision or other qualifier.
                    let words: Vec<&str> =
                        decl.drain(..).filter(|w| !QUALIFIERS.contains(w)).collect();
                    let Some((ty, rest)) = words.split_first() else {
                        continue;
                    };
                    for d in rest.split(|w| *w == ",") {
                        if let Some((n, array)) = d.split_first() {
                            members.push((ty.to_string(), format!("{n}{}", array.concat())));
                        }
                    }
                }
                t if significant(t) => decl.push(t),
                _ => {}
            }
        }
        self.structs.insert(name.to_owned(), members);
    }

    /// `layout(std430, binding = N) [readonly] buffer Name { members } [instance];` (Iris
    /// Features): bound at set 0, binding 1000 + N; `bool` members stored as `uint`.
    pub(super) fn buffer_block(&mut self, words: &[usize], open: usize) {
        let text = |i: usize| self.tokens[i];
        let mut binding = None;
        let mut layout_at = None;
        let mut k = 0;
        while let Some(&i) = words.get(k) {
            match text(i) {
                "layout" => {
                    layout_at = Some(i);
                    let mut j = k + 1;
                    while let Some(&t) = words.get(j) {
                        if text(t) == "binding" {
                            binding = words
                                .get(j + 2)
                                .and_then(|&n| text(n).parse::<u32>().ok());
                        }
                        j += 1;
                        if text(t) == ")" {
                            // Everything after `layout` up to `)` goes; `layout` is rewritten.
                            self.removed.push(i + 1..t + 1);
                            break;
                        }
                    }
                    k = j;
                    continue;
                }
                w if MEMORY.contains(&w) && !matches!(w, "readonly" | "writeonly") => {
                    self.removed.push(i..i + 1);
                }
                _ => {}
            }
            k += 1;
        }
        let Some(binding) = binding else {
            let at = words.first().copied().unwrap_or(open);
            self.errors
                .push(self.error(at, "a storage buffer without `layout(binding = n)`"));
            return;
        };
        let decl = format!("layout(std430, set = 0, binding = {})", 1000 + binding);
        match layout_at {
            Some(at) => _ = self.replaced.insert(at, decl),
            None => {
                let first = words[0];
                let t = text(first).to_owned();
                self.replaced.insert(first, format!("{decl} {t}"));
            }
        }
        if !self.buffers.contains(&binding) {
            self.buffers.push(binding);
        }
        // The members, to the matching `}`.
        let mut depth = 0;
        let mut close = open;
        let mut statement_start = true;
        for i in open..self.tokens.len() {
            match text(i) {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = i;
                        break;
                    }
                }
                ";" => statement_start = true,
                t if significant(t) && depth == 1 => {
                    if statement_start && t == "bool" {
                        self.replaced.insert(i, "uint".into());
                        if let Some(n) = (i + 1..self.tokens.len()).find(|&j| significant(text(j))) {
                            self.bool_members.push(text(n).to_owned());
                        }
                    }
                    statement_start = false;
                }
                _ => {}
            }
        }
        self.block_ranges.push(open..close + 1);
    }

    /// Reads of a `bool` member stored as `uint` compare it with zero; writes convert.
    pub(super) fn bool_member_accesses(&mut self) {
        if self.bool_members.is_empty() {
            return;
        }
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let prev = |i: usize| (0..i).rev().find(|&j| significant(self.tokens[j]));
        let mut edits: Vec<(usize, String)> = Vec::new();
        for i in 0..self.tokens.len() {
            if !self.bool_members.iter().any(|m| m == text(i)) {
                continue;
            }
            if self.block_ranges.iter().any(|r| r.contains(&i)) {
                continue;
            }
            if prev(i).is_some_and(|j| text(j) == ".") {
                continue;
            }
            let eq = next(i).filter(|&j| text(j) == "=");
            let compare = eq.and_then(next).is_some_and(|j| text(j) == "=");
            match eq {
                Some(eq) if !compare => {
                    // `flag = expr;` → `flag = uint(expr);`
                    let mut depth = 0i32;
                    let mut end = None;
                    for j in eq + 1..self.tokens.len() {
                        match text(j) {
                            "(" | "[" | "{" => depth += 1,
                            ")" | "]" | "}" => depth -= 1,
                            ";" if depth == 0 => {
                                end = Some(j);
                                break;
                            }
                            _ => {}
                        }
                    }
                    if let Some(end) = end {
                        edits.push((eq, "= uint(".into()));
                        edits.push((end, ");".into()));
                    }
                }
                _ => edits.push((i, format!("({} != 0u)", text(i)))),
            }
        }
        for (i, t) in edits {
            self.replaced.entry(i).or_insert(t);
        }
    }

    /// `out DATA { flat vec3 a; vec2 b[2]; } [name];` (and `in`): each member a varying of its
    /// own, matched by name as other varyings are; the block goes.
    pub(super) fn interface_block(&mut self, words: &[usize], from: usize, open: usize) {
        let text = |i: usize| self.tokens[i];
        let storage = words
            .iter()
            .map(|&i| text(i))
            .find(|w| matches!(*w, "in" | "out" | "varying"))
            .unwrap_or("out");
        let fragment = self.kind == ShaderStage::Fragment;
        let input = storage == "in" || (storage == "varying" && fragment);
        // The members, to the matching `}`, then the instance name and `;`.
        let mut depth = 0;
        let mut close = open;
        for i in open..self.tokens.len() {
            match text(i) {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let mut end = close;
        for i in close + 1..self.tokens.len() {
            let t = text(i);
            if t == ";" {
                end = i;
                break;
            }
            if is_ident(t) {
                self.instances.push(t.to_owned());
            }
        }
        let member_tokens: Vec<usize> = (open + 1..close)
            .filter(|&i| significant(text(i)))
            .collect();
        for statement in member_tokens.split(|&i| text(i) == ";") {
            let mut k = 0;
            let mut interpolation = String::new();
            while let Some(&i) = statement.get(k) {
                let w = text(i);
                if QUALIFIERS.contains(&w) {
                    if matches!(w, "flat" | "smooth" | "noperspective" | "centroid" | "sample") {
                        write!(interpolation, "{w} ").unwrap();
                    }
                    k += 1;
                } else {
                    break;
                }
            }
            let Some(&ty_at) = statement.get(k) else {
                continue;
            };
            let ty = text(ty_at).to_owned();
            let rest = &statement[k + 1..];
            for declarator in rest.split(|&i| text(i) == ",") {
                let Some((&name, array)) = declarator.split_first() else {
                    continue;
                };
                let array: String = array.iter().map(|&i| text(i)).collect();
                let var = Var {
                    name: text(name).to_owned(),
                    decl: format!("{interpolation}{ty} @{array};"),
                    structure: false,
                };
                let list = if input { &mut self.ins } else { &mut self.outs };
                if !list.iter().any(|v| v.name == var.name) {
                    list.push(var);
                }
            }
        }
        self.removed.push(from..end + 1);
    }

    /// `data_out.color` → `color` for interface block instances.
    pub(super) fn instance_prefixes(&mut self) {
        if self.instances.is_empty() {
            return;
        }
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        for i in 0..self.tokens.len() {
            if !self.instances.iter().any(|n| n == self.tokens[i]) {
                continue;
            }
            if self.removed.iter().any(|r| r.contains(&i)) {
                continue;
            }
            if let Some(dot) = next(i).filter(|&j| self.tokens[j] == ".") {
                self.removed.push(i..dot + 1);
            }
        }
    }

    /// A function definition: remember its body and return type.
    pub(super) fn body_of(&mut self, words: &[usize], open: usize) {
        let Some(paren) = words.iter().position(|&i| self.tokens[i] == "(") else {
            return;
        };
        if paren < 2 {
            return;
        }
        // The return type: the words before the name, without qualifiers.
        let ty: String = words[..paren - 1]
            .iter()
            .map(|&i| self.tokens[i])
            .filter(|w| !QUALIFIERS.contains(w) && !matches!(*w, "inline" | "highp" | "mediump" | "lowp"))
            .collect();
        self.bodies.push((open, ty));
    }

    /// Every `return` braced as `{ if (true) return ...; }` (naga's frontend leaves code after
    /// an unconditional return half-built, and braces keep a following `else` where it was),
    /// and a function that may run off its end returns a zero value there, as GL drivers do.
    pub(super) fn returns(&mut self) {
        let text = |i: usize| self.tokens[i];
        let mut edits_before: Vec<(usize, String)> = Vec::new();
        let mut edits_after: Vec<(usize, String)> = Vec::new();
        for (open, ty) in self.bodies.clone() {
            let mut depth = 0;
            let mut close = None;
            for i in open..self.tokens.len() {
                match text(i) {
                    "{" => depth += 1,
                    "}" => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(i);
                            break;
                        }
                    }
                    "return" => {
                        let mut d = 0i32;
                        let mut end = None;
                        for j in i + 1..self.tokens.len() {
                            match text(j) {
                                "(" | "[" | "{" => d += 1,
                                ")" | "]" | "}" => d -= 1,
                                ";" if d == 0 => {
                                    end = Some(j);
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if let Some(end) = end {
                            edits_before.push((i, "{ if (true) ".into()));
                            edits_after.push((end, " }".into()));
                            // GLSL converts the value to the return type (`return 0;` from a
                            // `uint` function); naga wants it converted.
                            if CONVERTIBLE.contains(&ty.as_str()) && end > i + 1 {
                                edits_after.push((i, format!(" {ty}(")));
                                edits_before.push((end, ")".into()));
                            }
                        }
                    }
                    _ => {}
                }
            }
            if let Some(close) = close
                && ty != "void"
                && !ty.is_empty()
            {
                edits_before.push((close, format!("{{ {ty} hz_ret; return hz_ret; }} ")));
            }
        }
        for (i, t) in edits_before {
            self.before.entry(i).or_default().push_str(&t);
        }
        for (i, t) in edits_after {
            self.after.entry(i).or_default().push_str(&t);
        }
    }

    /// `ivec2(expr) % 512` → `ivec2(expr) % ivec2(512)`: naga has no vector % scalar.
    pub(super) fn vector_modulo(&mut self) {
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let prev = |i: usize| (0..i).rev().find(|&j| significant(self.tokens[j]));
        let mut edits = Vec::new();
        for i in 0..self.tokens.len() {
            if text(i) != "%" {
                continue;
            }
            let Some(close) = prev(i).filter(|&j| text(j) == ")") else {
                continue;
            };
            let mut depth = 0;
            let mut open = None;
            for j in (0..=close).rev() {
                match text(j) {
                    ")" => depth += 1,
                    "(" => {
                        depth -= 1;
                        if depth == 0 {
                            open = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let Some(callee) = open.and_then(prev).map(text) else {
                continue;
            };
            if !matches!(
                callee,
                "ivec2" | "ivec3" | "ivec4" | "uvec2" | "uvec3" | "uvec4"
            ) {
                continue;
            }
            let Some(operand) = next(i) else { continue };
            let after = next(operand).map(text);
            let simple = (is_ident(text(operand)) || text(operand).starts_with(|c: char| c.is_ascii_digit()))
                && !matches!(after, Some("(" | "." | "["));
            if simple && !self.replaced.contains_key(&operand) {
                edits.push((operand, format!("{callee}({})", text(operand))));
            }
        }
        self.replaced.extend(edits);
    }

    /// `texture(shadowSampler, p).x`: see `shadow_swizzles`.
    pub(super) fn shadow_swizzle_calls(&mut self) {
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let shadow: Vec<&str> = self
            .samplers
            .iter()
            .filter(|(_, ty)| ty.ends_with("Shadow"))
            .map(|(n, _)| n.as_str())
            .collect();
        if shadow.is_empty() {
            return;
        }
        let mut found = Vec::new();
        for i in 0..self.tokens.len() {
            if !matches!(text(i), "texture" | "textureLod" | "texture2D" | "texture2DLod" | "textureProj") {
                continue;
            }
            let Some(open) = next(i).filter(|&j| text(j) == "(") else {
                continue;
            };
            let Some(arg) = next(open) else { continue };
            if !shadow.contains(&text(arg)) {
                continue;
            }
            let mut depth = 0;
            let mut close = None;
            for j in open..self.tokens.len() {
                match text(j) {
                    "(" => depth += 1,
                    ")" => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if close.and_then(next).is_some_and(|j| text(j) == ".") {
                found.push((i, open));
            }
        }
        for (callee, open) in found {
            self.before.entry(callee).or_default().push_str("vec4(");
            self.shadow_swizzles.insert(open);
        }
    }

    /// The declared type of each name (`ivec3 p = ...`, `const uvec3 size = ...`, parameters):
    /// a flat table, the last declaration of a name winning; enough for the conversions below.
    pub(super) fn declared_types(&self) -> HashMap<String, String> {
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let mut types = HashMap::new();
        for i in 0..self.tokens.len() {
            if !CONVERTIBLE.contains(&text(i)) {
                continue;
            }
            let Some(name) = next(i).filter(|&j| is_ident(text(j))) else {
                continue;
            };
            if CONVERTIBLE.contains(&text(name)) {
                continue;
            }
            if next(name).is_some_and(|j| matches!(text(j), "=" | ";" | "," | ")" | "[")) {
                types.insert(text(name).to_owned(), text(i).to_owned());
            }
        }
        types
    }

    /// `greaterThanEqual(ivec3 a, uvec3 b)`: GLSL converts `a` to `uvec3`; naga calls it
    /// ambiguous. Arguments that are plain names of a signed and an unsigned vector meet.
    pub(super) fn relational_conversions(&mut self, types: &HashMap<String, String>) {
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let mut edits = Vec::new();
        for i in 0..self.tokens.len() {
            if !matches!(
                text(i),
                "lessThan" | "lessThanEqual" | "greaterThan" | "greaterThanEqual" | "equal" | "notEqual"
            ) {
                continue;
            }
            let Some(open) = next(i).filter(|&j| text(j) == "(") else { continue };
            let Some(a) = next(open) else { continue };
            let Some(comma) = next(a).filter(|&j| text(j) == ",") else { continue };
            let Some(b) = next(comma) else { continue };
            if next(b).is_none_or(|j| text(j) != ")") {
                continue;
            }
            let (Some(ta), Some(tb)) = (types.get(text(a)), types.get(text(b))) else {
                continue;
            };
            let signed = |t: &str| t.starts_with("ivec");
            let unsigned = |t: &str| t.starts_with("uvec");
            if signed(ta) && unsigned(tb) {
                edits.push((a, format!("{tb}({})", text(a))));
            } else if unsigned(ta) && signed(tb) {
                edits.push((b, format!("{ta}({})", text(b))));
            }
        }
        for (i, t) in edits {
            self.replaced.entry(i).or_insert(t);
        }
    }

    /// `m[i].xyz = v;` on a matrix column: naga can't assign through a swizzle of an indexed
    /// matrix, so it becomes `{ vec4 hz_c = m[i]; hz_c.xyz = v; m[i] = hz_c; }`.
    pub(super) fn matrix_swizzle_assignments(&mut self, types: &HashMap<String, String>) {
        let text = |i: usize| self.tokens[i];
        let next = |i: usize| (i + 1..self.tokens.len()).find(|&j| significant(self.tokens[j]));
        let mut before = Vec::new();
        let mut replaced = Vec::new();
        let mut after = Vec::new();
        for i in 0..self.tokens.len() {
            let Some(ty) = types.get(text(i)) else { continue };
            let Some(rows) = ty.strip_prefix("mat").and_then(|d| d.chars().last()) else {
                continue;
            };
            let Some(open) = next(i).filter(|&j| text(j) == "[") else { continue };
            let Some(close) = (open..self.tokens.len()).find(|&j| text(j) == "]") else {
                continue;
            };
            let Some(dot) = next(close).filter(|&j| text(j) == ".") else { continue };
            let Some(swizzle) = next(dot) else { continue };
            let Some(eq) = next(swizzle).filter(|&j| text(j) == "=") else { continue };
            if next(eq).is_some_and(|j| text(j) == "=") {
                continue;
            }
            let mut depth = 0i32;
            let Some(end) = (eq + 1..self.tokens.len()).find(|&j| {
                match text(j) {
                    "(" | "[" | "{" => depth += 1,
                    ")" | "]" | "}" => depth -= 1,
                    ";" if depth == 0 => return true,
                    _ => {}
                }
                false
            }) else {
                continue;
            };
            let index: String = (open..=close).map(text).collect();
            let name = text(i);
            before.push((i, format!("{{ vec{rows} hz_c = ")));
            replaced.push((dot, "; hz_c.".to_owned()));
            after.push((end, format!(" {name}{index} = hz_c; }}")));
        }
        for (i, t) in before {
            self.before.entry(i).or_default().push_str(&t);
        }
        for (i, t) in replaced {
            self.replaced.entry(i).or_insert(t);
        }
        for (i, t) in after {
            self.after.entry(i).or_default().push_str(&t);
        }
    }

    /// A function definition or prototype: remember which parameters are samplers.
    pub(super) fn function(&mut self, words: &[usize]) {
        let Some(open) = words.iter().position(|&i| self.tokens[i] == "(") else {
            return;
        };
        let Some(&name) = open.checked_sub(1).and_then(|n| words.get(n)) else {
            return;
        };
        let mut params = Vec::new();
        let mut is_sampler = false;
        for &i in &words[open + 1..] {
            match self.tokens[i] {
                "," | ")" => {
                    params.push(is_sampler);
                    is_sampler = false;
                }
                t if is_sampler_type(t) => is_sampler = true,
                _ => {}
            }
        }
        if params.iter().any(|&p| p) {
            self.sampler_functions
                .insert(self.tokens[name].to_owned(), params);
        }
    }
}
