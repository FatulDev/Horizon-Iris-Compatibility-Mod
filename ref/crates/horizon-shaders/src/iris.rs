// Iris-format features beyond Horizon's own loader (the Iris Features mod): the macros Iris
// defines, the pack's option defaults for its `.properties` files, the resources a pack declares
// in `shaders.properties` (storage buffers, images, custom textures) and its compute programs.
//
// Written from the format's public documentation (Iris's "shaders.properties" and "compute
// shaders" pages), as the rest of this crate is from Horizon's specification.

use std::collections::{BTreeMap, BTreeSet};

use crate::{Macros, Pack, Properties};

/// The Iris version this mod reports to packs (`IRIS_VERSION`, 1.11.3): late enough that packs
/// use their modern paths (storage-buffer layouts without the old bool flags, for one), and past
/// 1.11.2, whose terrain bug on Minecraft 26.2+ packs warn about (Horizon doesn't have it).
pub const IRIS_VERSION: u32 = 11103;

/// The macros Iris adds, for the features this mod provides: a pack tests them to turn on its
/// storage buffers, images and compute programs.
pub fn define_iris(m: &mut Macros) {
    m.define("IS_IRIS", "");
    m.define("IRIS_VERSION", IRIS_VERSION.to_string());
    m.define("IRIS_FEATURE_SSBO", "");
    m.define("IRIS_FEATURE_CUSTOM_IMAGES", "");
    m.define("IRIS_FEATURE_COMPUTE_SHADERS", "");
    m.define("IRIS_FEATURE_CUSTOM_TEXTURE_NAME", "");
}

const SOURCES: [&str; 8] = ["vsh", "fsh", "gsh", "csh", "tcs", "tes", "glsl", "inc"];

/// The pack's source texts: stage files, `.glsl` libraries, and files of other names that a
/// source includes (`shaders.settings`), as `crate::switches_on` reads them.
fn source_texts(pack: &Pack) -> Vec<String> {
    let files = pack.all_files();
    let source = |path: &str| SOURCES.contains(&path.rsplit_once('.').map_or("", |(_, e)| e));
    let mut texts: Vec<String> = files
        .iter()
        .filter(|p| source(p))
        .filter_map(|p| pack.read_text(p))
        .collect();
    let included: BTreeSet<String> = texts
        .iter()
        .flat_map(|t| t.lines())
        .filter_map(|l| l.trim().strip_prefix("#include"))
        .map(|r| r.trim().trim_matches(|c| c == '"' || c == '<' || c == '>'))
        .map(|r| r.rsplit('/').next().unwrap_or(r).to_owned())
        .collect();
    texts.extend(
        files
            .iter()
            .filter(|p| !source(p) && included.contains(p.rsplit('/').next().unwrap_or(p)))
            .filter_map(|p| pack.read_text(p)),
    );
    texts
}

/// The default value of each value option: `#define NAME VALUE // [a b c]`, a define whose
/// comment lists the allowed values (a define without the list is a constant, not an option).
/// A name given two different defaults is ambiguous and left out, as switches are.
pub fn value_options(pack: &Pack) -> BTreeMap<String, String> {
    let mut values: BTreeMap<String, Option<String>> = BTreeMap::new();
    for text in source_texts(pack) {
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix('#') else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix("define") else {
                continue;
            };
            if !rest.starts_with(char::is_whitespace) {
                continue;
            }
            let (code, comment) = rest.split_once("//").unwrap_or((rest, ""));
            if !comment.trim_start().starts_with('[') {
                continue;
            }
            let mut words = code.split_whitespace();
            let (Some(name), Some(value), None) = (words.next(), words.next(), words.next()) else {
                continue;
            };
            if name.contains('(') {
                continue;
            }
            match values.get_mut(name) {
                Some(seen) if seen.as_deref() != Some(value) => *seen = None,
                Some(_) => {}
                None => _ = values.insert(name.to_owned(), Some(value.to_owned())),
            }
        }
    }
    values
        .into_iter()
        .filter_map(|(k, v)| Some((k, v?)))
        .collect()
}

/// `base` with the pack's options at their defaults: what Iris evaluates a pack's
/// `.properties` files with (switches on by default defined, value options with their value).
/// Programs read their options from their own sources, so this is for `.properties` only.
pub fn properties_macros(pack: &Pack, base: &Macros) -> Macros {
    let mut m = base.clone();
    for name in crate::switches_on(pack) {
        m.define(name, "");
    }
    for (name, value) in value_options(pack) {
        m.define(name, value);
    }
    m
}

/// `bufferObject.<index> = <bytes>` or `= <bytes per pixel> true <scale x> <scale y>`: a
/// storage buffer every program sees at binding `index` (read and written by them; zeroed once).
#[derive(Clone, Debug, PartialEq)]
pub struct BufferDecl {
    pub index: u32,
    pub size: u64,
    /// Sized per pixel of the screen scaled by these factors (`size` bytes per pixel).
    pub relative: Option<[f32; 2]>,
}

/// How big an image is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageSize {
    /// The screen's size scaled.
    Relative([f32; 2]),
    /// 1, 2 or 3 dimensions.
    Fixed { dimensions: u8, size: [u32; 3] },
}

/// `image.<name> = <sampler|none> <format> <internal format> <pixel type> <clear> <relative>
/// <size...>`: an image programs load and store as `name`, and sample as `sampler`.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageDecl {
    pub name: String,
    pub sampler: Option<String>,
    /// The internal format as GL names it (`RGBA16F`, `R32UI`...).
    pub internal: String,
    /// Cleared to zero at the start of every frame.
    pub clear: bool,
    pub size: ImageSize,
}

/// A raw texture's description (`<path> <type> <internal> <w> [h] [d] <format> <pixel type>`).
#[derive(Clone, Debug, PartialEq)]
pub struct RawTexture {
    pub dimensions: u8,
    pub internal: String,
    pub size: [u32; 3],
    pub pixel_type: String,
}

/// A texture from the pack's files bound to a sampler name: `customTexture.<name>` (every
/// program), `texture.noise` (`noisetex`), or `texture.<stage>.<sampler>` (that stage's programs).
#[derive(Clone, Debug, PartialEq)]
pub struct TextureDecl {
    pub sampler: String,
    /// `None` for every stage; else `gbuffers`, `shadow`, `shadowcomp`, `prepare`, `deferred`,
    /// `composite`, `setup` or `begin`.
    pub stage: Option<String>,
    /// Relative to the pack's `shaders` folder.
    pub path: String,
    pub raw: Option<RawTexture>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Resources {
    pub buffers: Vec<BufferDecl>,
    pub images: Vec<ImageDecl>,
    pub textures: Vec<TextureDecl>,
    pub notes: Vec<String>,
}

const STAGES: [&str; 8] = [
    "gbuffers",
    "shadow",
    "shadowcomp",
    "prepare",
    "deferred",
    "composite",
    "setup",
    "begin",
];

fn texture_path(value: &str) -> String {
    value.trim().trim_start_matches('/').to_owned()
}

/// A raw texture's words after its path: `TEXTURE_3D RGBA8 256 256 256 RGBA UNSIGNED_BYTE`.
fn raw_texture(words: &[&str]) -> Option<RawTexture> {
    let (kind, rest) = words.split_first()?;
    let dimensions = match *kind {
        "TEXTURE_1D" => 1,
        "TEXTURE_2D" | "TEXTURE_RECTANGLE" => 2,
        "TEXTURE_3D" => 3,
        _ => return None,
    };
    let (internal, rest) = rest.split_first()?;
    let n = dimensions as usize;
    let mut size = [1u32; 3];
    for (k, s) in rest.get(..n)?.iter().enumerate() {
        size[k] = s.parse().ok()?;
    }
    let pixel_type = rest.get(n + 1)?.to_string();
    Some(RawTexture {
        dimensions,
        internal: internal.to_string(),
        size,
        pixel_type,
    })
}

impl Resources {
    /// The declarations of `shaders.properties`, already read with its conditionals.
    pub fn parse(properties: &Properties) -> Self {
        let mut r = Self::default();
        for e in &properties.entries {
            let words: Vec<&str> = e.value.split_whitespace().collect();
            if let Some(index) = e.key.strip_prefix("bufferObject.") {
                let Ok(index) = index.trim().parse::<u32>() else {
                    r.notes.push(format!("{}: not a buffer index", e.key));
                    continue;
                };
                let size = words.first().and_then(|w| w.parse::<u64>().ok());
                let Some(size) = size else {
                    r.notes.push(format!("{}: no size", e.key));
                    continue;
                };
                let relative = match words.get(1).copied() {
                    Some("true") => {
                        let x = words.get(2).and_then(|w| w.parse().ok()).unwrap_or(1.0);
                        let y = words.get(3).and_then(|w| w.parse().ok()).unwrap_or(1.0);
                        Some([x, y])
                    }
                    _ => None,
                };
                r.buffers.push(BufferDecl {
                    index,
                    size,
                    relative,
                });
            } else if let Some(name) = e.key.strip_prefix("image.") {
                // sampler format internal pixel-type clear relative size...
                if words.len() < 7 {
                    r.notes.push(format!("{}: too few words", e.key));
                    continue;
                }
                let sampler = (words[0] != "none").then(|| words[0].to_owned());
                let clear = words[4] == "true";
                let size = if words[5] == "true" {
                    let x = words[6].parse().unwrap_or(1.0);
                    let y = words.get(7).and_then(|w| w.parse().ok()).unwrap_or(x);
                    ImageSize::Relative([x, y])
                } else {
                    let dims: Vec<u32> = words[6..].iter().filter_map(|w| w.parse().ok()).collect();
                    if dims.is_empty() || dims.len() > 3 {
                        r.notes.push(format!("{}: no size", e.key));
                        continue;
                    }
                    let mut size = [1; 3];
                    size[..dims.len()].copy_from_slice(&dims);
                    ImageSize::Fixed {
                        dimensions: dims.len() as u8,
                        size,
                    }
                };
                r.images.retain(|i| i.name != name);
                r.images.push(ImageDecl {
                    name: name.to_owned(),
                    sampler,
                    internal: words[2].to_uppercase(),
                    clear,
                    size,
                });
            } else if let Some(name) = e.key.strip_prefix("customTexture.") {
                let Some((path, rest)) = words.split_first() else {
                    continue;
                };
                r.textures.push(TextureDecl {
                    sampler: name.to_owned(),
                    stage: None,
                    path: texture_path(path),
                    raw: raw_texture(rest),
                });
            } else if e.key == "texture.noise" {
                r.textures.push(TextureDecl {
                    sampler: "noisetex".into(),
                    stage: None,
                    path: texture_path(&e.value),
                    raw: None,
                });
            } else if let Some(rest) = e.key.strip_prefix("texture.")
                && let Some((stage, sampler)) = rest.split_once('.')
                && STAGES.contains(&stage)
            {
                let Some((path, raw)) = words.split_first() else {
                    continue;
                };
                if path.contains(':') {
                    r.notes.push(format!(
                        "{}: resource-pack textures are not provided",
                        e.key
                    ));
                    continue;
                }
                // A sampler may be named by its buffer number (`texture.composite.5`)...
                let sampler = match sampler.parse::<u32>() {
                    Ok(n) => format!("colortex{n}"),
                    Err(_) => sampler.to_owned(),
                };
                r.textures.push(TextureDecl {
                    sampler,
                    stage: Some(stage.to_owned()),
                    path: texture_path(path),
                    raw: raw_texture(raw),
                });
            }
        }
        r
    }

    /// The image a program reaches by `name`, as an image or through its sampler.
    pub fn image(&self, name: &str) -> Option<(usize, &ImageDecl)> {
        self.images.iter().enumerate().find(|(_, i)| i.name == name)
    }

    pub fn image_by_sampler(&self, name: &str) -> Option<(usize, &ImageDecl)> {
        self.images
            .iter()
            .enumerate()
            .find(|(_, i)| i.sampler.as_deref() == Some(name))
    }

    /// The texture sampler `name` reads in a program of `stage`: a stage's own first.
    pub fn texture(&self, name: &str, stage: &str) -> Option<&TextureDecl> {
        let own = self
            .textures
            .iter()
            .find(|t| t.sampler == name && t.stage.as_deref() == Some(stage));
        own.or_else(|| {
            self.textures
                .iter()
                .find(|t| t.sampler == name && t.stage.is_none())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::tests::zip;

    #[test]
    fn options_reach_properties() {
        let pack = Pack::from_zip_bytes(
            "P.zip",
            zip(&[
                (
                    "shaders/lib/settings.glsl",
                    "#define LIGHT\n//#define GRASS\n#define LEVEL 2 // [1 2 3]\n#define CONST 4\n\
                     #define SPEED 1.0 //[0.5 1.0]\n#define TWICE 1 // [1 2]\n",
                ),
                ("shaders/lib/other.glsl", "#define TWICE 2 // [1 2]\n"),
                (
                    "shaders/final.fsh",
                    "#include \"/lib/settings.glsl\"\n#ifdef LIGHT\n#endif\n#ifdef GRASS\n#endif\n",
                ),
            ]),
        )
        .unwrap();
        let values = value_options(&pack);
        assert_eq!(values.get("LEVEL").map(String::as_str), Some("2"));
        assert_eq!(values.get("SPEED").map(String::as_str), Some("1.0"));
        assert!(!values.contains_key("CONST") && !values.contains_key("TWICE"));
        let m = properties_macros(&pack, &Macros::standard());
        let p = Properties::parse(
            "#ifdef LIGHT\na=1\n#endif\n#ifdef GRASS\nb=1\n#endif\n#if LEVEL == 2\nc=1\n#endif\n",
            &m,
        );
        let keys: Vec<_> = p.entries.iter().map(|e| e.key.as_str()).collect();
        assert_eq!(keys, ["a", "c"]);
    }

    #[test]
    fn declarations_parse() {
        let text = "bufferObject.4 = 1024\nbufferObject.2 = 16 true 0.5 0.5\n\
            image.cloudDepth = cloudDepthSampler RGBA RGBA16F HALF_FLOAT true true 1.0 1.0\n\
            image.blocks = texBlocks RG_INTEGER RG32UI UNSIGNED_INT false false 2048\n\
            image.voxels = none RED_INTEGER R32UI UNSIGNED_INT true false 256 256 256\n\
            customTexture.moon = texture/moon.png\n\
            customTexture.lut = tex/lut.dat TEXTURE_3D RGBA8 32 32 32 RGBA UNSIGNED_BYTE\n\
            texture.noise = texture/noises.png\n\
            texture.composite.colortex6 = texture/blue.png\n\
            texture.deferred.5 = /texture/x.png\n";
        let r = Resources::parse(&Properties::parse(text, &Macros::standard()));
        assert!(r.notes.is_empty(), "{:?}", r.notes);
        assert_eq!(
            r.buffers,
            [
                BufferDecl { index: 4, size: 1024, relative: None },
                BufferDecl { index: 2, size: 16, relative: Some([0.5, 0.5]) },
            ]
        );
        assert_eq!(r.images.len(), 3);
        assert_eq!(r.images[0].sampler.as_deref(), Some("cloudDepthSampler"));
        assert_eq!(r.images[0].size, ImageSize::Relative([1.0, 1.0]));
        assert_eq!(
            r.images[1].size,
            ImageSize::Fixed { dimensions: 1, size: [2048, 1, 1] }
        );
        assert_eq!(r.images[2].sampler, None);
        assert!(r.images[2].clear);
        assert_eq!(
            r.images[2].size,
            ImageSize::Fixed { dimensions: 3, size: [256, 256, 256] }
        );
        assert_eq!(r.texture("moon", "composite").unwrap().path, "texture/moon.png");
        let lut = r.texture("lut", "gbuffers").unwrap().raw.clone().unwrap();
        assert_eq!((lut.dimensions, lut.size), (3, [32, 32, 32]));
        assert_eq!(r.texture("noisetex", "gbuffers").unwrap().path, "texture/noises.png");
        assert_eq!(r.texture("colortex6", "composite").unwrap().path, "texture/blue.png");
        assert!(r.texture("colortex6", "deferred").is_none());
        assert_eq!(r.texture("colortex5", "deferred").unwrap().path, "texture/x.png");
    }
}
