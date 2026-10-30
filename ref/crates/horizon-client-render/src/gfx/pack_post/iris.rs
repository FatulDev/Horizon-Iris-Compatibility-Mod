// Iris Features: what a shader pack binds and runs beyond Horizon's own pipeline — storage
// buffers (`bufferObject.*`), images (`image.*`, loaded and stored as images, read through
// their sampler name), custom textures (`customTexture.*`, `texture.noise`,
// `texture.<stage>.<sampler>`), samplers of other dimensions, and compute programs
// (`setup`, `begin`, `shadowcomp`, `prepare`, `deferred` and `composite` `.csh`).
//
// Bindings, all in set 0 beside Horizon's: samplers where the translator put them, storage
// buffers at 1000 + the pack's index, images from 2000 (`horizon_shaders::Interface`).

use std::cell::RefCell;
use std::collections::HashMap;

use horizon_shaders::iris::{ImageDecl, ImageSize, Resources, TextureDecl};
use horizon_shaders::{Group, Interface, WorkGroups};

use super::{BUFFERS, Binding, PackPost, SamplerKind, Source, checked, entry, source};

pub(super) const BUFFER_BINDING: u32 = 1000;

/// What a sampler reads when Horizon's own resources don't have it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Extra {
    /// A custom texture (`Iris::textures`).
    Custom(usize),
    /// An image, through its sampler name (`Iris::images`).
    Image(usize),
    /// A blank texture of this dimension and sample type (a sampler nothing provides).
    Blank(wgpu::TextureViewDimension, Blank),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Blank {
    Float,
    Uint,
    Sint,
    /// A flat normal (127, 127, 255, 255): what `normals` reads without a PBR resource pack.
    FlatNormal,
}

/// What one pass binds of ours.
#[derive(Clone, Default)]
pub(super) struct PassIris {
    /// Samplers reading something of ours, by their texture's binding.
    pub textures: Vec<(u32, Extra)>,
    /// (binding, image) of the images it loads or stores.
    pub images: Vec<(u32, usize)>,
    /// The pack indices of the storage buffers it uses.
    pub buffers: Vec<u32>,
}

pub(super) struct IrisImage {
    pub decl: ImageDecl,
    pub format: wgpu::TextureFormat,
    pub dimension: wgpu::TextureDimension,
    pub view_dimension: wgpu::TextureViewDimension,
    /// Its size, and the texture once made (images relative to the screen wait for its size).
    pub size: [u32; 3],
    pub texture: Option<(wgpu::Texture, wgpu::TextureView)>,
}

pub(super) struct IrisBuffer {
    pub index: u32,
    pub size: u64,
    pub relative: Option<[f32; 2]>,
    pub buffer: Option<wgpu::Buffer>,
}

pub(super) struct CustomTexture {
    pub decl: TextureDecl,
    pub view: wgpu::TextureView,
    pub view_dimension: wgpu::TextureViewDimension,
}

/// A compute program ready to dispatch.
pub(super) struct ComputeRun {
    pub name: String,
    /// `begin` 0, `shadowcomp` 1, `prepare` 2, `deferred` 3, `composite` 4; `setup` runs once.
    pub rank: u8,
    pub setup: bool,
    pub number: u32,
    pub pipeline: wgpu::ComputePipeline,
    pub layout: wgpu::BindGroupLayout,
    pub ubo: wgpu::Buffer,
    pub block: Vec<u8>,
    pub uniforms: Vec<(Option<usize>, String, u32)>,
    pub samplers: Vec<Binding>,
    pub iris: PassIris,
    pub reads: u16,
    pub shadow_reads: u16,
    pub work: WorkGroups,
    pub local: [u32; 3],
    pub groups: HashMap<u64, wgpu::BindGroup>,
}

pub(super) struct Iris {
    pub resources: Resources,
    pub images: Vec<IrisImage>,
    pub buffers: Vec<IrisBuffer>,
    pub textures: Vec<CustomTexture>,
    pub computes: Vec<ComputeRun>,
    pub blanks: HashMap<(wgpu::TextureViewDimension, Blank), wgpu::TextureView>,
    pub setup_done: bool,
    /// The render size the screen-relative resources were made for.
    pub size: [u32; 2],
    /// Which computes ran this frame, by index.
    pub ran: Vec<bool>,
    pub can_clear: bool,
}

/// What `build_pass` needs to know of ours while `PackPost::new` builds its passes (its
/// signature is Horizon's, so this travels beside it).
pub(super) struct BuildContext {
    pub images: Vec<(String, Option<String>, wgpu::TextureFormat, wgpu::TextureViewDimension)>,
    pub textures: Vec<(TextureDecl, wgpu::TextureViewDimension)>,
    pub features: wgpu::Features,
}

thread_local! {
    static BUILD: RefCell<Option<BuildContext>> = const { RefCell::new(None) };
}

pub(super) fn set_build(context: Option<BuildContext>) {
    BUILD.with(|b| *b.borrow_mut() = context);
}

fn with_build<T>(f: impl FnOnce(&BuildContext) -> T) -> Option<T> {
    BUILD.with(|b| b.borrow().as_ref().map(f))
}

/// The custom-texture stage of a program group (`texture.<stage>.<sampler>`).
pub(super) fn stage_of(group: Group) -> &'static str {
    match group {
        Group::Gbuffers | Group::Translucent => "gbuffers",
        Group::Shadow => "shadow",
        Group::ShadowComp => "shadowcomp",
        Group::Prepare => "prepare",
        Group::Deferred => "deferred",
        Group::Composite | Group::Final => "composite",
    }
}

/// A sampler type's view dimension and integer kind (`usampler3D` → D3, Uint).
fn sampler_shape(ty: &str) -> (wgpu::TextureViewDimension, Blank) {
    use wgpu::TextureViewDimension as D;
    let dimension = if ty.contains("Cube") {
        D::Cube
    } else if ty.contains("2DArray") {
        D::D2Array
    } else if ty.contains("1D") {
        D::D1
    } else if ty.contains("3D") {
        D::D3
    } else {
        D::D2
    };
    let blank = if ty.starts_with('u') {
        Blank::Uint
    } else if ty.starts_with('i') {
        Blank::Sint
    } else {
        Blank::Float
    };
    (dimension, blank)
}

/// The wgpu format of an image's internal format (`image.*`'s third word).
pub(super) fn internal_format(gl: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as T;
    Some(match gl {
        "R8" => T::R8Unorm,
        "RG8" => T::Rg8Unorm,
        "RGB8" | "RGBA8" => T::Rgba8Unorm,
        "R8_SNORM" => T::R8Snorm,
        "RGBA8_SNORM" => T::Rgba8Snorm,
        "R16" => T::R16Unorm,
        "RG16" => T::Rg16Unorm,
        "RGB16" | "RGBA16" => T::Rgba16Unorm,
        "R16F" => T::R16Float,
        "RG16F" => T::Rg16Float,
        "RGB16F" | "RGBA16F" => T::Rgba16Float,
        "R32F" => T::R32Float,
        "RG32F" => T::Rg32Float,
        "RGB32F" | "RGBA32F" => T::Rgba32Float,
        "R11F_G11F_B10F" => T::Rg11b10Ufloat,
        "RGB10_A2" => T::Rgb10a2Unorm,
        "R8I" => T::R8Sint,
        "RG8I" => T::Rg8Sint,
        "RGBA8I" => T::Rgba8Sint,
        "R16I" => T::R16Sint,
        "RG16I" => T::Rg16Sint,
        "RGBA16I" => T::Rgba16Sint,
        "R32I" => T::R32Sint,
        "RG32I" => T::Rg32Sint,
        "RGBA32I" => T::Rgba32Sint,
        "R8UI" => T::R8Uint,
        "RG8UI" => T::Rg8Uint,
        "RGBA8UI" => T::Rgba8Uint,
        "R16UI" => T::R16Uint,
        "RG16UI" => T::Rg16Uint,
        "RGBA16UI" => T::Rgba16Uint,
        "R32UI" => T::R32Uint,
        "RG32UI" => T::Rg32Uint,
        "RGBA32UI" => T::Rgba32Uint,
        _ => return None,
    })
}

/// The wgpu format of an image's GLSL format qualifier (`layout(rgba16f)`).
pub(super) fn qualifier_format(q: &str) -> Option<wgpu::TextureFormat> {
    let gl = match q {
        "rgba8" => "RGBA8",
        "rgba8_snorm" => "RGBA8_SNORM",
        "r11f_g11f_b10f" => "R11F_G11F_B10F",
        "rgb10_a2" => "RGB10_A2",
        q => return internal_format(&q.to_uppercase()),
    };
    internal_format(gl)
}

/// The sample type a pass reads `format` with: float formats are filterable when the device
/// allows it, integer formats are read as such.
fn sample_type(format: wgpu::TextureFormat, features: wgpu::Features) -> wgpu::TextureSampleType {
    format
        .sample_type(None, Some(features))
        .unwrap_or(wgpu::TextureSampleType::Float { filterable: false })
}

/// The samplers of a program, resolved: Horizon's sources and ours, the layout entries of
/// their textures and of the sampler objects they share (`build_pass`'s loop, shared with the
/// compute programs).
pub(super) struct Samplers {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
    pub samplers: Vec<Binding>,
    pub slots: HashMap<u32, u32>,
    pub reads: u16,
    pub shadow_reads: u16,
    pub extras: Vec<(u32, Extra)>,
}

/// One of `build_pass`'s samplers: what it reads, of ours or Horizon's.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_samplers(
    name: &str,
    group: Group,
    interface: &Interface,
    formats: &[wgpu::TextureFormat; BUFFERS],
    shadow: Option<&[wgpu::TextureFormat; 2]>,
    features: wgpu::Features,
    in_place: u16,
    gbuffers: bool,
    visibility: wgpu::ShaderStages,
    notes: &mut Vec<String>,
) -> Samplers {
    let mut out = Samplers {
        entries: Vec::new(),
        samplers: Vec::new(),
        slots: HashMap::new(),
        reads: 0,
        shadow_reads: 0,
        extras: Vec::new(),
    };
    let mut kinds: HashMap<SamplerKind, u32> = HashMap::new();
    let watershadow = interface.samplers.iter().any(|s| s.name == "watershadow");
    let stage = stage_of(group);
    let float = wgpu::TextureSampleType::Float { filterable: true };
    let texture_entry = |binding: u32, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    };
    for s in &interface.samplers {
        let (shape, kind_of) = sampler_shape(&s.ty);
        // Ours first: a custom texture for this stage, an image's sampler, else a blank of any
        // shape Horizon doesn't provide. As in Iris, the pack's buffers claim their names
        // before custom textures do: `texture.composite.colortex6` doesn't hide colortex6 (only
        // `noisetex` is meant to be replaced).
        let horizon_has = s.name != "noisetex"
            && source(&s.name, group.shadow(), shadow.is_some(), watershadow) != Source::Dummy;
        let ours = with_build(|b| {
            if horizon_has {
                return None;
            }
            if let Some(k) = b
                .textures
                .iter()
                .position(|(t, _)| t.sampler == s.name && t.stage.as_deref() == Some(stage))
                .or_else(|| {
                    b.textures
                        .iter()
                        .position(|(t, _)| t.sampler == s.name && t.stage.is_none())
                })
            {
                let (_, d) = &b.textures[k];
                return Some((Extra::Custom(k), *d, float));
            }
            if let Some(k) = b.images.iter().position(|i| i.1.as_deref() == Some(&s.name)) {
                let (_, _, format, d) = &b.images[k];
                return Some((Extra::Image(k), *d, sample_type(*format, b.features)));
            }
            None
        })
        .flatten();
        let ours = ours.or_else(|| {
            // Without a PBR resource pack, Iris gives `normals` a flat normal and `specular`
            // zeros; Horizon gives neither.
            match s.name.as_str() {
                "normals" if shape == wgpu::TextureViewDimension::D2 => {
                    return Some((Extra::Blank(shape, Blank::FlatNormal), shape, float));
                }
                "specular" if shape == wgpu::TextureViewDimension::D2 => {
                    return Some((Extra::Blank(shape, Blank::Float), shape, float));
                }
                _ => {}
            }
            let horizon = source(&s.name, group.shadow(), shadow.is_some(), watershadow);
            // A 2D float sampler Horizon doesn't know reads its white stand-in, as before.
            let provided = shape == wgpu::TextureViewDimension::D2
                && (horizon != Source::Dummy
                    || s.ty.ends_with("Shadow")
                    || kind_of == Blank::Float);
            (!provided).then(|| {
                let sample = match kind_of {
                    Blank::Float | Blank::FlatNormal => float,
                    Blank::Uint => wgpu::TextureSampleType::Uint,
                    Blank::Sint => wgpu::TextureSampleType::Sint,
                };
                if horizon == Source::Dummy || shape != wgpu::TextureViewDimension::D2 {
                    notes.push(format!(
                        "{name}: `{} {}` reads a blank texture (nothing provides it)",
                        s.ty, s.name
                    ));
                }
                (Extra::Blank(shape, kind_of), shape, sample)
            })
        });
        let (src, sample_type, dimension) = match ours {
            Some((extra, dimension, sample_type)) => {
                out.extras.push((s.binding, extra));
                // A filterable custom texture repeats (noise...); the rest clamp, nearest when
                // they can't be filtered.
                let src = match (extra, sample_type) {
                    (Extra::Custom(_), t) if t == float => Source::Noise,
                    _ => Source::Dummy,
                };
                (src, sample_type, dimension)
            }
            None => {
                let (src, sample_type) =
                    horizon_sampler(s, group, formats, shadow, features, in_place, gbuffers, watershadow, name, notes);
                (src, sample_type, wgpu::TextureViewDimension::D2)
            }
        };
        let filtering = sample_type == float;
        match src {
            Source::Color(n) => out.reads |= 1 << n,
            Source::ShadowColor(n) => out.shadow_reads |= 1 << n,
            _ => {}
        }
        out.entries
            .push(texture_entry(s.binding, sample_type, dimension));
        let kind = SamplerKind::of(src, filtering);
        let entries = &mut out.entries;
        let sampler = *kinds.entry(kind).or_insert_with(|| {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: s.binding + 1,
                visibility,
                ty: wgpu::BindingType::Sampler(kind.binding_type()),
                count: None,
            });
            s.binding + 1
        });
        out.slots.insert(s.binding + 1, sampler);
        out.samplers.push(Binding {
            source: src,
            binding: s.binding,
            filtering,
            sampler,
        });
    }
    out
}

/// Horizon's own resolution of a 2D sampler (`build_pass`'s, unchanged).
#[allow(clippy::too_many_arguments)]
fn horizon_sampler(
    s: &horizon_shaders::Sampler,
    group: Group,
    formats: &[wgpu::TextureFormat; BUFFERS],
    shadow: Option<&[wgpu::TextureFormat; 2]>,
    features: wgpu::Features,
    in_place: u16,
    gbuffers: bool,
    watershadow: bool,
    name: &str,
    notes: &mut Vec<String>,
) -> (Source, wgpu::TextureSampleType) {
    let mut src = source(&s.name, group.shadow(), shadow.is_some(), watershadow);
    let blank = match src {
        Source::ShadowDepth(..) => group == Group::Shadow,
        Source::Atlas => !gbuffers,
        Source::Depth => gbuffers,
        Source::Depth1 => group == Group::Gbuffers,
        Source::Color(n) => in_place & 1 << n != 0,
        _ => false,
    };
    let float = wgpu::TextureSampleType::Float { filterable: true };
    let mut sample_type = match src {
        Source::Color(n) => formats[n].sample_type(None, Some(features)).unwrap_or(float),
        Source::ShadowColor(n) => shadow
            .and_then(|f| f[n].sample_type(None, Some(features)))
            .unwrap_or(float),
        Source::Depth | Source::Depth1 | Source::ShadowDepth(..) => {
            wgpu::TextureSampleType::Float { filterable: false }
        }
        _ => float,
    };
    let int = !matches!(sample_type, wgpu::TextureSampleType::Float { .. });
    if s.ty == "sampler2DShadow" {
        src = match src {
            Source::ShadowDepth(n, _) => Source::ShadowDepth(n, true),
            _ => Source::Shadow,
        };
        sample_type = wgpu::TextureSampleType::Depth;
    }
    let wanted = match s.ty.as_str() {
        "sampler2DShadow" => true,
        "sampler2D" => !int,
        "isampler2D" => sample_type == wgpu::TextureSampleType::Sint,
        "usampler2D" => sample_type == wgpu::TextureSampleType::Uint,
        _ => false,
    };
    if !wanted {
        notes.push(format!("{name}: `{} {}` is not provided yet", s.ty, s.name));
        (src, sample_type) = (Source::Dummy, float);
    }
    if blank && sample_type == wgpu::TextureSampleType::Depth {
        src = Source::Shadow;
    } else if blank {
        src = Source::Dummy;
        if int {
            sample_type = float;
        }
    }
    (src, sample_type)
}

/// The stages of `module` that declare a global at set 0, `binding`, and whether one writes it.
fn uses(modules: &[(wgpu::ShaderStages, &naga::Module)], binding: u32) -> (wgpu::ShaderStages, bool) {
    let mut stages = wgpu::ShaderStages::empty();
    let mut writes = false;
    for (stage, m) in modules {
        for (_, g) in m.global_variables.iter() {
            let Some(b) = &g.binding else { continue };
            if b.group != 0 || b.binding != binding {
                continue;
            }
            stages |= *stage;
            match (&g.space, &m.types[g.ty].inner) {
                (naga::AddressSpace::Storage { access }, _) => {
                    writes |= access.contains(naga::StorageAccess::STORE)
                }
                (
                    _,
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { access, .. },
                        ..
                    },
                ) => writes |= access.contains(naga::StorageAccess::STORE),
                _ => {}
            }
        }
    }
    (stages, writes)
}

/// The layout entries of the images and storage buffers a program uses, and what it binds.
pub(super) fn storage_entries(
    name: &str,
    interface: &Interface,
    modules: &[(wgpu::ShaderStages, &naga::Module)],
    entries: &mut Vec<wgpu::BindGroupLayoutEntry>,
    notes: &mut Vec<String>,
) -> Option<PassIris> {
    let mut pass = PassIris::default();
    let features = with_build(|b| b.features).unwrap_or_default();
    for u in &interface.images {
        let Some((k, format, dimension)) = with_build(|b| {
            b.images
                .iter()
                .position(|i| i.0 == u.name)
                .map(|k| (k, b.images[k].2, b.images[k].3))
        })
        .flatten() else {
            notes.push(format!("{name}: image `{}` is not declared (`image.{}`); skipped", u.name, u.name));
            return None;
        };
        if let Some(q) = qualifier_format(&u.format)
            && q != format
        {
            notes.push(format!(
                "{name}: image `{}` is {:?} but the program reads it as {}",
                u.name, format, u.format
            ));
        }
        let (stages, writes) = uses(modules, u.binding);
        let access = match u.access.as_str() {
            "readonly" => wgpu::StorageTextureAccess::ReadOnly,
            "writeonly" => wgpu::StorageTextureAccess::WriteOnly,
            _ => wgpu::StorageTextureAccess::ReadWrite,
        };
        if writes
            && stages.contains(wgpu::ShaderStages::VERTEX)
            && !features.contains(wgpu::Features::VERTEX_WRITABLE_STORAGE)
        {
            notes.push(format!(
                "{name}: writes image `{}` from its vertex stage, which this GPU can't; skipped",
                u.name
            ));
            return None;
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: u.binding,
            visibility: if stages.is_empty() { wgpu::ShaderStages::all() } else { stages },
            ty: wgpu::BindingType::StorageTexture {
                access,
                format,
                view_dimension: dimension,
            },
            count: None,
        });
        pass.images.push((u.binding, k));
    }
    for &index in &interface.buffers {
        let binding = BUFFER_BINDING + index;
        let (stages, writes) = uses(modules, binding);
        if writes
            && stages.contains(wgpu::ShaderStages::VERTEX)
            && !features.contains(wgpu::Features::VERTEX_WRITABLE_STORAGE)
        {
            notes.push(format!(
                "{name}: writes storage buffer {index} from its vertex stage, which this GPU can't; skipped"
            ));
            return None;
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: if stages.is_empty() { wgpu::ShaderStages::all() } else { stages },
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: !writes },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        pass.buffers.push(index);
    }
    Some(pass)
}

/// Marks every storage global of `module` read-only when the module never writes it: a
/// read-only binding in a vertex stage needs no device feature.
pub(super) fn read_only_where_unwritten(module: &mut naga::Module) {
    let Ok(info) = naga::valid::Validator::new(
        naga::valid::ValidationFlags::empty(),
        naga::valid::Capabilities::all(),
    )
    .validate(module) else {
        return;
    };
    let mut written = vec![false; module.global_variables.len()];
    for (i, ep) in module.entry_points.iter().enumerate() {
        let f = info.get_entry_point(i);
        for (h, _) in module.global_variables.iter() {
            if f[h].contains(naga::valid::GlobalUse::WRITE) {
                written[h.index()] = true;
            }
        }
        let _ = ep;
    }
    for (h, g) in module.global_variables.iter_mut() {
        if written[h.index()] {
            continue;
        }
        if let naga::AddressSpace::Storage { access } = &mut g.space {
            *access = naga::StorageAccess::LOAD;
        }
    }
    let types: Vec<_> = module
        .global_variables
        .iter()
        .filter(|(h, _)| !written[h.index()])
        .map(|(_, g)| g.ty)
        .collect();
    for ty in types {
        if let naga::TypeInner::Image {
            dim,
            arrayed,
            class: naga::ImageClass::Storage { format, access },
        } = module.types[ty].inner.clone()
            && access.contains(naga::StorageAccess::STORE)
        {
            let new = naga::Type {
                name: module.types[ty].name.clone(),
                inner: naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class: naga::ImageClass::Storage {
                        format,
                        access: naga::StorageAccess::LOAD,
                    },
                },
            };
            let handle = module.types.insert(new, naga::Span::UNDEFINED);
            for (h, g) in module.global_variables.iter_mut() {
                if g.ty == ty && !written[h.index()] {
                    g.ty = handle;
                }
            }
        }
    }
}

fn blank_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dimension: wgpu::TextureViewDimension,
    blank: Blank,
) -> wgpu::TextureView {
    let format = match blank {
        Blank::Float | Blank::FlatNormal => wgpu::TextureFormat::Rgba8Unorm,
        Blank::Uint => wgpu::TextureFormat::Rgba8Uint,
        Blank::Sint => wgpu::TextureFormat::Rgba8Sint,
    };
    let (texture_dimension, layers) = match dimension {
        wgpu::TextureViewDimension::D1 => (wgpu::TextureDimension::D1, 1),
        wgpu::TextureViewDimension::D3 => (wgpu::TextureDimension::D3, 1),
        wgpu::TextureViewDimension::Cube | wgpu::TextureViewDimension::CubeArray => {
            (wgpu::TextureDimension::D2, 6)
        }
        _ => (wgpu::TextureDimension::D2, 1),
    };
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: layers,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Shader pack blank"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: texture_dimension,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texel: [u8; 4] = if blank == Blank::FlatNormal { [127, 127, 255, 255] } else { [0; 4] };
    let zeros: Vec<u8> = texel.repeat(layers as usize);
    queue.write_texture(
        texture.as_image_copy(),
        &zeros,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(dimension),
        ..Default::default()
    })
}

/// A PNG as RGBA8 texels: (width, height, texels).
fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0; reader.output_buffer_size().ok_or("too big")?];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width, info.height);
    let pixels = (w * h) as usize;
    let data = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => data.to_vec(),
        png::ColorType::Rgb => data
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => data
            .chunks_exact(2)
            .flat_map(|c| [c[0], c[0], c[0], c[1]])
            .collect(),
        png::ColorType::Grayscale => data.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return Err("indexed colour after expansion".into()),
    };
    if rgba.len() != pixels * 4 {
        return Err("unexpected texel count".into());
    }
    Ok((w, h, rgba))
}

fn custom_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    decl: &TextureDecl,
    bytes: &[u8],
) -> Result<(wgpu::TextureView, wgpu::TextureViewDimension), String> {
    let (dimension, view_dimension, size, texels, bytes_per_texel) = match &decl.raw {
        None => {
            let (w, h, texels) = decode_png(bytes)?;
            (
                wgpu::TextureDimension::D2,
                wgpu::TextureViewDimension::D2,
                [w, h, 1],
                texels,
                4,
            )
        }
        Some(raw) => {
            let per = match (raw.internal.as_str(), raw.pixel_type.as_str()) {
                ("RGBA8" | "RGBA", "UNSIGNED_BYTE") => 4,
                (i, t) => return Err(format!("raw {i} {t} is not supported")),
            };
            let n = (raw.size[0] * raw.size[1] * raw.size[2]) as usize * per;
            if bytes.len() < n {
                return Err(format!("{} bytes, {n} expected", bytes.len()));
            }
            let (d, v) = match raw.dimensions {
                1 => (wgpu::TextureDimension::D1, wgpu::TextureViewDimension::D1),
                3 => (wgpu::TextureDimension::D3, wgpu::TextureViewDimension::D3),
                _ => (wgpu::TextureDimension::D2, wgpu::TextureViewDimension::D2),
            };
            (d, v, raw.size, bytes[..n].to_vec(), per as u32)
        }
    };
    let extent = wgpu::Extent3d {
        width: size[0],
        height: size[1],
        depth_or_array_layers: size[2],
    };
    let texture = checked(device, || {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&decl.sampler),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    })?;
    queue.write_texture(
        texture.as_image_copy(),
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size[0] * bytes_per_texel),
            rows_per_image: Some(size[1]),
        },
        extent,
    );
    Ok((
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dimension),
            ..Default::default()
        }),
        view_dimension,
    ))
}

impl Iris {
    /// The pack's resources: custom textures loaded, fixed-size images and buffers made (those
    /// sized by the screen wait for `resize`).
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        post: &mut horizon_shaders::IrisPost,
        notes: &mut Vec<String>,
    ) -> Self {
        let features = device.features();
        let resources = std::mem::take(&mut post.resources);
        let mut textures = Vec::new();
        for decl in &resources.textures {
            let Some(bytes) = post.texture_files.get(&decl.path) else {
                continue;
            };
            match custom_texture(device, queue, decl, bytes) {
                Ok((view, view_dimension)) => textures.push(CustomTexture {
                    decl: decl.clone(),
                    view,
                    view_dimension,
                }),
                Err(e) => notes.push(format!("custom texture {}: {e}", decl.path)),
            }
        }
        let mut images = Vec::new();
        for decl in &resources.images {
            let Some(format) = internal_format(&decl.internal) else {
                notes.push(format!(
                    "image {}: internal format {} is not supported",
                    decl.name, decl.internal
                ));
                continue;
            };
            let (dimension, view_dimension, size) = match decl.size {
                ImageSize::Relative(_) => (
                    wgpu::TextureDimension::D2,
                    wgpu::TextureViewDimension::D2,
                    [0; 3],
                ),
                ImageSize::Fixed { dimensions, size } => match dimensions {
                    1 => (wgpu::TextureDimension::D1, wgpu::TextureViewDimension::D1, size),
                    3 => (wgpu::TextureDimension::D3, wgpu::TextureViewDimension::D3, size),
                    _ => (wgpu::TextureDimension::D2, wgpu::TextureViewDimension::D2, size),
                },
            };
            let mut image = IrisImage {
                decl: decl.clone(),
                format,
                dimension,
                view_dimension,
                size,
                texture: None,
            };
            if size[0] > 0 {
                image.make(device, notes);
            }
            images.push(image);
        }
        let buffers = resources
            .buffers
            .iter()
            .map(|b| {
                let mut buffer = IrisBuffer {
                    index: b.index,
                    size: b.size,
                    relative: b.relative,
                    buffer: None,
                };
                if b.relative.is_none() {
                    buffer.make(device, b.size);
                }
                buffer
            })
            .collect();
        let can_clear = features.contains(wgpu::Features::CLEAR_TEXTURE);
        if !can_clear && resources.images.iter().any(|i| i.clear) {
            notes.push("images are not cleared each frame (no CLEAR_TEXTURE on this GPU)".into());
        }
        Self {
            resources,
            images,
            buffers,
            textures,
            computes: Vec::new(),
            blanks: HashMap::new(),
            setup_done: false,
            size: [0; 2],
            ran: Vec::new(),
            can_clear,
        }
    }

    /// What `build_pass` needs of ours, while the passes are built.
    pub(super) fn context(&self, features: wgpu::Features) -> BuildContext {
        BuildContext {
            images: self
                .images
                .iter()
                .map(|i| {
                    (
                        i.decl.name.clone(),
                        i.decl.sampler.clone(),
                        i.format,
                        i.view_dimension,
                    )
                })
                .collect(),
            textures: self
                .textures
                .iter()
                .map(|t| (t.decl.clone(), t.view_dimension))
                .collect(),
            features,
        }
    }

    /// Blanks for every shape a pass reads one of.
    pub(super) fn make_blanks<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        extras: impl Iterator<Item = &'a (u32, Extra)>,
    ) {
        for (_, e) in extras {
            if let Extra::Blank(d, b) = e {
                self.blanks
                    .entry((*d, *b))
                    .or_insert_with(|| blank_texture(device, queue, *d, *b));
            }
        }
    }

    /// Screen-relative images and buffers for a render size of `size`; true when any changed
    /// (bind groups must be made again).
    pub(super) fn resize(&mut self, device: &wgpu::Device, size: [u32; 2], notes: &mut Vec<String>) -> bool {
        if self.size == size {
            return false;
        }
        self.size = size;
        for image in &mut self.images {
            if let ImageSize::Relative([x, y]) = image.decl.size {
                image.size = [
                    ((size[0] as f32 * x).ceil() as u32).max(1),
                    ((size[1] as f32 * y).ceil() as u32).max(1),
                    1,
                ];
                image.make(device, notes);
            }
        }
        for buffer in &mut self.buffers {
            if let Some([x, y]) = buffer.relative {
                let pixels = (size[0] as f32 * x).ceil() as u64 * (size[1] as f32 * y).ceil() as u64;
                buffer.make(device, buffer.size * pixels.max(1));
            }
        }
        true
    }

    pub(super) fn image_view(&self, k: usize) -> Option<&wgpu::TextureView> {
        self.images.get(k)?.texture.as_ref().map(|t| &t.1)
    }

    pub(super) fn buffer(&self, index: u32) -> Option<&wgpu::Buffer> {
        self.buffers
            .iter()
            .find(|b| b.index == index)
            .and_then(|b| b.buffer.as_ref())
    }

    pub(super) fn extra_view(&self, extra: Extra) -> Option<&wgpu::TextureView> {
        match extra {
            Extra::Custom(k) => self.textures.get(k).map(|t| &t.view),
            Extra::Image(k) => self.image_view(k),
            Extra::Blank(d, b) => self.blanks.get(&(d, b)),
        }
    }

    /// Clears the images that ask for it, at the start of a frame.
    pub(super) fn clear_images(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.can_clear {
            return;
        }
        for image in &self.images {
            if let (true, Some((texture, _))) = (image.decl.clear, &image.texture) {
                encoder.clear_texture(texture, &wgpu::ImageSubresourceRange::default());
            }
        }
    }
}

impl IrisImage {
    fn make(&mut self, device: &wgpu::Device, notes: &mut Vec<String>) {
        let usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let made = checked(device, || {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&self.decl.name),
                size: wgpu::Extent3d {
                    width: self.size[0],
                    height: self.size[1],
                    depth_or_array_layers: self.size[2],
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: self.dimension,
                format: self.format,
                usage,
                view_formats: &[],
            })
        });
        match made {
            Ok(texture) => {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(self.view_dimension),
                    ..Default::default()
                });
                self.texture = Some((texture, view));
            }
            Err(e) => {
                notes.push(format!("image {}: {e}", self.decl.name));
                self.texture = None;
            }
        }
    }
}

impl IrisBuffer {
    fn make(&mut self, device: &wgpu::Device, size: u64) {
        // Storage bindings are multiples of 4 bytes; wgpu zeroes new buffers.
        let size = size.max(16).next_multiple_of(16);
        self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shader pack buffer object"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }
}

/// The rank of a stage among those a compute program runs in.
fn rank(stage: &str) -> u8 {
    match stage {
        "begin" | "setup" => 0,
        "shadowcomp" => 1,
        "prepare" => 2,
        "deferred" => 3,
        _ => 4,
    }
}

/// The rank of a program group, as `rank` (final after every composite).
pub(super) fn group_rank(group: Group) -> u8 {
    match group {
        Group::ShadowComp => 1,
        Group::Prepare => 2,
        Group::Deferred => 3,
        Group::Composite => 4,
        Group::Final => 5,
        _ => 0,
    }
}

/// The number of a program by its name (`composite12` → 12, `deferred` → 0).
pub(super) fn number_of(name: &str) -> u32 {
    let digits: String = name
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().unwrap_or(0)
}

/// Builds a compute program's pipeline; `None` (with a note) when wgpu refuses it.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_compute(
    device: &wgpu::Device,
    c: horizon_shaders::ComputePass,
    formats: &[wgpu::TextureFormat; BUFFERS],
    shadow: Option<&[wgpu::TextureFormat; 2]>,
    custom: &[String],
    notes: &mut Vec<String>,
) -> Option<ComputeRun> {
    let features = device.features();
    let interface = &c.program.interface;
    let group = match c.stage {
        "shadowcomp" => Group::ShadowComp,
        "prepare" => Group::Prepare,
        "deferred" => Group::Deferred,
        _ => Group::Composite,
    };
    let compute = wgpu::ShaderStages::COMPUTE;
    let mut s = resolve_samplers(
        &c.name, group, interface, formats, shadow, features, 0, false, compute, notes,
    );
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: compute,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    entries.append(&mut s.entries);
    let mut module = c.program.module;
    let modules = [(compute, &module)];
    let mut iris = storage_entries(&c.name, interface, &modules, &mut entries, notes)?;
    iris.textures = s.extras;
    super::share_samplers(&mut module, &s.slots);
    let created = checked(device, || {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&c.name),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&c.name),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&c.name),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&c.name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (layout, pipeline)
    });
    let (layout, pipeline) = match created {
        Ok(created) => created,
        Err(e) => {
            notes.push(format!("{}: skipped: {e}", c.name));
            return None;
        }
    };
    let index = |name: &str| {
        super::pack_uniforms::index(name)
            .or_else(|| {
                let k = custom.iter().position(|n| n == name)?;
                Some(super::pack_uniforms::NAMES.len() + k)
            })
            .or_else(|| {
                let k = super::pack_uniforms::iris_index(name)?;
                Some(super::pack_uniforms::NAMES.len() + custom.len() + k)
            })
    };
    Some(ComputeRun {
        rank: rank(c.stage),
        setup: c.stage == "setup",
        number: c.number,
        pipeline,
        layout,
        ubo: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&c.name),
            size: u64::from(interface.uniform_size.max(16)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        block: vec![0; interface.uniform_size as usize],
        uniforms: interface
            .uniforms
            .iter()
            .map(|u| (index(&u.name), u.ty.clone(), u.offset))
            .collect(),
        samplers: s.samplers,
        iris,
        reads: s.reads,
        shadow_reads: s.shadow_reads,
        work: c.program.work,
        local: c.program.local_size,
        groups: HashMap::new(),
        name: c.name,
    })
}

impl PackPost {
    /// The bind group entries of our resources a pass uses (images, buffers).
    pub(super) fn iris_entries<'a>(&'a self, pass: &PassIris, entries: &mut Vec<wgpu::BindGroupEntry<'a>>) {
        let Some(iris) = &self.iris else { return };
        for &(binding, k) in &pass.images {
            if let Some(view) = iris.image_view(k) {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
        }
        for &index in &pass.buffers {
            if let Some(buffer) = iris.buffer(index) {
                entries.push(wgpu::BindGroupEntry {
                    binding: BUFFER_BINDING + index,
                    resource: buffer.as_entire_binding(),
                });
            }
        }
    }

    /// What a sampler of ours reads, when it is one.
    pub(super) fn iris_view(&self, extras: &[(u32, Extra)], binding: u32) -> Option<&wgpu::TextureView> {
        let iris = self.iris.as_ref()?;
        let extra = extras.iter().find(|e| e.0 == binding)?.1;
        iris.extra_view(extra)
    }

    /// Runs the compute programs of ranks below `rank`, and of `rank` with a number up to
    /// `number`, that have not run this frame, in order.
    pub(super) fn run_computes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
        rank: u8,
        number: u32,
    ) {
        let Some(mut iris) = self.iris.take() else { return };
        let mut computes = std::mem::take(&mut iris.computes);
        iris.ran.resize(computes.len(), false);
        self.iris = Some(iris);
        for (k, c) in computes.iter_mut().enumerate() {
            let iris = self.iris.as_ref().expect("iris");
            if iris.ran[k] || c.setup {
                continue;
            }
            if c.rank > rank || (c.rank == rank && c.number > number) {
                continue;
            }
            self.dispatch(device, queue, encoder, c, size);
            self.iris.as_mut().expect("iris").ran[k] = true;
        }
        self.iris.as_mut().expect("iris").computes = computes;
    }

    /// The `setup` programs, once, before the first frame.
    pub(super) fn run_setup(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        size: [u32; 2],
    ) {
        let Some(mut iris) = self.iris.take() else { return };
        if iris.setup_done {
            self.iris = Some(iris);
            return;
        }
        iris.setup_done = true;
        let mut computes = std::mem::take(&mut iris.computes);
        self.iris = Some(iris);
        for c in computes.iter_mut().filter(|c| c.setup) {
            self.dispatch(device, queue, encoder, c, size);
        }
        self.iris.as_mut().expect("iris").computes = computes;
    }

    /// The start of a frame: images cleared, nothing run yet.
    pub(super) fn iris_begin_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(iris) = &mut self.iris {
            iris.clear_images(encoder);
            iris.ran.iter_mut().for_each(|r| *r = false);
        }
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        c: &mut ComputeRun,
        size: [u32; 2],
    ) {
        let group = self.bind_compute(device, queue, c, size);
        let groups = match c.work {
            WorkGroups::Fixed(g) => g,
            WorkGroups::Render([x, y]) => [
                ((size[0] as f32 * x).ceil() as u32).div_ceil(c.local[0].max(1)),
                ((size[1] as f32 * y).ceil() as u32).div_ceil(c.local[1].max(1)),
                1,
            ],
        };
        if groups.contains(&0) {
            return;
        }
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&c.name),
            timestamp_writes: None,
        });
        cp.set_pipeline(&c.pipeline);
        cp.set_bind_group(0, &group, &[]);
        cp.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }

    /// A compute program's uniforms written and its bind group for the buffers' mapping now.
    fn bind_compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        c: &mut ComputeRun,
        out: [u32; 2],
    ) -> wgpu::BindGroup {
        let fog = self.uniforms.fog();
        let ours = [
            [
                out[0] as f32,
                out[1] as f32,
                1.0 / out[0].max(1) as f32,
                1.0 / out[1].max(1) as f32,
            ],
            fog[0].to_array(),
            fog[1].to_array(),
        ];
        let head: &[u8] = bytemuck::cast_slice(&ours);
        let n = head.len().min(c.block.len());
        c.block[..n].copy_from_slice(&head[..n]);
        for (index, ty, offset) in &c.uniforms {
            if let Some(v) = index.and_then(|i| self.uniforms.values.get(i)) {
                v.write(ty, &mut c.block, *offset as usize);
            }
        }
        if !c.block.is_empty() {
            queue.write_buffer(&c.ubo, 0, &c.block);
        }
        let shadow_main = self.shadow_targets.as_ref().map_or(0, |s| s.main);
        let key = u64::from(self.main & c.reads) | u64::from(shadow_main & c.shadow_reads) << 32;
        let group = c.groups.entry(key).or_insert_with(|| {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: c.ubo.as_entire_binding(),
            }];
            self.sampler_entries(Group::Composite, &c.samplers, &c.iris.textures, key, 0, None, &mut entries);
            self.iris_entries(&c.iris, &mut entries);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&c.name),
                layout: &c.layout,
                entries: &entries,
            })
        });
        group.clone()
    }

    /// The texture and sampler entries of `samplers` for a pass of `group`, with the buffers'
    /// sides in `key` (`bind`'s, shared with the compute programs).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sampler_entries<'a>(
        &'a self,
        group: Group,
        samplers: &[Binding],
        extras: &[(u32, Extra)],
        key: u64,
        current: u16,
        pass_atlas: Option<&'a wgpu::TextureView>,
        entries: &mut Vec<wgpu::BindGroupEntry<'a>>,
    ) {
        let targets = self.targets.as_ref().expect("resized");
        let shadow = self.shadow_targets.as_ref();
        let shadow_space = group.shadow();
        for s in samplers {
            let shadow_color =
                |i: usize, side: usize| shadow.and_then(|t| t.color[i].as_ref()).map(|c| &c[side]);
            let texture = match self.iris_view(extras, s.binding) {
                Some(view) => view,
                None => match s.source {
                    Source::Color(i) if shadow_space => {
                        shadow_color(i, usize::from(key & 1 << i != 0)).unwrap_or(&self.dummy)
                    }
                    Source::ShadowColor(i) => {
                        shadow_color(i, usize::from(key & 1 << (32 + i) != 0)).unwrap_or(&self.dummy)
                    }
                    Source::ShadowDepth(i, _) => match shadow {
                        Some(t) => match (i, &t.depth1) {
                            (1, Some(d)) => &d.1,
                            _ => &t.depth.1,
                        },
                        None => &self.shadow.0,
                    },
                    Source::Color(i) => {
                        let side = usize::from(key & 1 << i != 0);
                        match (&targets.mips[i], &targets.color[i]) {
                            (Some(m), _) if current & 1 << i != 0 => &m[side].all,
                            (_, Some(t)) => &t[side],
                            (_, None) => &self.dummy,
                        }
                    }
                    Source::Depth => &targets.depth,
                    Source::Depth1 => match &targets.depth1 {
                        Some(d)
                            if matches!(group, Group::Composite | Group::Final | Group::Translucent) =>
                        {
                            &d.1
                        }
                        _ => &targets.depth,
                    },
                    Source::Noise => &self.noise,
                    Source::Dummy => &self.dummy,
                    Source::Shadow => &self.shadow.0,
                    Source::Atlas => pass_atlas.or(self.atlas.as_ref()).unwrap_or(&self.dummy),
                    Source::Lightmap => &self.lightmap,
                },
            };
            let sampler = match SamplerKind::of(s.source, s.filtering) {
                SamplerKind::Linear => &self.linear,
                SamplerKind::Nearest => &self.nearest,
                SamplerKind::Repeat => &self.repeat,
                SamplerKind::NearestRepeat => &self.nearest_repeat,
                SamplerKind::Comparison => shadow.map_or(&self.shadow.1, |t| &t.comparison),
            };
            entries.push(wgpu::BindGroupEntry {
                binding: s.binding,
                resource: wgpu::BindingResource::TextureView(texture),
            });
            if s.sampler == s.binding + 1 {
                entries.push(wgpu::BindGroupEntry {
                    binding: s.sampler,
                    resource: wgpu::BindingResource::Sampler(sampler),
                });
            }
        }
    }
}

#[allow(dead_code)]
fn _uses_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    entry(binding, ty)
}

impl PackPost {
    /// Every pass's bind groups dropped (an image or buffer they bind was made again).
    pub(super) fn clear_all_groups(&mut self) {
        let clear = |p: &mut super::Pass| p.groups.clear();
        for p in self.passes.iter_mut() {
            clear(p);
        }
        for p in self.shadow_comp.iter_mut() {
            clear(p);
        }
        for p in [&mut self.terrain, &mut self.water, &mut self.clouds, &mut self.shadow_terrain, &mut self.shadow_water]
            .into_iter()
            .flatten()
        {
            clear(p);
        }
        if let Some(sky) = &mut self.sky {
            for p in [&mut sky.basic, &mut sky.textured].into_iter().flatten() {
                clear(p);
            }
        }
        if let Some(e) = &mut self.entities {
            for p in e.passes.iter_mut().chain(e.shadow.iter_mut()).flatten() {
                clear(p);
            }
        }
        if let Some(w) = &mut self.weather {
            for p in w.iter_mut() {
                clear(p);
            }
        }
        if let Some((h, _)) = &mut self.hand {
            clear(h);
        }
        if let Some(iris) = &mut self.iris {
            for c in iris.computes.iter_mut() {
                c.groups.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::gfx::pack_atlas::PackAtlas;
    use crate::gfx::{TERRAIN_ATTRIBUTES, TerrainVertex};
    use horizon_shaders::{GameInputs, Macros, Pack, Properties, dimension_folder, load_post};

    fn view(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat, usage: wgpu::TextureUsages, size: [u32; 2]) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: usage | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        if format == wgpu::TextureFormat::Rgba8Unorm {
            let texels = vec![128u8; (size[0] * size[1] * 4) as usize];
            queue.write_texture(
                texture.as_image_copy(),
                &texels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0] * 4), rows_per_image: None },
                wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            );
        }
        let v = texture.create_view(&Default::default());
        (texture, v)
    }

    fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &read, 0, size);
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = read.slice(..).get_mapped_range().unwrap().to_vec();
        data
    }

    fn read_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, bytes_per_texel: u32) -> Vec<u8> {
        let s = texture.size();
        let row = (s.width * bytes_per_texel).next_multiple_of(256);
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * s.height * s.depth_or_array_layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(s.height) },
            },
            s,
        );
        queue.submit([encoder.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = read.slice(..).get_mapped_range().unwrap();
        let mut out = Vec::new();
        for y in 0..(s.height * s.depth_or_array_layers) {
            let start = (y * row) as usize;
            out.extend_from_slice(&data[start..start + (s.width * bytes_per_texel) as usize]);
        }
        out
    }

    fn half(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = (bits >> 10) & 0x1f;
        let frac = f32::from(bits & 0x3ff);
        match exp {
            0 => sign * frac * 2f32.powi(-24),
            31 => if frac == 0.0 { sign * f32::INFINITY } else { f32::NAN },
            e => sign * (1.0 + frac / 1024.0) * 2f32.powi(i32::from(e) - 15),
        }
    }

    fn small_float(bits: u32, mantissa: u32) -> f32 {
        let exp = (bits >> mantissa) & 0x1f;
        let frac = (bits & ((1 << mantissa) - 1)) as f32;
        let scale = (1u32 << mantissa) as f32;
        match exp {
            0 => frac / scale * 2f32.powi(-14),
            31 => if frac == 0.0 { f32::INFINITY } else { f32::NAN },
            e => (1.0 + frac / scale) * 2f32.powi(e as i32 - 15),
        }
    }

    /// Each colour buffer's "main" side: statistics, and a preview PNG (Reinhard-mapped; NaN and
    /// infinities magenta).
    fn dump_buffers(device: &wgpu::Device, queue: &wgpu::Queue, pp: &PackPost, out: &str, dim: i32) {
        let Some(targets) = &pp.targets else { return };
        for i in 0..BUFFERS {
            let Some(sides) = &targets.color[i] else { continue };
            let side = usize::from(pp.main & 1 << i != 0);
            let texture = sides[side].texture();
            let format = texture.format();
            let bytes = format.block_copy_size(None).unwrap_or(4);
            let data = read_texture(device, queue, texture, bytes);
            let texel = |c: &[u8]| -> [f32; 4] {
                use wgpu::TextureFormat as T;
                match format {
                    T::Rgba16Float => [0, 1, 2, 3].map(|k| half(u16::from_le_bytes([c[2 * k], c[2 * k + 1]]))),
                    T::Rgba16Unorm => [0, 1, 2, 3].map(|k| f32::from(u16::from_le_bytes([c[2 * k], c[2 * k + 1]])) / 65535.0),
                    T::Rgba8Unorm => [0, 1, 2, 3].map(|k| f32::from(c[k]) / 255.0),
                    T::Rgba32Float => [0, 1, 2, 3].map(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap())),
                    T::Rg11b10Ufloat => {
                        let v = u32::from_le_bytes(c[..4].try_into().unwrap());
                        [small_float(v & 0x7ff, 6), small_float((v >> 11) & 0x7ff, 6), small_float(v >> 22, 5), 1.0]
                    }
                    _ => [f32::NAN; 4],
                }
            };
            let texels: Vec<[f32; 4]> = data.chunks_exact(bytes as usize).map(texel).collect();
            let bad = texels.iter().filter(|t| t.iter().any(|v| !v.is_finite())).count();
            let mut min = [f32::MAX; 4];
            let mut max = [f32::MIN; 4];
            let mut sum = [0f64; 4];
            for t in texels.iter().filter(|t| t.iter().all(|v| v.is_finite())) {
                for k in 0..4 {
                    min[k] = min[k].min(t[k]);
                    max[k] = max[k].max(t[k]);
                    sum[k] += f64::from(t[k]);
                }
            }
            let n = (texels.len() - bad).max(1) as f64;
            println!(
                "  colortex{i} {format:?}: nan/inf {bad}, min [{:.3} {:.3} {:.3} {:.3}] max [{:.3} {:.3} {:.3} {:.3}] mean [{:.3} {:.3} {:.3} {:.3}]",
                min[0], min[1], min[2], min[3], max[0], max[1], max[2], max[3],
                sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n
            );
            let size = texture.size();
            if let Ok(probe) = std::env::var("IRIS_TEST_PROBE") {
                for spot in probe.split(';') {
                    let mut it = spot.split([':', ',']);
                    let (Some(b), Some(x), Some(y)) = (it.next(), it.next(), it.next()) else { continue };
                    if b.parse::<usize>().ok() != Some(i) {
                        continue;
                    }
                    let (x, y): (u32, u32) = (x.parse().unwrap_or(0), y.parse().unwrap_or(0));
                    let t = texels[(y * size.width + x) as usize];
                    println!("    colortex{i} at ({x},{y}): [{:.4} {:.4} {:.4} {:.4}]", t[0], t[1], t[2], t[3]);
                }
            }
            let rgba: Vec<u8> = texels
                .iter()
                .flat_map(|t| {
                    if t.iter().any(|v| !v.is_finite()) {
                        [255, 0, 255, 255]
                    } else {
                        let m = |v: f32| ((v.max(0.0) / (1.0 + v.max(0.0))) * 255.0) as u8;
                        [m(t[0]), m(t[1]), m(t[2]), 255]
                    }
                })
                .collect();
            let path = format!("{out}/dim{dim}_colortex{i}.png");
            let file = std::fs::File::create(&path).unwrap();
            let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size.width, size.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header().unwrap().write_image_data(&rgba).unwrap();
        }
    }

    /// `IRIS_TEST_PACK=<pack folder> cargo test -p horizon-client-render --release
    /// iris::tests -- --ignored --nocapture`: builds every pipeline of a real pack on this
    /// machine's GPU (or a software one) and runs frames of it over an empty world.
    #[test]
    #[ignore = "needs IRIS_TEST_PACK"]
    fn runs_a_real_pack() {
        let Some(dir) = std::env::var_os("IRIS_TEST_PACK") else { return };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).expect("an adapter");
        println!("adapter: {:?}", adapter.get_info());
        let wanted = wgpu::Features::CLEAR_TEXTURE
            | wgpu::Features::VERTEX_WRITABLE_STORAGE
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu::Features::SUBGROUP
            | wgpu::Features::SUBGROUP_BARRIER;
        let features = adapter.features() & wanted;
        println!("features: {features:?}");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e| sink.lock().unwrap().push(e.to_string())));

        // A small block atlas: two arrays of 2x2 sprites.
        let arrays: Vec<wgpu::Texture> = (0..2)
            .map(|_| {
                wgpu::util::DeviceExt::create_texture_with_data(
                    &device,
                    &queue,
                    &wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: 2, height: 2, depth_or_array_layers: 3 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &[200u8; 2 * 2 * 3 * 4],
                )
            })
            .collect();
        let mut encoder = device.create_command_encoder(&Default::default());
        let atlas = PackAtlas::new(&device, &mut encoder, &arrays, 3, 6, 2).unwrap();
        queue.submit([encoder.finish()]);

        let pack = Pack::open(std::path::Path::new(&dir)).unwrap();
        let macros = Macros::standard();
        let properties = Properties::parse(
            &pack.read_text("shaders.properties").unwrap_or_default(),
            &horizon_shaders::iris::properties_macros(&pack, &macros),
        );
        let size: [u32; 2] = match std::env::var("IRIS_TEST_SIZE").ok().and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some([w.parse().ok()?, h.parse().ok()?])
        }) {
            Some(s) => s,
            None => [320, 180],
        };
        let instance_attrs = wgpu::vertex_attr_array![5=>Float32x4];
        let buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<TerrainVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &TERRAIN_ATTRIBUTES,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            }),
        ];
        let only: Option<i32> = std::env::var("IRIS_TEST_DIM").ok().and_then(|d| d.parse().ok());
        for dim in [0, -1, 1] {
            if only.is_some_and(|o| o != dim) {
                continue;
            }
            errors.lock().unwrap().clear();
            let folder = dimension_folder(&pack, dim);
            let inputs = GameInputs {
                terrain: Some(atlas.terrain_inputs(false)),
                sky: Some(super::super::super::pack_sky::inputs()),
                celestial: Some(super::super::super::pack_sky::celestial_inputs()),
                entities: Some(super::super::super::pack_entities::inputs()),
                clouds: None,
                weather: Some(super::super::super::pack_weather::inputs()),
                hand: Some(super::super::super::pack_hand::inputs()),
            };
            let post = load_post(&pack, &folder, &properties, &macros, &inputs);
            // Storage buffer 4's members, by the layout the programs were translated with.
            let mut members: Vec<(String, u32, String)> = Vec::new();
            if let Some(c) = post.iris.computes.iter().find(|c| c.program.interface.buffers.contains(&4)) {
                let m = &c.program.module;
                for (_, g) in m.global_variables.iter() {
                    if g.binding.as_ref().is_some_and(|b| b.binding == 1004)
                        && let naga::TypeInner::Struct { members: ms, .. } = &m.types[g.ty].inner
                    {
                        for mem in ms {
                            let kind = match &m.types[mem.ty].inner {
                                naga::TypeInner::Scalar(_) => "1".to_string(),
                                naga::TypeInner::Vector { size, .. } => (*size as u8).to_string(),
                                _ => "skip".to_string(),
                            };
                            members.push((mem.name.clone().unwrap_or_default(), mem.offset, kind));
                        }
                    }
                }
            }
            let (_, lightmap) = view(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::empty(), [16, 16]);
            let (_, celestial) = view(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::empty(), [16, 16]);
            let world = World {
                atlas: Some((atlas.view.clone(), atlas.size())),
                lightmap,
                terrain: Some(Geometry { buffers: &buffers, table: None, depth_write: true, atlas: None, ids: None, cull: true }),
                block_ids: &[],
                celestial: Some(celestial),
                clouds: None,
                weather: true,
                hand: true,
            };
            let started = std::time::Instant::now();
            let mut pp = PackPost::new(&device, &queue, post, wgpu::TextureFormat::Rgba8Unorm, world);
            println!("=== dimension {dim} ({folder}): built in {:.1} s", started.elapsed().as_secs_f32());
            let mut zero: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            for n in pp.notes.iter().filter_map(|n| n.split_once(": zero for now: ")) {
                zero.extend(n.1.split_whitespace().map(str::to_owned));
            }
            println!("  uniforms read as zero: {zero:?}");
            for n in pp.notes.iter().filter(|n| n.starts_with("custom uniforms")) {
                println!("  {}", n.chars().take(600).collect::<String>());
            }
            for n in pp.notes.iter().filter(|n| n.contains("skipped") || n.contains("compute") || n.contains("blank") || n.contains("image") || n.contains("texture") || n.contains("not provided")) {
                println!("  {}", n.chars().take(300).collect::<String>());
            }
            let built = errors.lock().unwrap().clone();
            for e in &built {
                println!("  GPU error while building: {}", e.chars().take(400).collect::<String>());
            }
            let (_, world_color) = view(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT, size);
            let (_, world_depth) = view(&device, &queue, wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT, size);
            let (screen_texture, screen) = view(&device, &queue, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::RENDER_ATTACHMENT, size);
            let hud = crate::hud::HudBatch::new(size);
            let day_time: i64 = std::env::var("IRIS_TEST_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(6000);
            let pack_inputs = PackInputs { day_time, eye_light: [0, 15], brightness: 0.5, ..Default::default() };
            let mut frame = crate::gfx::Frame {
                camera: glam::DVec3::new(0.5, 80.0, 0.5),
                yaw: 30.0,
                pitch: -10.0,
                fov_degrees: std::env::var("IRIS_TEST_FOV").ok().and_then(|v| v.parse().ok()).unwrap_or(70.0),
                bob: glam::Mat4::IDENTITY,
                spin: glam::Mat4::IDENTITY,
                // Noon: the sun overhead, the moon opposite, vanilla's plains sky.
                sky: crate::environment::Sky {
                    // Vanilla's colours: the plains' sky, the Nether wastes' fog, the End's.
                    sky_color: match dim { -1 => [0.2, 0.03, 0.03], 1 => [0.0, 0.0, 0.0], _ => [0.47, 0.65, 1.0] },
                    fog_color: match dim { -1 => [0.2, 0.03, 0.03], 1 => [0.06, 0.0, 0.06], _ => [0.75, 0.85, 1.0] },
                    sky_light_factor: 1.0,
                    sky_light_color: [1.0, 1.0, 1.0],
                    block_light_tint: [1.0, 1.0, 1.0],
                    ambient_light_color: [1.0, 1.0, 1.0],
                    sun_angle: std::env::var("IRIS_TEST_SUN").ok().and_then(|a| a.parse().ok()).unwrap_or(0.0),
                    moon_angle: std::env::var("IRIS_TEST_SUN").ok().and_then(|a| a.parse::<f32>().ok()).unwrap_or(0.0) + 180.0,
                    star_angle: 0.0,
                    star_brightness: 0.0,
                    sunrise_color: [0.0; 4],
                    cloud_color: [1.0; 4],
                    sky_light_level: 1.0,
                    fog_start: 0.0,
                    fog_end: 0.0,
                    cloud_fog_end: 0.0,
                },
                distance: 128.0,
                detail_distance: 128.0,
                depth_far: 512.0,
                underwater: false,
                cave_culling: false,
                title: false,
                sky_blocked: false,
                skybox: true,
                end_sky: dim == 1,
                end_sky_sprite: None,
                environment_fog: None,
                water_fog: [0.0; 3],
                fog: None,
                dark_disc: false,
                outline: Vec::new(),
                outline_after_terrain: false,
                outline_high_contrast: false,
                debug_lines: &[],
                debug_lines_on_top: &[],
                debug_crosshair: &[],
                hud: &hud,
                glint_alpha: 0.0,
                glint_ticks: 0.0,
                nether_lighting: dim == -1,
                entities: &[],
                glowing: &[],
                entity_spans: &[],
                weather: &[],
                weather_time: bytemuck::Zeroable::zeroed(),
                dynamic_lights: &[],
                clouds: None,
                lightning: Default::default(),
                additive: &[],
                see_through: &[],
                celestial: &[],
                stars: None,
                hand: &[],
                hand_fov_degrees: 70.0,
                screen: &[],
                pack: pack_inputs,
                border: None,
                game_time: 0.0,
                shader_game_time: 0.0,
                end_portal_sprites: [[0.0; 4]; 2],
                crumbling_sprite: [0.0; 4],
                ticks: 0,
                darkness: (0.0, 0.0),
                night_vision: 0.0,
                block_flicker: 0.0,
                boss_darken: 0.0,
                menu_blur: None,
            };
            let frames: usize = std::env::var("IRIS_TEST_FRAMES").ok().and_then(|f| f.parse().ok()).unwrap_or(6);
            let started = std::time::Instant::now();
            for f in 0..frames {
                // `IRIS_TEST_AIM_MOON`: after a few frames, look where the pack's own compute
                // program put the moon (storage buffer 4, after its shadow matrix).
                if f == 8 && std::env::var_os("IRIS_TEST_AIM_MOON").is_some()
                    && let Some(b) = pp.iris.as_ref().and_then(|i| i.buffer(4))
                {
                    let data = read_buffer(&device, &queue, b, 128);
                    let v = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
                    let m = [v(64), v(68), v(72)];
                    frame.pitch = -m[1].asin().to_degrees();
                    frame.yaw = (-m[0]).atan2(m[2]).to_degrees();
                    println!("  aiming at the moon [{:.3} {:.3} {:.3}]: yaw {:.1}, pitch {:.1}", m[0], m[1], m[2], frame.yaw, frame.pitch);
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                pp.run(
                    &device,
                    &queue,
                    &mut encoder,
                    &world_color,
                    &world_depth,
                    &screen,
                    &frame,
                    &pack_inputs,
                    size,
                    size,
                    &[],
                    &mut |_: &mut wgpu::RenderPass<'_>, _: TerrainList, _: &dyn Fn(&mut wgpu::RenderPass<'_>, u32)| {},
                );
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            println!("  {frames} frames in {:.1} s", started.elapsed().as_secs_f32());
            for e in errors.lock().unwrap().iter().skip(built.len()) {
                println!("  GPU error while running: {}", e.chars().take(600).collect::<String>());
            }
            let iris = pp.iris.as_ref().unwrap();
            println!("  post passes: {:?}", pp.passes.iter().map(|p| p.name.as_str()).collect::<Vec<_>>());
            println!("  passes {} (computes {}), images {}, buffers {}, custom textures {}",
                pp.passes.len(), iris.computes.len(), iris.images.len(), iris.buffers.len(), iris.textures.len());
            // What the programs wrote: storage buffer 4's moon vector (after a mat4), the setup
            // program's block table, and the picture.
            if let Some(b) = iris.buffer(4) {
                let all = read_buffer(&device, &queue, b, 1024);
                let want = std::env::var("IRIS_TEST_MEMBERS").unwrap_or_default();
                for (name, offset, kind) in &members {
                    if !want.split(',').any(|w| !w.is_empty() && name.contains(w)) || kind == "skip" {
                        continue;
                    }
                    let n: usize = kind.parse().unwrap_or(1);
                    let v: Vec<f32> = (0..n)
                        .map(|k| f32::from_le_bytes(all[*offset as usize + 4 * k..*offset as usize + 4 * k + 4].try_into().unwrap()))
                        .collect();
                    println!("    {name} @{offset}: {v:?}");
                }
                let data = read_buffer(&device, &queue, b, 256);
                let f = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
                let m = [f(64), f(68), f(72)];
                let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
                println!("  buffer 4: shadow matrix row0 [{:.3} {:.3} {:.3} {:.3}], moon vector [{:.3} {:.3} {:.3}] (length {:.3})", f(0), f(16), f(32), f(48), m[0], m[1], m[2], len);
            }
            if let Some((k, image)) = iris.images.iter().enumerate().find(|(_, i)| i.decl.name == "imgBlockData") {
                if let Some((texture, _)) = &image.texture {
                    let data = read_texture(&device, &queue, texture, 8);
                    let filled = data.chunks_exact(8).filter(|c| c.iter().any(|&b| b != 0)).count();
                    println!("  image {k} imgBlockData: {filled} of {} entries written by setup", data.len() / 8);
                }
            }
            let picture = read_texture(&device, &queue, &screen_texture, 4);
            let n = (picture.len() / 4) as f64;
            let mean: Vec<f64> = (0..3).map(|c| picture.chunks_exact(4).map(|p| f64::from(p[c])).sum::<f64>() / n).collect();
            let distinct = picture.chunks_exact(4).map(|p| (p[0], p[1], p[2])).collect::<std::collections::BTreeSet<_>>().len();
            println!("  picture: mean rgb [{:.1} {:.1} {:.1}], {distinct} distinct colours", mean[0], mean[1], mean[2]);
            if let Ok(out) = std::env::var("IRIS_TEST_DUMP") {
                dump_buffers(&device, &queue, &pp, &out, dim);
            }
            for e in errors.lock().unwrap().iter() {
                println!("  GPU error (all): {}", e.chars().take(300).collect::<String>());
            }
            if let Ok(out) = std::env::var("IRIS_TEST_PNG") {
                let path = format!("{out}/eclipse_dim{dim}.png");
                let file = std::fs::File::create(&path).unwrap();
                let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size[0], size[1]);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header().unwrap();
                let opaque: Vec<u8> = picture.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
                writer.write_image_data(&opaque).unwrap();
                println!("  wrote {path}");
            }
        }
    }
}

#[cfg(test)]
mod spv_tests {
    use horizon_shaders::{GameInputs, Macros, Pack, Properties, dimension_folder, load_post};

    /// `IRIS_TEST_PACK=<pack> IRIS_TEST_PROGRAM=gbuffers_hand cargo test ... spv_tests -- --ignored`:
    /// writes the program's stages to SPIR-V one by one and reports where naga's writer fails.
    #[test]
    #[ignore = "needs IRIS_TEST_PACK"]
    fn writes_spirv() {
        let Some(dir) = std::env::var_os("IRIS_TEST_PACK") else { return };
        let want = std::env::var("IRIS_TEST_PROGRAM").unwrap_or_else(|_| "gbuffers_hand".into());
        let pack = Pack::open(std::path::Path::new(&dir)).unwrap();
        let macros = Macros::standard();
        let properties = Properties::parse(
            &pack.read_text("shaders.properties").unwrap_or_default(),
            &horizon_shaders::iris::properties_macros(&pack, &macros),
        );
        let g = horizon_shaders::GbuffersInputs {
            declarations: "layout(location = 0) in vec3 hz_v_pos;".into(),
            body: "    hz_Vertex = vec4(hz_v_pos, 1.0);\n    hz_Color = vec4(1.0);\n".into(),
            ..Default::default()
        };
        let inputs = GameInputs {
            terrain: Some(g.clone()),
            sky: Some(g.clone()),
            celestial: Some(g.clone()),
            entities: Some(g.clone()),
            clouds: None,
            weather: Some(g.clone()),
            hand: Some(g),
        };
        let post = load_post(&pack, &dimension_folder(&pack, 0), &properties, &macros, &inputs);
        let all = post.programs.iter()
            .chain(&post.terrain).chain(&post.water).chain(&post.hand).chain(&post.entities)
            .chain(&post.block).chain(&post.particles).chain(&post.weather).chain(&post.sky).chain(&post.sky_textured)
            .chain(&post.shadow).chain(&post.shadow_water).chain(&post.shadow_entities);
        for p in all.filter(|p| want == "*" || p.name == want) {
            for (stage, module) in [("vertex", &p.program.vertex), ("fragment", &p.program.fragment)] {
                let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                    .validate(module)
                    .expect("valid");
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    naga::back::spv::write_vec(module, &info, &Default::default(), None)
                }));
                match result {
                    Ok(Ok(_)) => println!("{} {stage}: ok", p.name),
                    Ok(Err(e)) => println!("{} {stage}: error {e}", p.name),
                    Err(_) => {
                        println!("{} {stage}: PANIC", p.name);
                        // Each function alone, called with zero values from an entry point of
                        // its own: the failing one whose callees pass is the one to read.
                        let mut failing = Vec::new();
                        for (h, f) in module.functions.iter() {
                            let mut m = module.clone();
                            let mut fun = naga::Function::default();
                            let mut args = Vec::new();
                            for a in &f.arguments {
                                args.push(fun.expressions.append(naga::Expression::ZeroValue(a.ty), naga::Span::UNDEFINED));
                            }
                            if !args.is_empty() {
                                let start = args[0];
                                let end = *args.last().unwrap();
                                let _ = (start, end);
                            }
                            let result = f.result.as_ref().map(|_| fun.expressions.append(naga::Expression::CallResult(h), naga::Span::UNDEFINED));
                            fun.body.push(naga::Statement::Call { function: h, arguments: args, result }, naga::Span::UNDEFINED);
                            m.entry_points = vec![naga::EntryPoint {
                                name: "probe".into(),
                                stage: naga::ShaderStage::Fragment,
                                early_depth_test: None,
                                workgroup_size: [0, 0, 0],
                                workgroup_size_overrides: None,
                                function: fun,
                                mesh_info: None,
                                task_payload: None,
                                incoming_ray_payload: None,
                            }];
                            naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
                            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&m);
                            let Ok(info) = info else {
                                failing.push(format!("(invalid alone) {}", f.name.clone().unwrap_or_default()));
                                continue;
                            };
                            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                naga::back::spv::write_vec(&m, &info, &Default::default(), None)
                            }));
                            if r.is_err() {
                                failing.push(f.name.clone().unwrap_or_default());
                            }
                        }
                        println!("  failing functions: {failing:?}");
                        if let Some((_, f)) = module.functions.iter().find(|(_, f)| f.name.as_deref() == Some("hz_main")) {
                            println!("  hz_main: {} expressions", f.expressions.len());
                            for (h, e) in f.expressions.iter() {
                                if (138..148).contains(&h.index()) {
                                    println!("    [{}] {:?}", h.index(), e);
                                }
                            }
                            // Expressions that use [143]
                            for (h, e) in f.expressions.iter() {
                                let t = format!("{e:?}");
                                if t.contains("[143]") {
                                    println!("    user [{}] {}", h.index(), t);
                                }
                            }
                            let t = format!("{:?}", f.body);
                            for needle in ["[143]", "[144]"] {
                                let mut from = 0;
                                while let Some(at) = t[from..].find(needle) {
                                    let at = at + from;
                                    println!("    body near {needle}: ...{}...", &t[at.saturating_sub(400)..(at + 200).min(t.len())]);
                                    from = at + 5;
                                    if from > t.len() { break; }
                                    break;
                                }
                            }
                        }
                        let ep = &module.entry_points[0].function;
                        println!("  entry point: {} expressions, {} statements", ep.expressions.len(), ep.body.len());
                        for (h, e) in ep.expressions.iter() {
                            if (135..150).contains(&h.index()) {
                                println!("    [{}] {:?}", h.index(), e);
                            }
                        }
                        for (k, st) in ep.body.iter().enumerate() {
                            let t = format!("{st:?}");
                            if t.contains("[143]") || t.contains("index: 143") || k < 4 {
                                println!("    stmt {k}: {}", t.chars().take(700).collect::<String>());
                            }
                        }
                        let t = format!("{:?}", ep.body);
                        if let Some(at) = t.find("143") {
                            println!("    around: {}", &t[at.saturating_sub(600)..(at + 300).min(t.len())]);
                        }
                    }
                }
            }
        }
    }
}
