
/// A raw texture's format, channels and bytes per channel (`RGB8 UNSIGNED_BYTE`: RGBA8, 3, 1;
/// three channels are stored as four).
fn raw_layout(internal: &str, pixel: &str) -> Option<(wgpu::TextureFormat, usize, usize)> {
    use wgpu::TextureFormat as F;
    let channels = ["RGBA", "RGB", "RG", "R"]
        .iter()
        .find(|p| internal.starts_with(*p))
        .map(|p| p.len())?;
    let float = internal.ends_with('F');
    let pick = |formats: [F; 3]| formats[channels.min(3) - 1];
    if internal.ends_with("_SNORM") {
        return Some(match pixel {
            "BYTE" => (pick([F::R8Snorm, F::Rg8Snorm, F::Rgba8Snorm]), channels, 1),
            "SHORT" => (pick([F::R16Snorm, F::Rg16Snorm, F::Rgba16Snorm]), channels, 2),
            _ => return None,
        });
    }
    Some(match (pixel, float) {
        ("UNSIGNED_BYTE", false) => (pick([F::R8Unorm, F::Rg8Unorm, F::Rgba8Unorm]), channels, 1),
        ("UNSIGNED_SHORT", false) => (pick([F::R16Unorm, F::Rg16Unorm, F::Rgba16Unorm]), channels, 2),
        ("HALF_FLOAT", true) => (pick([F::R16Float, F::Rg16Float, F::Rgba16Float]), channels, 2),
        ("FLOAT", true) => (pick([F::R32Float, F::Rg32Float, F::Rgba32Float]), channels, 4),
        _ => return None,
    })
}

/// `bytes`, with an opaque alpha after each texel of three channels.
fn widen_rgb(bytes: &[u8], channels: usize, size_of: usize, pixel: &str) -> Vec<u8> {
    if channels != 3 {
        return bytes.to_vec();
    }
    let one: &[u8] = match pixel {
        "HALF_FLOAT" => &[0x00, 0x3C],
        "FLOAT" => &[0x00, 0x00, 0x80, 0x3F],
        "UNSIGNED_SHORT" => &[0xFF, 0xFF],
        "SHORT" => &[0xFF, 0x7F],
        "BYTE" => &[0x7F],
        _ => &[0xFF],
    };
    let mut out = Vec::with_capacity(bytes.len() / 3 * 4);
    for texel in bytes.chunks_exact(3 * size_of) {
        out.extend_from_slice(texel);
        out.extend_from_slice(one);
    }
    out
}
