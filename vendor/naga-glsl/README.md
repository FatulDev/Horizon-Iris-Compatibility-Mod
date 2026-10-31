# naga's GLSL frontend, modified

The GLSL frontend of [naga](https://github.com/gfx-rs/wgpu) 30.0.1 (`src/front/glsl`), by the
naga authors, under the MIT or Apache-2.0 licence (both texts are here). Iris Features builds
it into Horizon in place of the one inside naga, so it can read GLSL that Iris shader packs use
and naga's own frontend refuses. Changes:

- Builds outside naga: `crate::` paths point at naga's public API, the few crate-private
  helpers it used are rewritten here, and `log`, `thiserror`, `bitflags` and
  `codespan-reporting` are replaced (no dependencies but naga).
- `pp.rs` replaces `pp-rs`: a tokenizer only, since Horizon has already preprocessed the text.
- Plain structs are laid out as `std430` (the shader's own array strides), not `std140`.
- `%` widens a scalar operand to the other's vector size, as the other operators do.
- A scalar takes a swizzle of its one component (`f().x`, `t.xx`).
- The tests are left out.
