# Iris Features: a native mod for Horizon

This mod makes Horizon load shader packs written for Iris's own extensions. Unmodified Eclipse
Shader runs with it: compute programs, storage buffers, images, custom textures (the real moon,
noise, cloud and water textures), and the Iris-only uniforms and macros.

It is a native mixin bundle (`horizon-mixins`). The launcher compiles it into a game build of its own.

## Install

You need the Horizon launcher with native mods (Settings → Game → Native mods).

1. Build it from the command line, pointing at the unpacked folder:

   ```sh
   horizon-launcher native-build path/to/iris-features --trust-native-code
   ```

   Alternatively, copy `iris-features.hnmod` into your instance's `horizon-native-mods/` folder,
   turn it on, and press **Build** on the Mods page.

   The first build downloads Rust and Horizon's source, then compiles the game: 10 to 30 minutes
   depending on the computer. Later builds are faster.
2. Put Eclipse (the folder or the zip, unmodified) in `shaderpacks/` and pick it in
   Options → Horizon Settings → Shader Packs, in the instance the launcher built.

### Pins

The mod is pinned to these exact commits. The launcher reads them from `horizon-native.json`.

- Horizon `fca2169de9c014be8b6606ac1f3dba0bcda3731a` (main on 2026-10-30).
- horizon-mixins `5140cdf81915c25f10a7745eb532e1c09f275068` (main on 2026-10-30). The hooks need
  its support for `format!` and the other standard macros in hooked functions.

A Horizon Nightly with other changes needs the mod rebuilt against it. Run `tools/build.py` with
`HORIZON` set to the new commit. A hook whose target changed reports a diagnostic naming that
function.

## What it adds

| Iris feature | What the mod does |
|---|---|
| Macros | Defines `IS_IRIS`, `IRIS_VERSION` (11103), `IRIS_FEATURE_SSBO`, `IRIS_FEATURE_CUSTOM_IMAGES`, `IRIS_FEATURE_COMPUTE_SHADERS`, `IRIS_FEATURE_CUSTOM_TEXTURE_NAME`, and `MC_GL_ARB_shader_image_load_store` (the GL extension Iris reports for images; Eclipse's floodfill coloured lighting is off without it) |
| Options in `.properties` | `#if` on the pack's options in `shaders.properties`/`block.properties`, and option values inside `uniform.*`/`variable.*` expressions |
| `smooth()` timing | Custom uniforms' `smooth(x, up, down)` take half-lives in tenths of a second, as in Iris; Horizon alone reads them as seconds, 10x slower |
| View bobbing | In `gbufferModelView`, not `gbufferProjection`, as Iris has it (packs read the projection as a plain perspective; otherwise clouds and lighting sway with each step) |
| Matrix components in custom uniforms | `m.i.j` is GLSL's `m[i][j]` (column, row), as in Iris; Horizon alone reads row, column (so the sun's world direction turned with the camera) |
| Flat terrain draws | Turned on wherever the GPU supports them (Horizon uses them only on Metal): a pack's terrain programs get `mc_Entity` (waving plants, water, glass) and each face's normal and tangent only from them. `HORIZON_FLAT_DRAWS=0` turns them off |
| Translucent particles | `gbuffers_particles_translucent` draws the particles on the translucent layer (campfire smoke, splashes...) with its own blend, the others draw with `gbuffers_particles`. `particles.ordering=mixed` draws the opaque ones before deferred and the translucent ones after, as Iris does (Horizon alone drew every particle with `gbuffers_particles` after deferred, so Eclipse's particles came out as raw G-buffer data) |
| Rain and snow depth | Weather writes depth only with `rain.depth=true`, as in Iris (Horizon alone always wrote it, so Eclipse took each whole rain quad for a surface in front of the camera: dark boxes, discoloured streaks, and the wrong lighting behind them) |
| `at_midBlock` | Flat terrain draws give each vertex its offset to the block's centre (in 64ths), as Iris does. Eclipse's floodfill voxelises blocks there; without it each block landed on whichever neighbouring voxel its corners rounded to, and that changed as the camera moved |
| Rain, snow and clouds that start later | Horizon dropped a pack's rain, snow and `gbuffers_clouds` passes on the first frame without them (a world loaded in clear weather), then drew its own rain into the pack's image: drops that showed block textures and an unlit scene behind them. The passes now stay |
| Shader options | Options → Mods → Iris Features lists the running pack's own options as Iris's shader pack screen has them: its screens and sub-screens, profiles, and the names its `lang/` files give options and values. A click moves an option to its next value (Shift+click: the one before); **Apply** writes Iris's own `shaderpacks/<pack>.txt` (only values that differ from the pack's defaults, so the file moves between Iris and Horizon) and loads the pack again. The page's last switch, **Floodfill diagnostics in the log**, writes the floodfill voxels and light totals of four frames in a row to the game log every few seconds, and the voxels near the camera whose light kept changing (with the block ids around them), for reporting flickering lights. Switches (`#define X` / `//#define X`), value options (`#define X 2 // [1 2 3]`, the list anywhere in the comment) and `const` options (`shadowMapResolution`, `sunPathRotation`) all work, in the programs and in `.properties` conditions |
| Block ids for floodfill | Flat draws know a face's sprite, not its block, and Horizon gave each sprite the id most of its blocks have: an unlit furnace's sides were the lit furnace's (194), pumpkins were jack o'lanterns, and one block's faces could carry two ids, so Eclipse's voxels held whichever face the GPU drew last (lights that flickered). The mod picks the id by sprite **and** by how brightly the face's block glows (the HDR tier Horizon puts in each vertex colour), counts blocks without an id, and leaves a shared sprite's faces out of the voxels when the block's other faces place it. Redstone dust gets its power's id from its tint |
| Decimal `#if` | `#if SPEED < 1.0`, as Iris's preprocessor allows |
| `bufferObject.N` | Storage buffers (fixed or screen-relative), at set 0 binding 1000 + N |
| `image.*` | Images, cleared per frame when asked, sized by the screen or fixed, readable through their sampler name |
| `customTexture.*`, `texture.noise`, `texture.<stage>.<sampler>` | PNG and raw (`RGBA8`) textures, 1D/2D/3D. As in Iris, `texture.<stage>.colortexN` replaces buffer N in that stage's programs until one of them has written the buffer (Eclipse's blue noise in `colortex6`) |
| Compute programs | `setup`, `begin`, `shadowcomp`, `prepare`, `deferred` and `composite` `.csh` files (with `_a`..`_z` suffixes), with `workGroups`/`workGroupsRender`. Each runs before the program of its own stage and number |
| Iris uniforms | `currentPlayerHealth/Hunger/Air` (0 to 1) and their maximums, `firstPersonCamera`, `isSpectator`, `playerLookVector`, `is_burning/hurt/invisible/on_ground/sneaking/sprinting`, `isRightHanded`, `alphaTestRef` (0.1: cutout textures like plants and leaves) |
| `normals`/`specular` | Without a PBR resource pack: a flat normal, and zeros |
| Geometry stages | A `.gsh` that emits one triangle per triangle (Eclipse's terrain and beacon ones, with shader grass off) is skipped and the program runs without it |
| GLSL that Iris's drivers accept | `const` initialised with function calls, interface blocks, bool members of storage buffers, `ivec % int`, swizzles of shadow lookups, functions that end without a `return`, matrix-column swizzle assignments, `sampler`/`image` used as names, mixed `ivec`/`uvec` comparisons, and vector components passed to `out`/`inout` parameters; also `dot` of two floats, `vec % int` with a constant, struct varyings, `float[6] name` struct members, functions called before they are defined, unmatched `#endif` (ignored, as Iris's preprocessor does), `texture.<stage>.<sampler>.N` keys, and SNORM raw textures |

## Limits

- **Geometry and tessellation shaders that add geometry are impossible on wgpu.** Eclipse's shader
  grass (`SHADER_GRASS_SETTING`, off by default) can't run. Turning it on makes the terrain
  program skip with a note in the log.
- Image atomics are translated only as statements (`imageAtomicMax(img, p, v);`); their return
  value isn't available. Buffer atomics (`atomicAdd` on buffers) are not translated.
- Other Iris packs, in the software-Vulkan harness: Bliss, Pegasus, Sildur's Vibrant and Just
  Colored Lighting render; Luster and Reverie build but come out black, Noble white. Buffer
  atomics (`atomicAdd`) aren't translated yet.
- The mod carries its own copy of naga's GLSL frontend (`vendor/naga-glsl`, MIT or Apache-2.0,
  changes listed in its README): naga's own refuses GLSL that Iris packs rely on (array
  members of plain structs, `ivec2 % int`, `f().x`, `float a[4], b[4]`).
- `lightningBoltPosition` always reads "no bolt", so lightning doesn't cast Eclipse's lightning
  shadows.
- The options page is a list of buttons (what Horizon gives native mods): an option with many
  values (sliders in Iris) is stepped through a click at a time, and the pack's descriptions of
  its options (Iris's tooltips) aren't shown.
- Flat draws carry a face's sprite and glow, not its block: two blocks that share every sprite
  and glow alike but have different ids in `block.properties` still look alike to the pack.
- Writing images or buffers from a vertex stage needs the GPU's `VERTEX_WRITABLE_STORAGE`. Most
  Vulkan, DX12 and Metal GPUs have it; without it, those programs are skipped with a note.
- Testing so far: the bundle was applied with `horizon-mixins apply` to pristine Horizon
  `fca2169` and the whole game compiled with `--locked` (Cargo.lock unchanged, no warnings).
  Horizon's 587 renderer tests pass; the two `crash::` tests fail without the mod too.
- Unmodified Eclipse ran on that build under Mesa's software Vulkan driver (lavapipe) in all
  three dimensions. All 15 post passes and 8 compute programs ran with no GPU errors, and the
  sky, clouds and textured moon rendered. That was over an empty world: it has not been tested
  on a real GPU, in the game itself, or with terrain in view. Later tests drew terrain, water, particles
  and rain; floodfill voxelises terrain in the shadow pass and spreads light from glowstone, steadily
  frame to frame with the camera walking and turning, and only from the faces whose glow says so.
- The options are read from unmodified Eclipse as Iris reads them (762 options, 83 screens, 11
  profiles; the default profile shows as "High"), and values written back reach both the programs
  and `shaders.properties`. The page itself was tested on its own (cycling, profiles, screens,
  Apply), not in the running game.

## How it is built

`tools/build.py` assembles the bundle from the reference implementation (Horizon branch
`iris-features`) and the pinned pristine source. Support modules carry the new code. Cancellable
HEAD hooks replace a handful of entry points: `eval_condition`, `load_post`, the translator,
`Pack::read_text` (the player's option values), `FrameUniforms::eval_custom`,
`PackPost::{bind, run, run_shadow, prepare_entities, draw_entities, draw_weather, draw_clouds}`,
`PackAtlas::terrain_inputs` and the game's `shader_pack_inputs`; the options page uses Horizon's
native-mod hook points (`native_mods`, `native_settings`, `native_settings_click`). RETURN hooks
add the device features and turn flat draws on. horizon-mixins can't add struct
fields to functions that use closures, so a pack's GPU resources are kept beside its `PackPost`,
keyed by one of its textures.

To rebuild the bundle, apply `reference.patch` to Horizon `fca2169`, then copy the changed files into
`ref/` (same paths) and run `tools/build.py` and `tools/package.py`. `tools/stage.sh` applies the bundle to a pristine
export for checking.

## Licence

The mod's code is MIT, like Horizon. Eclipse Shader is © Chocapic13 and its authors, all rights
reserved. The mod contains none of it: you supply your own copy of the pack.
