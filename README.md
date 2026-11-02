# Iris Features: a native mod for Horizon

This mod extends Horizon's shader pack support toward running every pack written for Iris,
unmodified. Horizon renders many packs on its own; the mod fills in what Iris packs rely on and
Horizon doesn't yet do: the GLSL that Iris's GL drivers accept, Iris's extra programs, images,
macros and uniforms, and the places where Iris's behaviour differs from Horizon's.

Packs are tested as they ship, and every fix is general: it implements what Iris does, not a
special case for one pack. The packs named below are examples of what needed each fix.

It is a native mixin bundle (`horizon-mixins`). The launcher compiles it into a game build of its own.

## Install

You need the Horizon launcher with native mods (Settings → Game → Native mods).

1. Build it from the command line, pointing at the unpacked folder:

   ```sh
   horizon-launcher native-build path/to/iris_features --trust-native-code
   ```

   Alternatively, copy `iris_features.hnmod` into your instance's `horizon-native-mods/` folder,
   turn it on, and press **Build** on the Mods page.

   The first build downloads Rust and Horizon's source, then compiles the game: 10 to 30 minutes
   depending on the computer. Later builds are faster.
2. Put any Iris shader pack (the folder or the zip, unmodified) in `shaderpacks/` and pick it
   in Options → Horizon Settings → Shader Packs, in the instance the launcher built.

### Pins

The mod is pinned to these exact commits. The launcher reads them from `horizon-native.json`.

- Horizon `10b16c4b94d6ebdf3234545f53243c6466dfa32c` (main on 2026-11-01).
- horizon-mixins `4342a0bfbe71d4d14185528d64e496d6022f16a1` (main on 2026-10-31).

## Since 0.1.26: built on Horizon's own Iris support

Horizon now runs much of what this mod used to add itself: compute programs, images, storage
buffers, custom textures, geometry stages for terrain's shadow, struct varyings and a Shader
Options page. From 0.1.26 the mod keeps Horizon's implementation and adds only what packs still
need on top of it.

## What it adds

The packs in brackets are examples of packs that needed each change.

| Area | What the mod does |
|---|---|
| GLSL that Iris's drivers accept | naga's GLSL frontend, vendored with fixes (`vendor/naga-glsl`), tried first, with Horizon's own frontend as the fallback: `const in` parameters, interface blocks (`out DATA { ... } data_out;`), `bool` members of storage buffers, functions that end without `return`, `ivec % int`, swizzles of shadow lookups, relational comparisons of mixed types, matrix-column swizzle assignments, `dot` of two floats, `textureGather`/`textureGatherOffset(s)`/`textureQueryLod`, `texelFetch2D`/`textureSize2D`, memory barriers, `texture` and `sampler`/`image` used as names, integer varyings made flat, `type[N] name` declarations, implicit-LOD lookups outside fragment stages, and passes over the parsed module (workgroup barriers, vector modulo, struct strides, callees before callers, atomics, vertex-stage voxel stores) |
| `setup.csh`, `begin.csh` | Run before the frame's prepare programs (lookup tables, sun and moon data; Eclipse). Setup runs every frame here |
| 1D images | `image.<name>` with a single size (block tables; Eclipse) |
| Geometry stages | One that emits one triangle per triangle (Eclipse's terrain stage with shader grass off) is skipped and the program runs without it |
| Macros | `IRIS_VERSION` 11103, `MC_NORMAL_MAP`, `MC_SPECULAR_MAP`, `MC_GL_ARB_shader_image_load_store`, `MC_GL_ARB_texture_gather`, `MC_GL_EXT_gpu_shader4` (packs that check the Iris version, as Eclipse does, or test for normal maps, as Bliss does) |
| Options in custom uniforms | `uniform.*`/`variable.*` expressions read the pack's options as their values |
| Iris's player uniforms | `currentPlayerHealth/Hunger/Air/Armor` and maximums, `isSpectator`, `is_burning/hurt/invisible/on_ground/sneaking/sprinting`, `lightningBoltPosition` (no bolt) |
| `normals`/`specular` | Without a PBR resource pack: a flat normal and zeros (Horizon's white stand-in made every surface metal, smooth and glowing) |
| `texture.<stage>.colortexN` | Stands in for buffer N only until a program has written it, as in Iris (a noise texture in a buffer the pack later renders into; Eclipse's bloom) |
| Terrain alpha test | Cutout faces (leaves, plants, campfires) shed texels below alpha 0.1, as Iris tests its cutout layers, also with declared outputs; solid faces aren't tested |
| `mc_midTexCoord` | The middle of the quad's own texture coordinates, not of its sprite (stretched grass sides) |
| Distant terrain | While a pack runs, distant sides are meshed a block each (grass sides stretched across far hills); tops only when small (meshing every distant top a block each filled video memory on 8 GB cards) |
| Block outline | Drawn by the mod after translucent terrain, into the buffer it writes, with the pack's own camera, black at 40% as vanilla's; Horizon's own outline (drawn into its world pass, where it floated at an angle or didn't show) is skipped while a pack runs |
| Held translucent items | Stained glass and other translucent items in hand draw with `gbuffers_hand_water`, or `gbuffers_hand` as Iris falls back (they were a solid colour in packs without `gbuffers_hand_water`), blended over the scene, back faces culled as vanilla draws items, as in Iris (they were grainy through `gbuffers_hand`; Eclipse) |
| Block-breaking cracks | Drawn with `gbuffers_damagedblock` (or Iris's fallbacks: terrain, textured_lit, textured, basic), multiplied into the block under them as vanilla's are unless the pack sets its own blend. Each crack samples its own destroy stage's sprite (0.1.28 drew the whole atlas over the block), pulled toward the camera as vanilla's crumbling layer is (they flickered against the block) |
| Translucent particles | `gbuffers_particles_translucent` when the pack has it, after the translucent terrain |
| Rain depth | Rain and snow write depth only with `rain.depth=true`, as in Iris. A frame without weather keeps the pack's weather passes (they were lost after the first clear frame, and Horizon drew its own rain into the pack's buffers) |
| Block ids for floodfill | Flat terrain knows a face's sprite, not its block. Ids are now picked by sprite **and** by how the face glows, so an unlit furnace isn't the lit one and a pumpkin isn't a jack o'lantern. A face whose sprite blocks of other ids share is left out of voxels when the block's other faces place it (one block's faces carried two ids, and lights flickered with draw order). Redstone dust takes its power's id from its tint |
| Shader profiles | A profile's values are set, and the profile recognised again, even when written differently from the option's list (`0.5` for `0.50`) or not in it at all (Eclipse's `shadowMapResolution:3172`), as in Iris; such values stay chosen when the page opens again (the profile button worked once, then kept starting over) |
| Per-buffer blending | `blend.<program>.<buffer>` (Iris): one buffer's own blend, or `off`, apart from the program's. Packs keep their data buffers unblended under translucent geometry (Eclipse's colortex11, Pegasus's colortex1, Complementary's colortex4 and 8); Horizon blended every buffer alike |
| Iris's horizon | The sky program also draws Iris's horizon (`HorizonRenderer`): an octagonal prism around the camera from 16 blocks below it to 16 above, and a plane 16 below. Vanilla's sky disc ends 1.8 degrees above the horizon, and the space under it kept colortex0's clear colour (the fog colour) where packs that colour their sky by direction expect their sky: the band between distant terrain and the sky (Complementary, Rethinking Voxels, BSL) |
| `cloudHeight` | The clouds' altitude (it read 0: Complementary drew its clouds at the bottom of the world, and none in the sky) |
| Options in sizes | `image.*`, `bufferObject.*` and `size.buffer.*` read the pack's options as their values, as Iris preprocesses the whole file (Complementary's voxel images, sized by its colored lighting option) |
| Storage blocks in older GLSL | `buffer Name {` in a `#version 130` program with `GL_ARB_shader_storage_buffer_object` stays a storage block (Complementary's `composite` was dropped with colored lighting or voxel reflections on) |
| A variable named as its struct | `faceData faceData = ...;` (Complementary's voxel reflections) |
| `fogColor`, `skyColor` undeclared | Declared as Iris's uniforms when a program uses them without a declaration (Luster) |
| Narrowed outputs | The alpha test reads a narrowed location-0 output (`vec3 shadowcolor0_out`) by its name (Photon's shadow programs were dropped) |
| Atomics whose old value is used | `if ((imageAtomicOr(img, c, bit) >> 16 & 1) == 0)`: the old value is read before the statement and the atomic done there (Rethinking Voxels' shadow pass, which voxelises the world for its block lights, failed to build) |
| Images passed to functions | `void update_lpv(writeonly image3D img, ...)`: the parameter's memory qualifier is read, and each function is copied for the images its calls pass, as naga stores only to a global image (Photon's colored lights: `setup` and `shadowcomp` were dropped) |
| Float array indices | `table[clamp(i, 0u, 47u)]` with an `int i`: GLSL takes the `uint` overload, naga's frontend the float one; the index is made an `int` again (Photon's colored lights) |
| Stray `;` at the top level | After a function's body or another `;`, as GL drivers take it |
| GPU errors | Logged once each instead of ending the game |

## Known issues

Reported in the game:

- Block outline: not showing in the game, though it draws in the test setup. The log says what
  it draws into, and since 0.1.31 where its first corner lands on the screen (`block outline's
  first corner ... lands at`).
- Eclipse: smoke and maybe other particles reflect light.
- Pegasus: the water simulation looks frame-rate limited, with a repeating grid.
- Reverie: static noise, flat untextured water, a flat sky, no light from blocks, jittery leaves.
- Solas: light flickers on blocks beside a light source; the night sky turns white.
- Sildur's: water reflections white.
- RedHat: the water warps the view, with a flat ring that follows the camera.
- Alpha Piscium: transparent terrain and flat water. It needs subgroup operations
  (`subgroupElect`, `subgroupMin`...) that the GLSL frontend doesn't have yet, plus
  `SEPARATE_HARDWARE_SAMPLERS` and `HIGHER_SHADOWCOLOR`.
- Nostalgia 5.1: renders, then the screen goes black and the game crashes (the GPU driver
  resets).
- Luster: a black world. It renders its lighting into colortex16 to colortex21, and Horizon has
  16 colour buffers (colortex0 to colortex15); that limit is Horizon's and not one a mod can lift.
- Just Colored Lighting: caves see-through, no shader effects.

## Not the mod's

Reported along with it, and not caused by it:

- `horizon-launcher: unknown option` on start: the game was started with the launcher's
  arguments. Start it from the launcher.
- Out of memory while the game starts, before any shader pack loads (a 2 GB GPU with 32x
  textures and Fresh Animations): the resource packs' textures don't fit. Fewer or smaller
  resource packs fix it.
- `no shaders folder at the root of the pack`: the zip holds a folder that holds `shaders/`.
  Zip the pack so that `shaders/` is at the top, or put the unzipped folder in `shaderpacks/`.

## Pack status

Tested under Mesa's software Vulkan driver over a test scene, with each pack's default settings:

| Renders | Builds but renders wrong |
|---|---|
| Eclipse, Bliss, BSL, Complementary, Solas, Sildur's, Pegasus, Rethinking Voxels, Just Colored Lighting, Mellow, RedHat, OPAL | Reverie (dark), Noble (white), Luster, Nostalgia, Photon (black) |

In the game (reported on an RTX 4080 SUPER with 0.1.28): Photon and Pegasus render well apart
from held items; RedHat renders better than before; Nostalgia renders shaders and lighting before
it crashes; Luster's large white splotch is gone.

A pack that renders here can still have problems in the game on a real GPU. Reports of any pack,
listed or not, are welcome: a screenshot, the pack's name and version, and the game log help the
most.

## Limits

- Geometry and tessellation stages that add geometry can't run on wgpu: a pack feature built on
  them (Eclipse's shader grass) stays off.
- Testing: applied with `horizon-mixins apply` to pristine Horizon `10b16c4`, the whole game
  compiled with `--locked` and no warnings; Horizon's renderer tests and the mod's own pass, 813 in all (the two `crash::`
  tests fail without the mod too); of the shader crate's 93 tests, one fails by design
  (`the_alpha_test_discards_below_its_reference`: Iris tests alpha with declared outputs too).

## How it is built

The mod's changes are made in place on Horizon's branch `iris-dev` (`reference.patch`; the
changed and translator files in `ref/`). `tools/build.py` copies each changed function into a
module of the mod beside the original and hooks the original to call the copy;
`tools/build_tests.py` writes the test harness as a bundle of its own; `tools/stage.sh` applies
the bundles to a pristine export; `tools/package.py` writes `dist/`.

## Licence

The mod's code is MIT, like Horizon. Shader packs belong to their authors and keep their own
licences: the mod contains none of them, and you supply your own copies.
