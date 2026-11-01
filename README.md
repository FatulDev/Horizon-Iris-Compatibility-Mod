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
   horizon-launcher native-build path/to/iris-features --trust-native-code
   ```

   Alternatively, copy `iris-features.hnmod` into your instance's `horizon-native-mods/` folder,
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
| Distant terrain | While a pack runs, distant faces are meshed a block each (grass sides stretched across far hills) |

## Not carried over from 0.1.25 yet

These worked on the old Horizon and still need porting onto Horizon's new pack renderer:

- Translucent held items with `gbuffers_hand_water` (stained glass in hand grainy on Eclipse).
- The block outline drawn after the pack's deferred programs (it can look reflective).
- Block ids by sprite and glow for floodfill (lights that flicker with some blocks).
- `gbuffers_particles_translucent`, rain depth with `rain.depth`, GPU errors logged instead of
  ending the game.

## Pack status

Tested under Mesa's software Vulkan driver over a test scene, with each pack's default settings:

| Renders | Builds but renders wrong |
|---|---|
| Eclipse, Bliss, BSL, Complementary, Solas, Sildur's, Pegasus, Rethinking Voxels, Just Colored Lighting | Reverie (dark), Noble (white), Luster, Nostalgia, Photon (black) |

A pack that renders here can still have problems in the game on a real GPU. Reports of any pack,
listed or not, are welcome: a screenshot, the pack's name and version, and the game log help the
most.

## Limits

- Geometry and tessellation stages that add geometry can't run on wgpu: a pack feature built on
  them (Eclipse's shader grass) stays off.
- Testing: applied with `horizon-mixins apply` to pristine Horizon `10b16c4`, the whole game
  compiled with `--locked` and no warnings; Horizon's 810 renderer tests pass (the two `crash::`
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
