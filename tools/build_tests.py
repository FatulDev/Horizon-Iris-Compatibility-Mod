#!/usr/bin/env python3
"""A local-only bundle with the GPU test of the mod (runs a real pack on a software Vulkan)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rsx import take, sub1, item_span

ROOT = os.path.dirname(HERE)
TERRAIN = r"""            // `IRIS_TEST_TERRAIN`: a block ground and a few pillars, drawn as the game's sections.
            let terrain_mesh = std::env::var_os("IRIS_TEST_TERRAIN").map(|_| {
                let mut v: Vec<TerrainVertex> = Vec::new();
                let mut idx: Vec<u32> = Vec::new();
                // One-sided quads facing outward (`IRIS_TEST_SINGLE`), as the game meshes them,
                // else both sides.
                fn quad(v: &mut Vec<TerrainVertex>, idx: &mut Vec<u32>, p: [[f32; 3]; 4], shade: u8, layer: u16) {
                    let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
                    let single = std::env::var_os("IRIS_TEST_SINGLE").is_some();
                    let windings: &[[usize; 4]] = if single { &[[0, 1, 2, 3]] } else { &[[0, 1, 2, 3], [0, 3, 2, 1]] };
                    for winding in windings {
                        let base = v.len() as u32;
                        for &k in winding {
                            v.push(TerrainVertex::new(p[k], uvs[k], layer, u32::from_le_bytes([shade, shade, shade, 255]), 240));
                        }
                        idx.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                    }
                }
                for x in -64..64 {
                    for z in -64..64 {
                        let (x, z) = (x as f32, z as f32);
                        quad(&mut v, &mut idx, [[x, 64.0, z], [x, 64.0, z + 1.0], [x + 1.0, 64.0, z + 1.0], [x + 1.0, 64.0, z]], 255, 0);
                    }
                }
                // Far terrain: a ring of 4-block tiles out to 200 blocks, and a far ridge.
                for x in -50..50 {
                    for z in -50..50 {
                        if (-16..16).contains(&x) && (-16..16).contains(&z) {
                            continue;
                        }
                        let (x, z) = (x as f32 * 4.0, z as f32 * 4.0);
                        quad(&mut v, &mut idx, [[x, 64.0, z], [x, 64.0, z + 4.0], [x + 4.0, 64.0, z + 4.0], [x + 4.0, 64.0, z]], 255, 0);
                    }
                }
                for k in -20..20 {
                    let x = k as f32 * 6.0;
                    let h = 20.0 + (k as f32 * 0.7).sin().abs() * 25.0;
                    let (z0, z1) = (150.0f32, 160.0f32);
                    quad(&mut v, &mut idx, [[x, 64.0, z0], [x + 6.0, 64.0, z0], [x + 6.0, 64.0 + h, z0], [x, 64.0 + h, z0]], 180, 1);
                    quad(&mut v, &mut idx, [[x, 64.0 + h, z0], [x + 6.0, 64.0 + h, z0], [x + 6.0, 64.0 + h, z1], [x, 64.0 + h, z1]], 230, 1);
                }
                for (k, (px, pz, h)) in [(5.0f32, 20.0f32, 8.0f32), (-10.0, 25.0, 14.0), (15.0, 32.0, 6.0), (0.0, 12.0, 3.0), (-20.0, 40.0, 20.0)].into_iter().enumerate() {
                    let first = v.len();
                    let (x0, x1, z0, z1, y0, y1) = (px, px + 3.0, pz, pz + 3.0, 64.0, 64.0 + h);
                    quad(&mut v, &mut idx, [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], 255, 1);
                    quad(&mut v, &mut idx, [[x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]], 200, 1);
                    quad(&mut v, &mut idx, [[x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [x0, y0, z1]], 200, 1);
                    quad(&mut v, &mut idx, [[x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]], 160, 1);
                    quad(&mut v, &mut idx, [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], 160, 1);
                    // HDR tier 2 in the colour's alpha (`mesher::emitting`) for the tier test.
                    if k < 2 && std::env::var("IRIS_TEST_GLOW").as_deref() == Ok("tier") {
                        for vertex in &mut v[first..] {
                            vertex.color = vertex.color & 0x00FF_FFFF | 0xFD00_0000;
                        }
                    }
                }
                // `IRIS_TEST_WATER`: a pool's surface (sprite layer 2, read as water: mc_Entity 8).
                let solid_count = idx.len() as u32;
                if std::env::var_os("IRIS_TEST_WATER").is_some() {
                    for x in -30..30 {
                        for z in 4..60 {
                            let (x, z) = (x as f32, z as f32);
                            quad(&mut v, &mut idx, [[x, 64.9, z], [x, 64.9, z + 1.0], [x + 1.0, 64.9, z + 1.0], [x + 1.0, 64.9, z]], 255, 2);
                        }
                    }
                }
                let cam = frame.camera;
                let origin = [-(cam.x as f32), -(cam.y as f32), -(cam.z as f32), 1.0 / crate::mesher::UV_STEPS];
                use wgpu::util::DeviceExt;
                let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX });
                let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&idx), usage: wgpu::BufferUsages::INDEX });
                let inst = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&origin), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST });
                (vb, ib, inst, idx.len() as u32, solid_count)
            });
"""
DRAW = r"""&mut |rp: &mut wgpu::RenderPass<'_>, list: TerrainList, _: &dyn Fn(&mut wgpu::RenderPass<'_>, u32)| {
                        if let Some((vb, ib, inst, n, solid)) = &terrain_mesh {
                            let range = match list {
                                TerrainList::Solid | TerrainList::ShadowSolid => 0..*solid,
                                _ => *solid..*n,
                            };
                            if !range.is_empty() {
                                rp.set_vertex_buffer(0, vb.slice(..));
                                rp.set_vertex_buffer(1, inst.slice(..));
                                rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                rp.draw_indexed(range, 0, 0..1);
                            }
                        }
                    },"""
src = open(os.path.join(ROOT, 'ref/crates/horizon-client-render/src/gfx/pack_post/iris.rs')).read()
s = src.index('#[cfg(test)]\nmod tests {')
e = src.index('#[cfg(test)]\nmod spv_tests {')
t = src[s:e]
t = sub1(t, '''        let macros = Macros::standard();
        let properties = Properties::parse(''', '''        let mut macros = Macros::standard();
        horizon_shaders::iris::define_iris(&mut macros);
        let properties = horizon_shaders::iris::parse_properties(''')
a = t.index('            // Storage buffer 4\'s members')
b = t.index('            let (_, lightmap) = view(')
t = t[:a] + t[b:]
t = sub1(t, 'PackPost::new(&device, &queue, post,', 'PackPost::iris_new(&device, &queue, post,')
t = sub1(t, '            | wgpu::Features::SUBGROUP_BARRIER;\n', '            | wgpu::Features::SUBGROUP_BARRIER\n            | wgpu::Features::TEXTURE_ATOMIC;\n')
t = sub1(t, '        println!("features: {features:?}");\n', '        println!("features: {features:?}");\n        horizon_shaders::iris::set_image_atomics(features.contains(wgpu::Features::TEXTURE_ATOMIC) && std::env::var_os("IRIS_TEST_NO_ATOMICS").is_none());\n')
t = sub1(t, '&& let Some(b) = pp.iris.as_ref().and_then(|i| i.buffer(4))',
         '&& let Some(b) = super::super::iris::state(&pp).and_then(|s| s.borrow().iris.buffer(4).cloned())')
t = sub1(t, 'let data = read_buffer(&device, &queue, b, 128);', 'let data = read_buffer(&device, &queue, &b, 128);')
t = sub1(t, '            let iris = pp.iris.as_ref().unwrap();',
         '            let st = super::super::iris::state(&pp).expect("registered");\n'
         '            let guard = st.borrow();\n            let iris = &guard.iris;')
a = t.index('                let all = read_buffer(&device, &queue, b, 1024);')
b = t.index('                let data = read_buffer(&device, &queue, b, 256);')
t = t[:a] + t[b:]
t = sub1(t, '''            if let Ok(out) = std::env::var("IRIS_TEST_DUMP") {
                dump_buffers(&device, &queue, &pp, &out, dim);
            }
''', '')
# Drop the buffer dump (render targets aren't readable outside Horizon's own test builds).
a = t.index('    /// Each colour buffer\'s "main" side')
b = t.index('    /// `IRIS_TEST_PACK=<pack folder>')
t = t[:a] + t[b:]
t = re.sub(r'\n    fn half\(.*?\n    }\n', '\n', t, flags=re.S)
t = re.sub(r'\n    fn small_float\(.*?\n    }\n', '\n', t, flags=re.S)

t = sub1(t, '            let frames: usize = std::env::var("IRIS_TEST_FRAMES")', TERRAIN + '            let frames: usize = std::env::var("IRIS_TEST_FRAMES")')
t = sub1(t, "&mut |_: &mut wgpu::RenderPass<'_>, _: TerrainList, _: &dyn Fn(&mut wgpu::RenderPass<'_>, u32)| {},", DRAW)
t = sub1(t, '                bob: glam::Mat4::IDENTITY,', '''                bob: if std::env::var_os("IRIS_TEST_BOB").is_some() {
                    glam::Mat4::from_translation(glam::Vec3::new(0.05, -0.05, 0.0))
                        * glam::Mat4::from_rotation_z(1.5f32.to_radians())
                        * glam::Mat4::from_rotation_x(2.5f32.to_radians())
                } else {
                    glam::Mat4::IDENTITY
                },''')
t = sub1(t, 'camera: glam::DVec3::new(0.5, 80.0, 0.5),', 'camera: glam::DVec3::new(0.5, std::env::var("IRIS_TEST_Y").ok().and_then(|v| v.parse().ok()).unwrap_or(80.0), 0.5),')
t = sub1(t, '                yaw: 30.0,\n                pitch: -10.0,', '                yaw: std::env::var("IRIS_TEST_YAW").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0),\n                pitch: std::env::var("IRIS_TEST_PITCH").ok().and_then(|v| v.parse().ok()).unwrap_or(-10.0),')

t = sub1(t, '            for f in 0..frames {\n', '''            // `IRIS_TEST_PROBE`: the floodfill diagnostics on (every 256 frames, to stdout too).
            if std::env::var_os("IRIS_TEST_PROBE").is_some() {
                super::super::super::iris_bridge::set_floodfill_probe(true);
                let _ = env_logger_probe();
            }
            let spin: f32 = std::env::var("IRIS_TEST_SPIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            for f in 0..frames {
                frame.yaw += spin;
                // `IRIS_TEST_WALK`: blocks per frame along +z, the terrain following the camera.
                if let Ok(w) = std::env::var("IRIS_TEST_WALK") {
                    frame.camera.z += w.parse::<f64>().unwrap_or(0.0);
                    if let Some((_, _, inst, _, _)) = &terrain_mesh {
                        let c = frame.camera;
                        let origin = [-(c.x as f32), -(c.y as f32), -(c.z as f32), 1.0 / crate::mesher::UV_STEPS];
                        queue.write_buffer(inst, 0, bytemuck::cast_slice(&origin));
                    }
                }
''')
t = sub1(t, '                terrain: Some(atlas.terrain_inputs(false)),', '''                terrain: Some({
                    let mut t = atlas.terrain_inputs(false);
                    if std::env::var_os("IRIS_TEST_WATER").is_some() {
                        for a in t.attributes.iter_mut().filter(|a| a.0 == "mc_Entity") {
                            a.1 = "vec4(hz_layer == 2u ? 8.0 : -1.0, hz_layer == 2u ? 1.0 : -1.0, 0.0, 0.0)".into();
                        }
                    }
                    t
                }),''')

FLAT = std = None
t = sub1(t, '            let world = World {\n', """            // `IRIS_TEST_FLAT`: terrain drawn as the game's flat draws (set 1: slots and origins;
            // set 3: the mc_Entity table and the page's vertices), sprite layer 2 as water.
            let flat = std::env::var_os("IRIS_TEST_FLAT").is_some();
            let storage = |binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let flat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[storage(0), storage(1)] });
            // The game's table (`client_block_ids.rs`): 16 redstone dust ids by power, then per
            // sprite layer Horizon's id and fluid flag, an id per HDR tier and the flags (bit t:
            // faces of tier t place their block in voxels).
            let mut id_table = vec![-1i32; 16 + 8 * 3];
            let mut set = |layer: usize, ids: [i32; 5], fluid: bool, flags: i32| {
                let at = 16 + 8 * layer;
                id_table[at] = ids[0];
                id_table[at + 1] = if fluid { 1 } else { -1 };
                id_table[at + 2..at + 7].copy_from_slice(&ids);
                id_table[at + 7] = flags;
            };
            set(0, [-1; 5], false, 31);
            set(1, [-1; 5], false, 31);
            set(2, [8; 5], true, 31);
            // `IRIS_TEST_GLOW`: the pillars (sprite layer 1) are glowstone (Eclipse's 196);
            // `IRIS_TEST_GLOW=tier`: only the first two shine (HDR tier 2), and the others, a
            // block sharing the sprite with another id, are left out of the voxels.
            match std::env::var("IRIS_TEST_GLOW").as_deref() {
                Ok("tier") => set(1, [-1, -1, 196, -1, -1], false, 1 << 2),
                Ok(_) => set(1, [196; 5], false, 31),
                Err(_) => {}
            }
            let world = World {
""")
t = sub1(t, '                terrain: Some(Geometry { buffers: &buffers, table: None, depth_write: true, atlas: None, ids: None, cull: true }),\n                block_ids: &[],',
         '                terrain: Some(Geometry { buffers: if flat { &buffers[..1] } else { &buffers }, table: flat.then_some(&flat_layout), depth_write: true, atlas: None, ids: None, cull: true }),\n                block_ids: if flat { &id_table } else { &[] },')
# inputs: flat terrain inputs
t = sub1(t, '                    let mut t = atlas.terrain_inputs(false);', '                    let mut t = atlas.terrain_inputs(std::env::var_os("IRIS_TEST_FLAT").is_some());')
t = sub1(t, '                    if std::env::var_os("IRIS_TEST_WATER").is_some() {\n                        for a in t.attributes',
            '                    if std::env::var_os("IRIS_TEST_WATER").is_some() && std::env::var_os("IRIS_TEST_FLAT").is_none() {\n                        for a in t.attributes')
# mesh: storage usage, slots + origins + group
t = sub1(t, "contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX });", "contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });")
t = sub1(t, "                (vb, ib, inst, idx.len() as u32, solid_count)\n", """                let slots = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&vec![0u32; v.len() / 256 + 1]), usage: wgpu::BufferUsages::STORAGE });
                let origins = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&origin), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST });
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &flat_layout, entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: slots.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: origins.as_entire_binding() },
                ] });
                (vb, ib, inst, idx.len() as u32, solid_count, group, origins)
""")
t = t.replace("if let Some((_, _, inst, _, _)) = &terrain_mesh {", "if let Some((_, _, inst, _, _, _, origins)) = &terrain_mesh {")
t = sub1(t, "                        queue.write_buffer(inst, 0, bytemuck::cast_slice(&origin));\n", "                        queue.write_buffer(inst, 0, bytemuck::cast_slice(&origin));\n                        queue.write_buffer(origins, 0, bytemuck::cast_slice(&origin));\n")
# run: pages + draw
t = sub1(t, "                    size,\n                    size,\n                    &[],\n", "                    size,\n                    size,\n                    &pages,\n")
t = sub1(t, "                let mut encoder = device.create_command_encoder(&Default::default());\n                pp.run(", "                let pages: Vec<(u32, wgpu::Buffer)> = match (&terrain_mesh, flat) {\n                    (Some(m), true) => vec![(0, m.0.clone())],\n                    _ => Vec::new(),\n                };\n                let mut encoder = device.create_command_encoder(&Default::default());\n                pp.run(")
t = sub1(t, "&mut |rp: &mut wgpu::RenderPass<'_>, list: TerrainList, _: &dyn Fn(&mut wgpu::RenderPass<'_>, u32)| {\n                        if let Some((vb, ib, inst, n, solid)) = &terrain_mesh {",
         "&mut |rp: &mut wgpu::RenderPass<'_>, list: TerrainList, page: &dyn Fn(&mut wgpu::RenderPass<'_>, u32)| {\n                        if let Some((vb, ib, inst, n, solid, group, _)) = &terrain_mesh {")
t = sub1(t, "                                rp.set_vertex_buffer(1, inst.slice(..));\n", "                                if flat {\n                                    rp.set_bind_group(1, group, &[]);\n                                    page(rp, 0);\n                                } else {\n                                    rp.set_vertex_buffer(1, inst.slice(..));\n                                }\n")
t = sub1(t, 'let pack_inputs = PackInputs { day_time, eye_light: [0, 15], brightness: 0.5, ..Default::default() };',
         'let snow = std::env::var_os("IRIS_TEST_SNOW").is_some();\n            let pack_inputs = PackInputs { day_time, eye_light: [0, 15], brightness: 0.5, rain: if snow || std::env::var_os("IRIS_TEST_RAIN").is_some() { 1.0 } else { 0.0 }, biome_precipitation: if snow { 2 } else if std::env::var_os("IRIS_TEST_RAIN").is_some() { 1 } else { 0 }, ..Default::default() };')
# `IRIS_TEST_PARTICLES`: a wall of particles 4 blocks ahead (yaw 0 looks along +z), the left
# half opaque, the right half on the translucent layer.
t = sub1(t, "                let pages: Vec<(u32, wgpu::Buffer)> = match (&terrain_mesh, flat) {", """                if std::env::var_os("IRIS_TEST_PARTICLES").is_some() {
                    let mut verts: Vec<crate::gfx::Vertex> = Vec::new();
                    for i in -6..6 {
                        for j in -3..3 {
                            let c = glam::Vec3::new(i as f32 * 0.5 + 0.25, j as f32 * 0.5 + 0.25, 4.0);
                            let layer = if i >= 0 { crate::gfx::TRANSLUCENT_PARTICLE_OVERLAY } else { 0 };
                            for (n, uv) in [((1.0f32, -1.0f32), [1.0f32, 1.0f32]), ((1.0, 1.0), [1.0, 0.0]), ((-1.0, 1.0), [0.0, 0.0]), ((-1.0, -1.0), [0.0, 1.0])] {
                                let p = c + glam::Vec3::new(-n.0, n.1, 0.0) * 0.2;
                                verts.push(crate::gfx::Vertex { normal: [0; 4], pos: p.to_array(), uv, layer, color: 0xffff_ffff, light: 240 | 240 << 8, glint_uv: [0.0, 0.0] });
                            }
                        }
                    }
                    let spans = [crate::gfx::pack_draw::EntitySpan { start: 0, stage: horizon_shaders::render_stage("PARTICLES"), ..Default::default() }];
                    let mut rest = Vec::new();
                    pp.prepare_entities(&device, &queue, &verts, &spans, &mut rest);
                    if f == 0 {
                        println!("  particles: {} vertices, {} left to Horizon", verts.len(), rest.len());
                    }
                }
                let pages: Vec<(u32, wgpu::Buffer)> = match (&terrain_mesh, flat) {""")
# `IRIS_TEST_RAIN`: curtains of rain 3 to 12 blocks ahead (yaw 0 looks along +z), with a streaky
# sprite.
t = sub1(t, "                let pages: Vec<(u32, wgpu::Buffer)> = match (&terrain_mesh, flat) {", """                // `IRIS_TEST_RAIN_FROM=<frame>`: clear weather before that frame, as when a world
                // loads in clear weather and it starts raining later.
                let rain_from: usize = std::env::var("IRIS_TEST_RAIN_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                if f < rain_from {
                    pp.set_weather(None);
                }
                if std::env::var_os("IRIS_TEST_RAIN").is_some() && f >= rain_from {
                    use wgpu::util::DeviceExt;
                    // The game's own columns (WeatherEffectRenderer), radius 10, from y 65 up.
                    let cam = [frame.camera.x, frame.camera.y, frame.camera.z];
                    let (cx, cy, cz) = (cam[0].floor() as i32, cam[1].floor() as i32, cam[2].floor() as i32);
                    let radius = 10;
                    let time = crate::weather::Time::new(f as i64 * 2, 0.0);
                    let mut verts: Vec<crate::gfx::Vertex> = Vec::new();
                    for z in cz - radius..=cz + radius {
                        for x in cx - radius..=cx + radius {
                            let column = crate::weather::Column { x, z, bottom_y: 65, top_y: cy + radius, u_offset: 0.0, v_offset: 0.0, light_coords: 15 << 20 };
                            let vs = crate::weather::vertices(&column, crate::weather::Parameters::new(x, z), false, cam, radius, 1.0, [0.0, 0.0, 1.0, 1.0]);
                            for v in &vs {
                                let [du, dv] = super::super::super::pack_weather::phase(v, time);
                                verts.push(crate::gfx::Vertex { normal: [0; 4], pos: v.pos, uv: [v.uv[0] + du, v.uv[1] + dv], layer: 0, color: v.color, light: v.light, glint_uv: [0.0, 0.0] });
                            }
                        }
                    }
                    let quads = verts.len() as u32 / 4;
                    let indices: Vec<u32> = (0..quads).flat_map(|q| [0, 1, 2, 0, 2, 3].map(|i| q * 4 + i)).collect();
                    let sprite = device.create_texture_with_data(&queue, &wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: 16, height: 64, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    }, wgpu::util::TextureDataOrder::LayerMajor, &(0..16 * 64).flat_map(|n| {
                        // Streaks a texel wide, like vanilla's rain.png.
                        let (x, y) = (n % 16, n / 16);
                        let streak = [(2, 3, 14), (7, 20, 30), (12, 40, 52), (4, 44, 58), (10, 8, 16)].iter().any(|&(sx, y0, y1)| x == sx && (y0..y1).contains(&y));
                        if streak { [200u8, 215, 255, 170] } else { [0, 0, 0, 0] }
                    }).collect::<Vec<u8>>());
                    pp.set_weather(Some(super::super::Weather {
                        vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX }),
                        indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&indices), usage: wgpu::BufferUsages::INDEX }),
                        quads: [0..quads, quads..quads],
                        sprites: [Some(sprite.create_view(&Default::default())).filter(|_| std::env::var_os("IRIS_TEST_NOSPRITE").is_none()), None],
                        generation: 1,
                    }));
                }
                let pages: Vec<(u32, wgpu::Buffer)> = match (&terrain_mesh, flat) {""")
# `IRIS_TEST_LPV_EACH`: each frame, the voxels holding a solid block or anything else, and the
# total light in each floodfill volume (a flicker shows as numbers that jump between frames).
t = sub1(t, """                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
""", """                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if std::env::var_os("IRIS_TEST_LPV_EACH").is_some() {
                    let st = super::super::iris::state(&pp).expect("registered");
                    let guard = st.borrow();
                    let mut line = format!("  lpv frame {f}:");
                    for image in guard.iris.images.iter().filter(|i| i.decl.name.starts_with("imgLpv") || i.decl.name == "imgVoxelMask") {
                        let Some((texture, _)) = &image.texture else { continue };
                        let bpt = texture.format().block_copy_size(None).unwrap_or(4);
                        let data = read_texture(&device, &queue, texture, bpt);
                        if image.decl.name == "imgVoxelMask" {
                            let (mut solid, mut other) = (0usize, 0usize);
                            for c in data.chunks_exact(bpt as usize) {
                                match u32::from(c[0]) | u32::from(c[1]) << 8 {
                                    0 => {}
                                    1 => solid += 1,
                                    _ => other += 1,
                                }
                            }
                            line += &format!(" voxels solid {solid} other {other};");
                        } else {
                            let sum: u64 = data.chunks_exact(4).map(|c| u64::from(c[0]) + u64::from(c[1]) + u64::from(c[2])).sum();
                            line += &format!(" {} {sum};", image.decl.name);
                        }
                    }
                    println!("{line}");
                }
            }
""")
# The floodfill volumes: how many voxels hold a block or light, and the most common ids.
t = sub1(t, "            let picture = read_texture(&device, &queue, &screen_texture, 4);\n", """            for image in iris.images.iter().filter(|i| i.decl.name.starts_with("imgLpv") || i.decl.name == "imgVoxelMask") {
                let Some((texture, _)) = &image.texture else { continue };
                let bpt = texture.format().block_copy_size(None).unwrap_or(4);
                let data = read_texture(&device, &queue, texture, bpt);
                let mut counts: std::collections::BTreeMap<Vec<u8>, usize> = std::collections::BTreeMap::new();
                for c in data.chunks_exact(bpt as usize) {
                    *counts.entry(c.to_vec()).or_default() += 1;
                }
                let mut top: Vec<_> = counts.into_iter().collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                println!("  {} ({:?} {:?}): {:?}", image.decl.name, texture.format(), texture.size(), top.iter().take(8).collect::<Vec<_>>());
                if std::env::var_os("IRIS_TEST_LPV_WHERE").is_some() {
                    let n = texture.size().width as usize;
                    let at = |i: usize| (i % n, (i / n) % n, i / (n * n));
                    if image.decl.name == "imgVoxelMask" {
                        let glow: Vec<_> = data.chunks_exact(bpt as usize).enumerate().filter(|(_, c)| c[0] == 196).map(|(i, _)| at(i)).take(12).collect();
                        println!("    glowstone voxels (x, y, z): {glow:?}");
                    } else {
                        let mut lit: Vec<_> = data.chunks_exact(4).enumerate().map(|(i, c)| (c[0] as u32 + c[1] as u32 + c[2] as u32, at(i))).collect();
                        lit.sort_by(|a, b| b.0.cmp(&a.0));
                        println!("    brightest (sum, (x, y, z)): {:?}", &lit[..8]);
                    }
                }
            }
            let picture = read_texture(&device, &queue, &screen_texture, 4);
""")
# `IRIS_TEST_COLORS`: the ground green, the pillars red, the water blue.
t = sub1(t, "                    &[200u8; 2 * 2 * 3 * 4],\n", """                    &if std::env::var_os("IRIS_TEST_COLORS").is_some() {
                        [[60u8, 150, 50, 255], [190, 40, 35, 255], [60, 90, 200, 255]].iter().flat_map(|c| std::iter::repeat_n(*c, 4).flatten()).collect::<Vec<u8>>()
                    } else {
                        vec![200u8; 2 * 2 * 3 * 4]
                    },
""")
# `IRIS_TEST_EMITTERS`: the block ids the pack's setup program gives a light range.
t = sub1(t, "                    println!(\"  image {k} imgBlockData: {filled} of {} entries written by setup\", data.len() / 8);\n", """                    println!("  image {k} imgBlockData: {filled} of {} entries written by setup", data.len() / 8);
                    if std::env::var_os("IRIS_TEST_EMITTERS").is_some() {
                        let lit: Vec<(usize, u8, [u8; 3])> = data.chunks_exact(8).enumerate().filter(|(_, c)| c[3] != 0).map(|(i, c)| (i, c[3], [c[0], c[1], c[2]])).collect();
                        println!("  emitters (id, range, rgb): {lit:?}");
                    }
""")
t = t.rstrip()
assert t.endswith('}')
t = t[:-1] + '''
    /// Prints `log` records (the floodfill probe's) to stdout.
    fn env_logger_probe() -> Result<(), log::SetLoggerError> {
        struct Out;
        impl log::Log for Out {
            fn enabled(&self, _: &log::Metadata<'_>) -> bool { true }
            fn log(&self, r: &log::Record<'_>) {
                if r.args().to_string().starts_with("floodfill probe") {
                    println!("  {}", r.args());
                }
            }
            fn flush(&self) {}
        }
        log::set_logger(&Out).map(|()| log::set_max_level(log::LevelFilter::Info))
    }
}
'''
out = os.path.join(ROOT, 'out', 'iris-features-tests')
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'tests.rs'), 'w').write(t)
HORIZON = re.search(r"^HORIZON = '([^']+)'", open(os.path.join(HERE, 'build.py')).read(), re.M).group(1)
open(os.path.join(out, 'horizon-mixin.toml'), 'w').write(f'''[mod]
id = "iris-features-tests"
version = "0.1.0"
priority = 100

[compatibility]
horizon_revision = "{HORIZON}"

[[module]]
id = "tests"
file = "crates/horizon-client-render/src/gfx/pack_post.rs"
name = "iris_tests"
source = "tests.rs"
''')
print('test bundle written')
