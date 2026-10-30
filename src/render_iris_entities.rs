// `PackEntities::prepare`, with the particles on the translucent layer drawn apart: Iris draws
// them with `gbuffers_particles_translucent` (after translucent terrain under `mixed`), the
// others with `gbuffers_particles`.
use super::*;

/// The kind of draw of the particles on the translucent layer (beside `ENTITY`..`PARTICLE`).
pub(crate) const PARTICLE_TRANSLUCENT: usize = PROGRAMS;

impl PackEntities {
    /// `prepare`; with `split`, each particle run becomes two draws, its opaque quads
    /// (`PARTICLE`) and those on the translucent layer (`PARTICLE_TRANSLUCENT`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn iris_prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        entities: &[Vertex],
        spans: &[EntitySpan],
        drawn: [bool; PROGRAMS],
        shadow: Casters,
        rest: &mut Vec<Vertex>,
        split: bool,
    ) {
        self.draws.clear();
        self.scratch.clear();
        self.uniforms.clear();
        rest.clear();
        let first = spans.first().map_or(entities.len(), |s| s.start as usize);
        rest.extend_from_slice(&entities[..first.min(entities.len())]);
        for (span, range) in runs(entities.len(), spans) {
            let run = &entities[range];
            let program = program_of(&span).filter(|_| plain(run));
            let pack = program.is_some_and(|p| drawn[p]) && !span.hidden;
            let casts = program.is_some_and(|p| shadow.cast(&span, p));
            if !pack && !span.hidden {
                rest.extend_from_slice(run);
            }
            let Some(program) = program.filter(|_| (pack || casts) && !run.is_empty()) else {
                continue;
            };
            let start = self.scratch.len();
            if program == PARTICLE && split && pack {
                let translucent =
                    |q: &[Vertex]| q[0].layer == crate::gfx::TRANSLUCENT_PARTICLE_OVERLAY;
                for q in run.chunks_exact(4).filter(|q| !translucent(q)) {
                    self.scratch.extend_from_slice(q);
                }
                let middle = self.scratch.len();
                for q in run.chunks_exact(4).filter(|q| translucent(q)) {
                    self.scratch.extend_from_slice(q);
                }
                let end = self.scratch.len();
                for (program, a, b) in [(PARTICLE, start, middle), (PARTICLE_TRANSLUCENT, middle, end)] {
                    if a < b {
                        self.iris_push(&span, program, a, b, pack, casts);
                    }
                }
            } else {
                self.scratch.extend_from_slice(run);
                let end = self.scratch.len();
                self.iris_push(&span, program, start, end, pack, casts);
            }
        }
        if self.draws.is_empty() {
            return;
        }
        let vertex_size = std::mem::size_of::<Vertex>() as u64;
        let (need_vertices, need_slots) = (self.scratch.len() as u64, self.draws.len() as u64);
        let have_vertices = self.vertices.size() / vertex_size;
        let have_slots = self.slots.size() / self.stride;
        if need_vertices > have_vertices {
            (self.vertices, self.table_group) =
                Self::vertex_buffer(device, &self.table, need_vertices.next_power_of_two());
        }
        if need_slots > have_slots {
            (self.slots, self.group) =
                Self::slot_buffer(device, layout, need_slots.next_power_of_two(), self.stride);
        }
        let quads = self.draws.iter().map(|d| d.quads).max().unwrap_or(0);
        if quads > self.quads {
            self.quads = quads.next_power_of_two();
            self.indices = quad_indices(device, self.quads);
        }
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&self.scratch));
        queue.write_buffer(&self.slots, 0, &self.uniforms);
    }

    /// One draw of `scratch[a..b]` (`span`'s), with its slot of `HzDraw`.
    fn iris_push(&mut self, span: &EntitySpan, program: usize, a: usize, b: usize, pack: bool, casts: bool) {
        let (entity_id, block_id) = match program {
            BLOCK => (0, span.id),
            _ => (span.id, 0),
        };
        let mut bytes = [0u8; SLOT as usize];
        bytes[0..4].copy_from_slice(&(span.stage as i32).to_le_bytes());
        bytes[4..8].copy_from_slice(&entity_id.to_le_bytes());
        bytes[8..12].copy_from_slice(&block_id.to_le_bytes());
        for (i, c) in entity_color(self.scratch[a].layer).iter().enumerate() {
            bytes[16 + 4 * i..20 + 4 * i].copy_from_slice(&c.to_le_bytes());
        }
        self.uniforms.extend_from_slice(&bytes);
        self.uniforms
            .resize(self.uniforms.len() + (self.stride - SLOT) as usize, 0);
        self.draws.push(Draw {
            base: a as u32,
            quads: (b - a) as u32 / 4,
            program,
            pack,
            shadow: casts && program != PARTICLE_TRANSLUCENT,
            slot: self.draws.len() as u32,
        });
    }
}
