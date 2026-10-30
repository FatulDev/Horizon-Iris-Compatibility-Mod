
/// `iris_id_table` for the game's block states: their ids, glow and the faces of their models.
pub(crate) fn iris_block_id_table(
    pack: &super::super::PackResources,
    ids: &BlockIds,
    pairs: &[i32],
) -> (Vec<i32>, IrisIdStats) {
    let registry = pack.assets.registry;
    let catalog = &pack.render.model_catalog;
    let tables = &pack.context.tables;
    let mut states = Vec::new();
    if !ids.is_empty() {
        for i in 0..registry.len() {
            let state_id = horizon_world::registry::BlockStateId(i as u32);
            let Some(state) = registry.state(state_id) else {
                continue;
            };
            let block = state.name.strip_prefix("minecraft:").unwrap_or(&state.name);
            let id = ids
                .id(&state.name, |k| state.properties.get(k).map(String::as_str))
                .unwrap_or(-1);
            let emission = tables.facts(state_id).map_or(0, |f| f.emission);
            let tier = horizon_client::mesher::hdr_tier(block, emission);
            let origin = horizon_core::position::BlockPos::new(0, 0, 0);
            let faces = catalog
                .collect(state_id, origin)
                .unwrap_or_default()
                .into_iter()
                .flat_map(|(_, model)| &model.faces)
                .map(|f| {
                    let glow = horizon_client::mesher::hdr_tier("", f.quad.light_emission);
                    (f.layer, tier.max(glow), f.quad.tint_index >= 0)
                })
                .collect();
            states.push(IrisState {
                name: &state.name,
                id,
                wire_power: (block == "redstone_wire")
                    .then(|| state.properties.get("power").and_then(|p| p.parse().ok()))
                    .flatten(),
                faces,
            });
        }
    }
    iris_id_table(&states, |l| pack.layer_names.get(&l).map(String::as_str), pairs)
}
