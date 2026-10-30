
/// `mc_Entity`'s table for flat draws, which know a face's sprite layer and the HDR tier its
/// colour carries (how brightly its block state glows, 0 when it does not) but not its block.
/// Horizon gives each layer the id most of its blocks have; a pack that voxelises terrain in its
/// shadow pass (Eclipse's floodfill) then gets one block's faces with different ids where blocks
/// share sprites (a furnace's sides are the lit furnace's too), and whichever face the GPU wrote
/// last decided what the block was: lights flickered, an unlit furnace or pumpkin shone.
///
/// The table, in `i32`s: 16 ids of `redstone_wire` by power, then 8 per sprite layer: Horizon's
/// pair (its id, 1 for fluids), the id for each of the 5 tiers, and flags: bit `t` set when faces
/// of tier `t` place their block in the voxels (`at_midBlock`), `IRIS_WIRE` when the layer is
/// redstone dust tinted by its power (`iris_atlas` reads the power back from the tint).
pub(crate) const IRIS_WIRE: i32 = 32;

/// HDR tiers a vertex colour carries (`mesher::emitting`): 0 to `VERY_BRIGHT`.
pub(crate) const IRIS_TIERS: usize = 5;

/// A block state as the table sees it.
pub(crate) struct IrisState<'a> {
    /// `minecraft:furnace`.
    pub name: &'a str,
    /// Its id in `block.properties`, -1 without one.
    pub id: i32,
    /// `redstone_wire`'s power.
    pub wire_power: Option<usize>,
    /// Its faces: (sprite layer, HDR tier, tinted).
    pub faces: Vec<(u32, u8, bool)>,
}

#[derive(Default, Debug)]
pub(crate) struct IrisIdStats {
    /// (layer, tier) uses whose blocks have different ids.
    pub mixed: usize,
    /// Of them, those left out of the voxels: each of their blocks has a face that places it.
    pub left_out: Vec<String>,
    pub wire_layers: usize,
}

/// The table for `states`, over Horizon's (id, fluid) pairs by layer; `layer_name` gives a
/// layer's sprite (`block/furnace_side`).
pub(crate) fn iris_id_table<'n>(
    states: &[IrisState<'_>],
    layer_name: impl Fn(u32) -> Option<&'n str>,
    pairs: &[i32],
) -> (Vec<i32>, IrisIdStats) {
    use std::collections::{BTreeMap, BTreeSet};
    let layers = pairs.len() / 2;
    let mut table = vec![-1; 16 + 8 * layers];
    for l in 0..layers {
        let at = 16 + 8 * l;
        table[at] = pairs[2 * l];
        table[at + 1] = pairs[2 * l + 1];
        table[at + 2..at + 2 + IRIS_TIERS].fill(pairs[2 * l]);
        table[at + 7] = (1 << IRIS_TIERS) - 1;
    }
    let mut stats = IrisIdStats::default();
    let fluid = |l: u32| pairs.get(2 * l as usize + 1) == Some(&1);
    // (layer, tier) -> id -> (a block named like the sprite, the blocks).
    let mut slots: BTreeMap<(u32, u8), BTreeMap<i32, (bool, BTreeSet<&str>)>> = BTreeMap::new();
    // The (layer, tier) uses of each state, and the states of each use.
    let mut uses: Vec<BTreeSet<(u32, u8)>> = Vec::with_capacity(states.len());
    let mut users: BTreeMap<(u32, u8), Vec<usize>> = BTreeMap::new();
    // Per layer: (every face on it is redstone dust's, every one tinted).
    let mut wire: BTreeMap<u32, (bool, bool)> = BTreeMap::new();
    for (k, s) in states.iter().enumerate() {
        let block = s.name.strip_prefix("minecraft:").unwrap_or(s.name);
        if let Some(power) = s.wire_power
            && power < 16
            && table[power] == -1
        {
            table[power] = s.id;
        }
        let mut used = BTreeSet::new();
        for &(layer, tier, tinted) in &s.faces {
            if layer as usize >= layers || fluid(layer) {
                continue;
            }
            let tier = tier.min(IRIS_TIERS as u8 - 1);
            let named = layer_name(layer).and_then(|n| n.strip_prefix("block/")) == Some(block);
            let v = slots.entry((layer, tier)).or_default().entry(s.id).or_default();
            v.0 |= named;
            v.1.insert(s.name);
            if used.insert((layer, tier)) {
                users.entry((layer, tier)).or_default().push(k);
            }
            let w = wire.entry(layer).or_insert((true, true));
            w.0 &= s.wire_power.is_some();
            w.1 &= tinted;
        }
        uses.push(used);
    }
    let wire_layer = |l: u32| wire.get(&l).is_some_and(|&(w, t)| w && t);
    // A use places its blocks rightly when they all have its id, or when it is tinted dust.
    let placed = |slot: &(u32, u8)| wire_layer(slot.0) || slots.get(slot).is_some_and(|v| v.len() == 1);
    for (&(layer, tier), votes) in &slots {
        let at = 16 + 8 * layer as usize;
        let best = votes
            .iter()
            .max_by_key(|(id, (named, blocks))| (*named, blocks.len(), -**id))
            .map(|(id, _)| *id);
        if let Some(best) = best {
            table[at + 2 + tier as usize] = best;
        }
        if votes.len() > 1 && !wire_layer(layer) {
            stats.mixed += 1;
            // Left out when each of its blocks has another face that places it; else kept with
            // the commonest id (a stair of a block's sprite is better as that block than absent).
            let of = users.get(&(layer, tier)).map_or(&[][..], Vec::as_slice);
            let covered = of
                .iter()
                .all(|&k| uses[k].iter().any(|s| *s != (layer, tier) && placed(s)));
            if covered {
                table[at + 7] &= !(1 << tier);
                let blocks: BTreeSet<&str> = votes.values().flat_map(|v| v.1.iter().copied()).collect();
                let name = layer_name(layer).unwrap_or("?");
                stats.left_out.push(format!("{name} (tier {tier}, {} blocks)", blocks.len()));
            }
        }
    }
    for &layer in wire.keys().filter(|l| wire_layer(**l)) {
        table[16 + 8 * layer as usize + 7] |= IRIS_WIRE;
        stats.wire_layers += 1;
    }
    (table, stats)
}
