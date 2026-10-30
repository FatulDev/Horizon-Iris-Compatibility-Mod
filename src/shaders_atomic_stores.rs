
/// Iris Features: a vertex stage's `imageStore` into a 32-bit integer image takes the largest
/// value instead of the last. Packs voxelise terrain in the shadow pass this way (Eclipse's
/// floodfill): several faces store into one voxel, and which came last changed from frame to
/// frame with the order the GPU ran them in, so a voxel was a light one frame and stone the
/// next. Vertex invocations have no order, so the largest is one of the results a pack can get
/// anyway, and the same one every frame. Only with the device's image atomics.
fn atomic_vertex_stores(module: &mut naga::Module) {
    use naga::{ImageClass, StorageAccess, StorageFormat, TypeInner};
    if !crate::iris::image_atomics() {
        return;
    }
    let targets: Vec<naga::Handle<naga::GlobalVariable>> = module
        .global_variables
        .iter()
        .filter(|(_, g)| {
            matches!(
                module.types[g.ty].inner,
                TypeInner::Image {
                    class: ImageClass::Storage {
                        format: StorageFormat::R32Uint | StorageFormat::R32Sint,
                        ..
                    },
                    ..
                }
            )
        })
        .map(|(h, _)| h)
        .collect();
    if targets.is_empty() {
        return;
    }
    let mut changed = Vec::new();
    for (_, f) in module.functions.iter_mut() {
        atomic_stores_in(&mut f.body, &mut f.expressions, &targets, &mut changed);
    }
    for ep in module.entry_points.iter_mut() {
        let f = &mut ep.function;
        atomic_stores_in(&mut f.body, &mut f.expressions, &targets, &mut changed);
    }
    for h in changed {
        let ty = module.global_variables[h].ty;
        let TypeInner::Image {
            dim,
            arrayed,
            class: ImageClass::Storage { format, .. },
        } = module.types[ty].inner
        else {
            continue;
        };
        let name = module.types[ty].name.clone();
        let access = StorageAccess::LOAD | StorageAccess::STORE | StorageAccess::ATOMIC;
        let new = module.types.insert(
            naga::Type {
                name,
                inner: TypeInner::Image {
                    dim,
                    arrayed,
                    class: ImageClass::Storage { format, access },
                },
            },
            naga::Span::UNDEFINED,
        );
        module.global_variables[h].ty = new;
    }
}

fn atomic_stores_in(
    block: &mut naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    targets: &[naga::Handle<naga::GlobalVariable>],
    changed: &mut Vec<naga::Handle<naga::GlobalVariable>>,
) {
    use naga::{Expression, Statement};
    for stmt in block.iter_mut() {
        match stmt {
            Statement::Block(b) => atomic_stores_in(b, expressions, targets, changed),
            Statement::If { accept, reject, .. } => {
                atomic_stores_in(accept, expressions, targets, changed);
                atomic_stores_in(reject, expressions, targets, changed);
            }
            Statement::Switch { cases, .. } => {
                for c in cases.iter_mut() {
                    atomic_stores_in(&mut c.body, expressions, targets, changed);
                }
            }
            Statement::Loop { body, continuing, .. } => {
                atomic_stores_in(body, expressions, targets, changed);
                atomic_stores_in(continuing, expressions, targets, changed);
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                let Expression::GlobalVariable(g) = expressions[*image] else {
                    continue;
                };
                if !targets.contains(&g) {
                    continue;
                }
                let x = expressions.append(
                    Expression::AccessIndex {
                        base: *value,
                        index: 0,
                    },
                    naga::Span::UNDEFINED,
                );
                let mut b = naga::Block::new();
                b.push(
                    Statement::Emit(naga::Range::new_from_bounds(x, x)),
                    naga::Span::UNDEFINED,
                );
                b.push(
                    Statement::ImageAtomic {
                        image: *image,
                        coordinate: *coordinate,
                        array_index: *array_index,
                        fun: naga::AtomicFunction::Max,
                        value: x,
                    },
                    naga::Span::UNDEFINED,
                );
                *stmt = Statement::Block(b);
                if !changed.contains(&g) {
                    changed.push(g);
                }
            }
            _ => {}
        }
    }
}
