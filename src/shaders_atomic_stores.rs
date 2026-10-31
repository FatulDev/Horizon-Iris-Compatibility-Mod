
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

/// Iris Features: GLSL's `barrier()` waits for the whole workgroup. naga's GLSL frontend reads
/// it as every kind of barrier, subgroups included, and its SPIR-V writer then gives the
/// barrier a subgroup execution scope: only 32 invocations waited for each other, the rest of
/// the workgroup raced on its shared memory (Eclipse's floodfill propagation came out a little
/// different each frame on NVIDIA, and its light flickered). The workgroup barrier it means.
fn workgroup_barriers(module: &mut naga::Module) {
    fn fix(block: &mut naga::Block) {
        use naga::Statement;
        for stmt in block.iter_mut() {
            match stmt {
                Statement::ControlBarrier(flags) | Statement::MemoryBarrier(flags)
                    if flags.contains(naga::Barrier::SUB_GROUP) =>
                {
                    flags.remove(naga::Barrier::SUB_GROUP);
                    flags.insert(naga::Barrier::WORK_GROUP);
                }
                Statement::Block(b) => fix(b),
                Statement::If { accept, reject, .. } => {
                    fix(accept);
                    fix(reject);
                }
                Statement::Switch { cases, .. } => {
                    for c in cases.iter_mut() {
                        fix(&mut c.body);
                    }
                }
                Statement::Loop { body, continuing, .. } => {
                    fix(body);
                    fix(continuing);
                }
                _ => {}
            }
        }
    }
    for (_, f) in module.functions.iter_mut() {
        fix(&mut f.body);
    }
    for ep in module.entry_points.iter_mut() {
        fix(&mut ep.function.body);
    }
}

/// Iris Features: GLSL's `ivec2 % int` (Luster's `dstTexel % cloudsTemporalUpscaling`). naga's
/// GLSL frontend widens the scalar of every other arithmetic operator to the vector, not the
/// one of `%`, and the module then fails validation. A literal or constant scalar there becomes
/// the vector constant it widens to, in place.
fn vector_modulo(module: &mut naga::Module) {
    use naga::{BinaryOperator, Expression, TypeInner};
    // (function, entry point index), the scalar operand, the vector's size.
    let mut fixes: Vec<(Option<naga::Handle<naga::Function>>, Option<usize>, naga::Handle<Expression>, naga::VectorSize)> =
        Vec::new();
    {
        let module_ref = &*module;
        let mut scan = |f: &naga::Function, fh, ei| {
            let ctx = naga::proc::ResolveContext::with_locals(module_ref, &f.local_variables, &f.arguments);
            let mut typifier = naga::front::Typifier::new();
            let Some((last, _)) = f.expressions.iter().last() else { return };
            if typifier.grow(last, &f.expressions, &ctx).is_err() {
                return;
            }
            for (_, e) in f.expressions.iter() {
                let Expression::Binary { op: BinaryOperator::Modulo, left, right } = *e else { continue };
                let (l, r) = (typifier.get(left, &module_ref.types), typifier.get(right, &module_ref.types));
                let (scalar, size) = match (l, r) {
                    (TypeInner::Vector { size, .. }, TypeInner::Scalar(_)) => (right, *size),
                    (TypeInner::Scalar(_), TypeInner::Vector { size, .. }) => (left, *size),
                    _ => continue,
                };
                if matches!(f.expressions[scalar], Expression::Literal(_) | Expression::Constant(_)) {
                    fixes.push((fh, ei, scalar, size));
                }
            }
        };
        for (h, f) in module_ref.functions.iter() {
            scan(f, Some(h), None);
        }
        for (i, ep) in module_ref.entry_points.iter().enumerate() {
            scan(&ep.function, None, Some(i));
        }
    }
    for (fh, ei, scalar, size) in fixes {
        let f = match (fh, ei) {
            (Some(h), _) => &mut module.functions[h],
            (_, Some(i)) => &mut module.entry_points[i].function,
            _ => continue,
        };
        let (value, kind) = match f.expressions[scalar] {
            Expression::Literal(lit) => {
                let s = lit.scalar();
                (module.global_expressions.append(Expression::Literal(lit), naga::Span::UNDEFINED), s)
            }
            Expression::Constant(c) => {
                let TypeInner::Scalar(s) = module.types[module.constants[c].ty].inner else { continue };
                (module.constants[c].init, s)
            }
            _ => continue,
        };
        let splat = module
            .global_expressions
            .append(Expression::Splat { size, value }, naga::Span::UNDEFINED);
        let ty = module.types.insert(
            naga::Type { name: None, inner: TypeInner::Vector { size, scalar: kind } },
            naga::Span::UNDEFINED,
        );
        let c = module.constants.append(
            naga::Constant { name: None, ty, init: splat },
            naga::Span::UNDEFINED,
        );
        let f = match (fh, ei) {
            (Some(h), _) => &mut module.functions[h],
            (_, Some(i)) => &mut module.entry_points[i].function,
            _ => continue,
        };
        f.expressions[scalar] = Expression::Constant(c);
    }
}

/// Iris Features: naga's GLSL frontend lays out every struct as `std140`, so a member
/// `float b[3]` has a stride of 16 while the `float[3](...)` built for it has 4, and
/// constructing the struct fails validation (Luster's ACES tables). A struct only the shader
/// itself holds (no buffer or uniform, no other type containing it) takes its arrays with
/// their own stride, as the shader's other arrays have.
fn private_struct_strides(module: &mut naga::Module) {
    use naga::{AddressSpace, Expression, Handle, Type, TypeInner};
    use std::collections::HashMap;
    let mut layouter = naga::proc::Layouter::default();
    if layouter.update(module.to_ctx()).is_err() {
        return;
    }
    let mut kept: Vec<Handle<Type>> = Vec::new();
    for (_, t) in module.types.iter() {
        match t.inner {
            TypeInner::Array { base, .. } | TypeInner::Pointer { base, .. } | TypeInner::BindingArray { base, .. } => kept.push(base),
            TypeInner::Struct { ref members, .. } => kept.extend(members.iter().map(|m| m.ty)),
            _ => {}
        }
    }
    for (_, g) in module.global_variables.iter() {
        if !matches!(g.space, AddressSpace::Private | AddressSpace::Function | AddressSpace::WorkGroup) {
            kept.push(g.ty);
        }
    }
    let candidates: Vec<(Handle<Type>, Type)> = module
        .types
        .iter()
        .filter(|(h, _)| !kept.contains(h))
        .filter_map(|(h, t)| match t.inner {
            TypeInner::Struct { ref members, .. } => members
                .iter()
                .any(|m| matches!(module.types[m.ty].inner, TypeInner::Array { base, stride, .. } if stride != layouter[base].to_stride()))
                .then(|| (h, t.clone())),
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    let mut remap: HashMap<Handle<Type>, Handle<Type>> = HashMap::new();
    for (old, t) in candidates {
        let TypeInner::Struct { mut members, .. } = t.inner else { continue };
        let mut offset = 0u32;
        let mut align = naga::proc::Alignment::ONE;
        for m in members.iter_mut() {
            let (ty, size, a) = match module.types[m.ty].inner {
                TypeInner::Array { base, size, stride } if stride != layouter[base].to_stride() => {
                    let stride = layouter[base].to_stride();
                    let n = match size {
                        naga::ArraySize::Constant(n) => n.get(),
                        _ => 1,
                    };
                    let name = module.types[m.ty].name.clone();
                    let ty = module.types.insert(
                        Type { name, inner: TypeInner::Array { base, size, stride } },
                        naga::Span::UNDEFINED,
                    );
                    (ty, stride * n, layouter[base].alignment)
                }
                _ => (m.ty, layouter[m.ty].size, layouter[m.ty].alignment),
            };
            offset = a.round_up(offset);
            m.ty = ty;
            m.offset = offset;
            offset += size;
            align = align.max(a);
        }
        let new = module.types.insert(
            Type { name: t.name.clone(), inner: TypeInner::Struct { members, span: align.round_up(offset) } },
            naga::Span::UNDEFINED,
        );
        remap.insert(old, new);
    }
    let fix = |h: &mut Handle<Type>| {
        if let Some(n) = remap.get(h) {
            *h = *n;
        }
    };
    let fix_expressions = |arena: &mut naga::Arena<Expression>| {
        for (_, e) in arena.iter_mut() {
            match e {
                Expression::Compose { ty, .. } | Expression::ZeroValue(ty) => fix(ty),
                _ => {}
            }
        }
    };
    for (_, g) in module.global_variables.iter_mut() {
        fix(&mut g.ty);
    }
    for (_, c) in module.constants.iter_mut() {
        fix(&mut c.ty);
    }
    for (_, o) in module.overrides.iter_mut() {
        fix(&mut o.ty);
    }
    fix_expressions(&mut module.global_expressions);
    let fix_function = |f: &mut naga::Function| {
        for a in f.arguments.iter_mut() {
            fix(&mut a.ty);
        }
        if let Some(r) = f.result.as_mut() {
            fix(&mut r.ty);
        }
        for (_, l) in f.local_variables.iter_mut() {
            fix(&mut l.ty);
        }
        fix_expressions(&mut f.expressions);
    };
    for (_, f) in module.functions.iter_mut() {
        fix_function(f);
    }
    for ep in module.entry_points.iter_mut() {
        fix_function(&mut ep.function);
    }
}

/// Iris Features: functions in the order they call each other. A pack may call a function
/// declared further down (after a prototype); naga's GLSL frontend keeps the file's order and
/// its validator wants every callee first ("depends on ... which has not been processed yet").
fn callees_first(module: &mut naga::Module) {
    use naga::{Expression, Function, Handle, Statement};
    fn calls(block: &naga::Block, out: &mut Vec<Handle<Function>>) {
        for s in block.iter() {
            match s {
                Statement::Call { function, .. } => out.push(*function),
                Statement::Block(b) => calls(b, out),
                Statement::If { accept, reject, .. } => {
                    calls(accept, out);
                    calls(reject, out);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| calls(&c.body, out)),
                Statement::Loop { body, continuing, .. } => {
                    calls(body, out);
                    calls(continuing, out);
                }
                _ => {}
            }
        }
    }
    let handles: Vec<Handle<Function>> = module.functions.iter().map(|(h, _)| h).collect();
    let deps: Vec<Vec<Handle<Function>>> = module
        .functions
        .iter()
        .map(|(_, f)| {
            let mut v = Vec::new();
            calls(&f.body, &mut v);
            v.extend(f.expressions.iter().filter_map(|(_, e)| match e {
                Expression::CallResult(c) => Some(*c),
                _ => None,
            }));
            v
        })
        .collect();
    if deps.iter().enumerate().all(|(i, d)| d.iter().all(|c| c.index() < i)) {
        return;
    }
    // Depth-first: each function after its callees; a cycle (recursion, invalid anyway) stops.
    let mut order: Vec<usize> = Vec::new();
    let mut state = vec![0u8; handles.len()];
    fn visit(i: usize, deps: &[Vec<Handle<Function>>], state: &mut [u8], order: &mut Vec<usize>) {
        if state[i] != 0 {
            return;
        }
        state[i] = 1;
        for c in &deps[i] {
            visit(c.index(), deps, state, order);
        }
        state[i] = 2;
        order.push(i);
    }
    for i in 0..handles.len() {
        visit(i, &deps, &mut state, &mut order);
    }
    let mut new_index = vec![0usize; handles.len()];
    for (n, &i) in order.iter().enumerate() {
        new_index[i] = n;
    }
    let mut old: Vec<Option<(Function, naga::Span)>> = Vec::new();
    let spans: Vec<naga::Span> = handles.iter().map(|&h| module.functions.get_span(h)).collect();
    for (k, f) in std::mem::take(&mut module.functions).into_inner().into_iter().enumerate() {
        old.push(Some((f, spans[k])));
    }
    let map = |h: Handle<Function>| handles[new_index[h.index()]];
    fn remap(block: &mut naga::Block, map: &dyn Fn(Handle<Function>) -> Handle<Function>) {
        for s in block.iter_mut() {
            match s {
                Statement::Call { function, .. } => *function = map(*function),
                Statement::Block(b) => remap(b, map),
                Statement::If { accept, reject, .. } => {
                    remap(accept, map);
                    remap(reject, map);
                }
                Statement::Switch { cases, .. } => cases.iter_mut().for_each(|c| remap(&mut c.body, map)),
                Statement::Loop { body, continuing, .. } => {
                    remap(body, map);
                    remap(continuing, map);
                }
                _ => {}
            }
        }
    }
    let fix = |f: &mut Function| {
        remap(&mut f.body, &map);
        for (_, e) in f.expressions.iter_mut() {
            if let Expression::CallResult(c) = e {
                *c = map(*c);
            }
        }
    };
    for &i in &order {
        let (mut f, span) = old[i].take().unwrap();
        fix(&mut f);
        module.functions.append(f, span);
    }
    for ep in module.entry_points.iter_mut() {
        fix(&mut ep.function);
    }
}

/// Iris Features: the pack's `imageAtomicMax(image, p, v)` and its kin, written
/// `imageStore(image, p, hz_atomic_max(v))` for naga's GLSL frontend (it has no image atomics):
/// the store becomes the atomic. Without the device's image atomics it stays a store.
fn image_atomics(module: &mut naga::Module) {
    use naga::{AtomicFunction, Expression, Statement};
    if !crate::iris::image_atomics() {
        return;
    }
    let funs: Vec<(naga::Handle<naga::Function>, AtomicFunction)> = module
        .functions
        .iter()
        .filter_map(|(h, f)| {
            let fun = match f.name.as_deref()?.strip_prefix("hz_atomic_")? {
                "add" => AtomicFunction::Add,
                "min" => AtomicFunction::Min,
                "max" => AtomicFunction::Max,
                "and" => AtomicFunction::And,
                "or" => AtomicFunction::InclusiveOr,
                "xor" => AtomicFunction::ExclusiveOr,
                "exchange" => AtomicFunction::Exchange { compare: None },
                _ => return None,
            };
            Some((h, fun))
        })
        .collect();
    if funs.is_empty() {
        return;
    }
    type Marked = Vec<(naga::Handle<Expression>, AtomicFunction, naga::Handle<Expression>)>;
    fn marked(block: &naga::Block, funs: &[(naga::Handle<naga::Function>, AtomicFunction)], out: &mut Marked) {
        for s in block.iter() {
            match s {
                Statement::Call { function, arguments, result: Some(r) } => {
                    if let Some((_, f)) = funs.iter().find(|(h, _)| h == function)
                        && let [v] = arguments[..]
                    {
                        out.push((*r, *f, v));
                    }
                }
                Statement::Block(b) => marked(b, funs, out),
                Statement::If { accept, reject, .. } => {
                    marked(accept, funs, out);
                    marked(reject, funs, out);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| marked(&c.body, funs, out)),
                Statement::Loop { body, continuing, .. } => {
                    marked(body, funs, out);
                    marked(continuing, funs, out);
                }
                _ => {}
            }
        }
    }
    fn swap(
        block: &mut naga::Block,
        expressions: &naga::Arena<Expression>,
        marks: &Marked,
        changed: &mut Vec<naga::Handle<naga::GlobalVariable>>,
    ) {
        for s in block.iter_mut() {
            match s {
                Statement::ImageStore { image, coordinate, array_index, value } => {
                    let Some(&(_, fun, v)) = marks.iter().find(|m| m.0 == *value) else { continue };
                    let Expression::GlobalVariable(g) = expressions[*image] else { continue };
                    if !changed.contains(&g) {
                        changed.push(g);
                    }
                    *s = Statement::ImageAtomic {
                        image: *image,
                        coordinate: *coordinate,
                        array_index: *array_index,
                        fun,
                        value: v,
                    };
                }
                Statement::Block(b) => swap(b, expressions, marks, changed),
                Statement::If { accept, reject, .. } => {
                    swap(accept, expressions, marks, changed);
                    swap(reject, expressions, marks, changed);
                }
                Statement::Switch { cases, .. } => {
                    cases.iter_mut().for_each(|c| swap(&mut c.body, expressions, marks, changed))
                }
                Statement::Loop { body, continuing, .. } => {
                    swap(body, expressions, marks, changed);
                    swap(continuing, expressions, marks, changed);
                }
                _ => {}
            }
        }
    }
    let mut changed = Vec::new();
    let mut each = |f: &mut naga::Function| {
        let mut marks = Vec::new();
        marked(&f.body, &funs, &mut marks);
        if !marks.is_empty() {
            swap(&mut f.body, &f.expressions, &marks, &mut changed);
        }
    };
    for (_, f) in module.functions.iter_mut() {
        each(f);
    }
    for ep in module.entry_points.iter_mut() {
        each(&mut ep.function);
    }
    for h in changed {
        let ty = module.global_variables[h].ty;
        let naga::TypeInner::Image { dim, arrayed, class: naga::ImageClass::Storage { format, .. } } =
            module.types[ty].inner
        else {
            continue;
        };
        let name = module.types[ty].name.clone();
        let access = naga::StorageAccess::LOAD | naga::StorageAccess::STORE | naga::StorageAccess::ATOMIC;
        module.global_variables[h].ty = module.types.insert(
            naga::Type {
                name,
                inner: naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class: naga::ImageClass::Storage { format, access },
                },
            },
            naga::Span::UNDEFINED,
        );
    }
}

thread_local! {
    // How many times `compile` has gone round with the text rewritten (`text_fixes`).
    static REWRITES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Iris Features: what to write differently in the GLSL text itself for naga to take it, found
/// from the parsed module's spans: the scalar of `ivec2 % int` becomes `ivec2(int)` (naga's
/// frontend widens it for every other operator). As (start, end, replacement) byte ranges.
fn text_fixes(module: &naga::Module, text: &str) -> Vec<(usize, usize, String)> {
    use naga::{BinaryOperator, Expression, ScalarKind, TypeInner};
    let mut fixes = Vec::new();
    let mut scan = |f: &naga::Function| {
        let ctx = naga::proc::ResolveContext::with_locals(module, &f.local_variables, &f.arguments);
        let mut typifier = naga::front::Typifier::new();
        let Some((last, _)) = f.expressions.iter().last() else { return };
        if typifier.grow(last, &f.expressions, &ctx).is_err() {
            return;
        }
        for (_, e) in f.expressions.iter() {
            let Expression::Binary { op: BinaryOperator::Modulo, left, right } = *e else { continue };
            let (scalar, size, kind) = match (typifier.get(left, &module.types), typifier.get(right, &module.types)) {
                (TypeInner::Vector { size, scalar }, TypeInner::Scalar(_)) => (right, *size, scalar.kind),
                (TypeInner::Scalar(_), TypeInner::Vector { size, scalar }) => (left, *size, scalar.kind),
                _ => continue,
            };
            let Some(range) = f.expressions.get_span(scalar).to_range() else { continue };
            if range.end > text.len() || range.start >= range.end || !text.is_char_boundary(range.start) || !text.is_char_boundary(range.end) {
                continue;
            }
            let prefix = match kind {
                ScalarKind::Sint => "i",
                ScalarKind::Uint => "u",
                _ => "",
            };
            let n = size as u8;
            fixes.push((range.start, range.end, format!("{prefix}vec{n}({})", &text[range.clone()])));
        }
    };
    for (_, f) in module.functions.iter() {
        scan(f);
    }
    for ep in module.entry_points.iter() {
        scan(&ep.function);
    }
    // Innermost last, none overlapping, applied from the end.
    fixes.sort_by(|a, b| b.0.cmp(&a.0));
    let mut kept: Vec<(usize, usize, String)> = Vec::new();
    for f in fixes {
        if kept.last().is_none_or(|k| f.1 <= k.0) {
            kept.push(f);
        }
    }
    kept
}

/// `text` with `fixes` (from the end, as `text_fixes` orders them) applied.
fn apply_text_fixes(text: &str, fixes: &[(usize, usize, String)]) -> String {
    let mut out = text.to_owned();
    for (start, end, with) in fixes {
        out.replace_range(*start..*end, with);
    }
    out
}
