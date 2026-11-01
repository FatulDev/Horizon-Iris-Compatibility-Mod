"""The harness (`runs_a_real_pack`) as a bundle of its own, from the development branch."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
dev = open(os.path.join(ROOT, 'ref/crates/horizon-client-render/src/gfx/pack_post.rs')).read()
i = dev.index('pub(crate) mod iris_tests {')
body = dev[i + len('pub(crate) mod iris_tests {'):].rstrip()
assert body.endswith('}')
body = body[:-1]
# The game defines Iris's macros in `Renderer::define_pack_macros` (the mod's hook); the test
# makes its own `Macros`, so it defines them itself.
macros_src = open(os.path.join(ROOT, 'ref/crates/horizon-shaders/src/macros.rs')).read()
defs = re.findall(r'^        m\.define\("(IRIS_VERSION|MC_GL_ARB_shader_image_load_store|MC_NORMAL_MAP|MC_SPECULAR_MAP|MC_GL_ARB_texture_gather|MC_GL_EXT_gpu_shader4)", "([^"]*)"\);', macros_src, re.M)
assert len(defs) == 6, defs
a = '            let mut macros = Macros::standard();\n'
assert body.count(a) == 1
body = body.replace(a, a + ''.join(f'            macros.define("{n}", "{v}");\n' for n, v in defs))
out = os.path.join(ROOT, 'out', 'iris-features-tests')
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'tests.rs'), 'w').write(re.sub(r'^    ', '', body, flags=re.M))
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
visibility = "pub(crate)"
''')
print('test bundle written')
