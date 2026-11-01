#!/usr/bin/env python3
"""Packs out/iris-features as the launcher's native mod formats: an unpacked .hnmod folder and the
.hnmod archive itself (horizon-native.json + mixins/<id>/), plus the deliverable zip."""
import json, os, re, shutil, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
# The version and pins are build.py's.
_build = open(os.path.join(HERE, 'build.py')).read()
def _pin(name):
    return re.search(rf"^{name} = '([^']+)'", _build, re.M).group(1)
MOD_ID, VERSION = 'iris-features', _pin('VERSION')
HORIZON, MIXINS = _pin('HORIZON'), _pin('MIXINS')

src = os.path.join(ROOT, 'out', MOD_ID)
dist = os.path.join(ROOT, 'dist')
shutil.rmtree(dist, ignore_errors=True)
unpacked = os.path.join(dist, MOD_ID)
os.makedirs(os.path.join(unpacked, 'mixins'))
shutil.copytree(src, os.path.join(unpacked, 'mixins', MOD_ID))
manifest = {
    'schema': 1,
    'id': MOD_ID,
    'version': VERSION,
    'game': '26.3',
    'horizon': HORIZON,
    'mixins': MIXINS,
}
open(os.path.join(unpacked, 'horizon-native.json'), 'w').write(json.dumps(manifest, indent=2) + '\n')

def add_tree(z, base, prefix=''):
    for dirpath, _, names in sorted(os.walk(base)):
        for n in sorted(names):
            full = os.path.join(dirpath, n)
            rel = os.path.relpath(full, base).replace(os.sep, '/')
            z.write(full, prefix + rel)

hnmod = os.path.join(dist, f'{MOD_ID}.hnmod')
with zipfile.ZipFile(hnmod, 'w', zipfile.ZIP_DEFLATED) as z:
    add_tree(z, unpacked)

deliver = os.path.join(dist, f'{MOD_ID}-{VERSION}.zip')
with zipfile.ZipFile(deliver, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(hnmod, f'{MOD_ID}-{VERSION}/{MOD_ID}.hnmod')
    add_tree(z, unpacked, f'{MOD_ID}-{VERSION}/{MOD_ID}/')
    z.write(os.path.join(ROOT, 'README.md'), f'{MOD_ID}-{VERSION}/README.md')
    z.write(os.path.join(ROOT, 'reference.patch'), f'{MOD_ID}-{VERSION}/reference.patch')
    for t in ['build.py', 'build_render.py', 'build_tests.py', 'rsx.py', 'stage.sh', 'package.py', 'naga_glsl.py']:
        z.write(os.path.join(HERE, t), f'{MOD_ID}-{VERSION}/tools/{t}')
    add_tree(z, os.path.join(ROOT, 'ref'), f'{MOD_ID}-{VERSION}/ref/')
    z.write(os.path.join(ROOT, 'LICENSE'), f'{MOD_ID}-{VERSION}/LICENSE')
    # naga's GLSL frontend, with its licences.
    add_tree(z, os.path.join(ROOT, 'vendor'), f'{MOD_ID}-{VERSION}/vendor/')
print(hnmod)
print(deliver)
