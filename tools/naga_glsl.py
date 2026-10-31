"""Folds vendor/naga-glsl (a directory of modules) into one module's text."""
import os, re

def fold(root, path_prefix):
    def read(rel):
        s = open(os.path.join(root, rel)).read()
        s = s.replace('crate::glsl::', path_prefix)
        # Inner doc comments become plain comments (they now sit inside `mod x { }`).
        s = re.sub(r'^//!', '//', s, flags=re.M)
        return s
    def inline(text, base_dir):
        out = []
        for line in text.split('\n'):
            m = re.match(r'^(\s*)((?:pub(?:\([^)]*\))? )?mod (\w+));\s*$', line)
            if m:
                indent, decl, name = m.group(1), m.group(2), m.group(3)
                f = os.path.join(base_dir, name + '.rs')
                if not os.path.exists(os.path.join(root, f)):
                    f = os.path.join(base_dir, name, 'mod.rs')
                sub = inline(read(f), os.path.join(base_dir, name))
                out.append(f'{indent}{decl} {{\n{sub}\n{indent}}}')
            else:
                out.append(line)
        return '\n'.join(out)
    return inline(read('mod.rs'), '')
