"""Small Rust source slicer: find items by name and brace-match their bodies, skipping
strings, chars and comments."""
import re

def _scan(src, i):
    """Index just past the matching close brace of the `{` at src[i]."""
    depth = 0
    n = len(src)
    while i < n:
        c = src[i]
        if src.startswith('//', i):
            j = src.find('\n', i)
            i = n if j < 0 else j
            continue
        if src.startswith('/*', i):
            d = 1; i += 2
            while i < n and d:
                if src.startswith('/*', i): d += 1; i += 2
                elif src.startswith('*/', i): d -= 1; i += 2
                else: i += 1
            continue
        m = re.match(r'b?r(#*)"', src[i:i+10])
        if m and (i == 0 or not (src[i-1].isalnum() or src[i-1] == '_')):
            hashes = m.group(1)
            end = '"' + hashes
            j = src.find(end, i + len(m.group(0)))
            i = j + len(end)
            continue
        if c == '"' or (c == 'b' and src.startswith('b"', i) and not (src[i-1].isalnum() or src[i-1]=='_')):
            i += 2 if c == 'b' else 1
            while src[i] != '"':
                i += 2 if src[i] == '\\' else 1
            i += 1
            continue
        if c == "'":
            # char literal vs lifetime
            m = re.match(r"'(\\.|\\u\{[0-9a-fA-F]+\}|[^'\\])'", src[i:i+12])
            if m:
                i += len(m.group(0)); continue
            i += 1; continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError('unbalanced')

def item_span(src, header_re, nth=0):
    """(start, end) of the item whose first line matches `header_re` (multiline regex at a
    line start), including its doc comments and attributes, through its closing brace (or `;`)."""
    ms = list(re.finditer(header_re, src, re.M))
    if len(ms) <= nth:
        raise KeyError(header_re)
    m = ms[nth]
    start = m.start()
    # Back over doc comments / attributes / plain comments directly above.
    lines_before = src[:start].split('\n')
    k = len(lines_before) - 1  # last element is the partial current line (empty)
    s = start
    while k > 0:
        prev = lines_before[k-1].strip()
        if prev.startswith('///') or prev.startswith('#[') or prev.startswith('//'):
            s -= len(lines_before[k-1]) + 1
            k -= 1
        else:
            break
    # Find body start: first `{` or `;` at depth 0 of parens after header.
    i = m.start(); par = 0
    while True:
        c = src[i]
        if c in '([': par += 1
        elif c in ')]': par -= 1
        elif c == '{' and par == 0:
            end = _scan(src, i); break
        elif c == ';' and par == 0:
            end = i + 1; break
        i += 1
    if end < len(src) and src[end] == '\n':
        end += 1
    return s, end

def take(src, header_re, nth=0):
    s, e = item_span(src, header_re, nth)
    return src[s:e]

def cut(src, header_re, nth=0):
    s, e = item_span(src, header_re, nth)
    return src[:s] + src[e:], src[s:e]

def strip_tests(src):
    while True:
        m = re.search(r'^#\[cfg\(test\)\]\s*\n\s*mod \w+ \{', src, re.M)
        if not m:
            return src
        i = src.index('{', m.start())
        e = _scan(src, i)
        src = src[:m.start()] + src[e:].lstrip('\n')

def outer_docs(src):
    src = re.sub(r'^(\s*)//!', r'\1//', src, flags=re.M)
    src = re.sub(r'^\s*#!\[.*\]\s*\n', '', src, flags=re.M)
    return src

def indent(src, n=4):
    pad = ' ' * n
    return '\n'.join((pad + l) if l.strip() else l for l in src.split('\n'))

def sub1(src, old, new, count=1):
    c = src.count(old)
    if c != count:
        raise ValueError(f'expected {count} of {old!r}, found {c}')
    return src.replace(old, new)
