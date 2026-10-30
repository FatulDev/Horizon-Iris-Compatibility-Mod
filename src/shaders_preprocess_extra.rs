
/// Whether `line` calls a function-like macro of `macros` whose parentheses it leaves open.
fn open_macro_call(line: &str, macros: &Macros) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let name = &line[start..i];
            if !(macros.is_defined(name) && macros.get(name).is_none()) {
                continue;
            }
            let rest = line[i..].trim_start();
            if rest.is_empty() {
                return true;
            }
            if !rest.starts_with('(') {
                continue;
            }
            let mut depth = 0i32;
            for ch in rest.chars() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth > 0 {
                return true;
            }
        } else if c.is_ascii_digit() {
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    false
}
