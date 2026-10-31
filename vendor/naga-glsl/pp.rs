//! A GLSL tokenizer standing in for `pp-rs`: the text it reads is already preprocessed (Horizon
//! expands macros and conditionals), so it only splits tokens and passes `#version`,
//! `#extension` and `#pragma` lines on. The token types keep `pp-rs`'s shapes, which the
//! frontend's parser reads.

use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Location {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Punct {
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    LeftShiftAssign,
    RightShiftAssign,
    AndAssign,
    XorAssign,
    OrAssign,
    Increment,
    Decrement,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LessEqual,
    GreaterEqual,
    EqualEqual,
    NotEqual,
    LeftShift,
    RightShift,
    LeftBrace,
    RightBrace,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftAngle,
    RightAngle,
    Semicolon,
    Comma,
    Colon,
    Dot,
    Equal,
    Bang,
    Minus,
    Tilde,
    Plus,
    Star,
    Slash,
    Percent,
    Pipe,
    Caret,
    Ampersand,
    Question,
}

#[derive(Clone, PartialEq, Debug)]
pub enum PreprocessorError {
    IntegerOverflow,
    FloatParsingError,
    UnexpectedCharacter,
    UnexpectedToken(TokenValue),
    UnexpectedNewLine,
    NotSupported16BitLiteral,
}

impl fmt::Display for PreprocessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntegerOverflow => write!(f, "integer overflow"),
            Self::FloatParsingError => write!(f, "invalid float literal"),
            Self::UnexpectedCharacter => write!(f, "unexpected character"),
            Self::UnexpectedToken(t) => write!(f, "unexpected token {t:?}"),
            Self::UnexpectedNewLine => write!(f, "unexpected end of line"),
            Self::NotSupported16BitLiteral => write!(f, "16-bit literals are not supported"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Integer {
    pub value: u64,
    pub signed: bool,
    pub width: i32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Float {
    pub value: f32,
    pub width: i32,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Version {
    pub tokens: Vec<Token>,
    pub is_first_directive: bool,
    pub has_comments_before: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Extension {
    pub tokens: Vec<Token>,
    pub has_non_directive_before: bool,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Pragma {
    pub tokens: Vec<Token>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum TokenValue {
    Ident(String),
    Integer(Integer),
    Float(Float),
    Punct(Punct),
    Version(Version),
    Extension(Extension),
    Pragma(Pragma),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Token {
    pub value: TokenValue,
    pub location: Location,
}

/// Tokens of preprocessed GLSL.
pub struct Preprocessor<'a> {
    src: &'a str,
    at: usize,
    /// Whether a directive has been read yet (`#version` must come first).
    directives: bool,
}

impl<'a> Preprocessor<'a> {
    pub fn new(src: &'a str) -> Self {
        Preprocessor { src, at: 0, directives: false }
    }

    /// Macros are expanded before the text gets here.
    pub fn add_define(&mut self, _name: &str, _value: &str) -> Result<(), PreprocessorError> {
        Ok(())
    }

    fn bytes(&self) -> &'a [u8] {
        self.src.as_bytes()
    }

    /// Skips spaces and comments; with `newlines` false, stops at a line end (inside a
    /// directive). A backslash before a line end continues the line.
    fn skip_space(&mut self, newlines: bool) {
        let b = self.bytes();
        while self.at < b.len() {
            match b[self.at] {
                b'\n' if !newlines => return,
                b' ' | b'\t' | b'\r' | b'\n' | 0x0b | 0x0c => self.at += 1,
                b'\\' if b.get(self.at + 1) == Some(&b'\n') => self.at += 2,
                b'\\' if b.get(self.at + 1) == Some(&b'\r') && b.get(self.at + 2) == Some(&b'\n') => self.at += 3,
                b'/' if b.get(self.at + 1) == Some(&b'/') => {
                    while self.at < b.len() && b[self.at] != b'\n' {
                        self.at += 1;
                    }
                }
                b'/' if b.get(self.at + 1) == Some(&b'*') => {
                    self.at += 2;
                    while self.at < b.len() && !(b[self.at] == b'*' && b.get(self.at + 1) == Some(&b'/')) {
                        self.at += 1;
                    }
                    self.at = (self.at + 2).min(b.len());
                }
                _ => return,
            }
        }
    }

    /// One token from here (spaces already skipped).
    fn token(&mut self) -> Result<Token, (PreprocessorError, Location)> {
        let b = self.bytes();
        let start = self.at;
        let loc = |end: usize| Location { start: start as u32, end: end as u32 };
        let c = b[start];
        if c.is_ascii_alphabetic() || c == b'_' {
            let mut end = start + 1;
            while end < b.len() && (b[end].is_ascii_alphanumeric() || b[end] == b'_') {
                end += 1;
            }
            self.at = end;
            return Ok(Token { value: TokenValue::Ident(self.src[start..end].to_owned()), location: loc(end) });
        }
        if c.is_ascii_digit() || (c == b'.' && b.get(start + 1).is_some_and(u8::is_ascii_digit)) {
            return self.number();
        }
        use Punct as P;
        let three = &b[start..(start + 3).min(b.len())];
        let two = &b[start..(start + 2).min(b.len())];
        let (punct, len) = match (three, two, c) {
            (b"<<=", _, _) => (P::LeftShiftAssign, 3),
            (b">>=", _, _) => (P::RightShiftAssign, 3),
            (_, b"+=", _) => (P::AddAssign, 2),
            (_, b"-=", _) => (P::SubAssign, 2),
            (_, b"*=", _) => (P::MulAssign, 2),
            (_, b"/=", _) => (P::DivAssign, 2),
            (_, b"%=", _) => (P::ModAssign, 2),
            (_, b"&=", _) => (P::AndAssign, 2),
            (_, b"^=", _) => (P::XorAssign, 2),
            (_, b"|=", _) => (P::OrAssign, 2),
            (_, b"++", _) => (P::Increment, 2),
            (_, b"--", _) => (P::Decrement, 2),
            (_, b"&&", _) => (P::LogicalAnd, 2),
            (_, b"||", _) => (P::LogicalOr, 2),
            (_, b"^^", _) => (P::LogicalXor, 2),
            (_, b"<=", _) => (P::LessEqual, 2),
            (_, b">=", _) => (P::GreaterEqual, 2),
            (_, b"==", _) => (P::EqualEqual, 2),
            (_, b"!=", _) => (P::NotEqual, 2),
            (_, b"<<", _) => (P::LeftShift, 2),
            (_, b">>", _) => (P::RightShift, 2),
            (_, _, b'{') => (P::LeftBrace, 1),
            (_, _, b'}') => (P::RightBrace, 1),
            (_, _, b'(') => (P::LeftParen, 1),
            (_, _, b')') => (P::RightParen, 1),
            (_, _, b'[') => (P::LeftBracket, 1),
            (_, _, b']') => (P::RightBracket, 1),
            (_, _, b'<') => (P::LeftAngle, 1),
            (_, _, b'>') => (P::RightAngle, 1),
            (_, _, b';') => (P::Semicolon, 1),
            (_, _, b',') => (P::Comma, 1),
            (_, _, b':') => (P::Colon, 1),
            (_, _, b'.') => (P::Dot, 1),
            (_, _, b'=') => (P::Equal, 1),
            (_, _, b'!') => (P::Bang, 1),
            (_, _, b'-') => (P::Minus, 1),
            (_, _, b'~') => (P::Tilde, 1),
            (_, _, b'+') => (P::Plus, 1),
            (_, _, b'*') => (P::Star, 1),
            (_, _, b'/') => (P::Slash, 1),
            (_, _, b'%') => (P::Percent, 1),
            (_, _, b'|') => (P::Pipe, 1),
            (_, _, b'^') => (P::Caret, 1),
            (_, _, b'&') => (P::Ampersand, 1),
            (_, _, b'?') => (P::Question, 1),
            _ => {
                // Skip the whole character.
                let len = self.src[start..].chars().next().map_or(1, char::len_utf8);
                self.at = start + len;
                return Err((PreprocessorError::UnexpectedCharacter, loc(start + len)));
            }
        };
        self.at = start + len;
        Ok(Token { value: TokenValue::Punct(punct), location: loc(start + len) })
    }

    fn number(&mut self) -> Result<Token, (PreprocessorError, Location)> {
        let b = self.bytes();
        let start = self.at;
        let mut end = start;
        // The literal's extent: digits, letters (hex digits, suffixes, exponent), dots and an
        // exponent's sign.
        while end < b.len() {
            let c = b[end];
            let exp_sign = (c == b'+' || c == b'-')
                && end > start
                && matches!(b[end - 1], b'e' | b'E')
                && !(b[start] == b'0' && matches!(b.get(start + 1), Some(b'x' | b'X')));
            if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || exp_sign {
                end += 1;
            } else {
                break;
            }
        }
        self.at = end;
        let text = &self.src[start..end];
        let location = Location { start: start as u32, end: end as u32 };
        let lower = text.to_ascii_lowercase();
        let hex = lower.starts_with("0x");
        let is_float = !hex && (lower.contains('.') || lower.contains('e') || lower.ends_with('f'));
        if is_float {
            let (digits, width) = if let Some(d) = lower.strip_suffix("lf") {
                (d, 64)
            } else if let Some(d) = lower.strip_suffix('f') {
                (d, 32)
            } else {
                (lower.as_str(), 32)
            };
            if digits.ends_with('h') {
                return Err((PreprocessorError::NotSupported16BitLiteral, location));
            }
            // A literal past f32's range (`1e39`) is the largest float, as drivers read it.
            let value: f32 = digits
                .parse::<f64>()
                .map_err(|_| (PreprocessorError::FloatParsingError, location))?
                .clamp(-f64::from(f32::MAX), f64::from(f32::MAX)) as f32;
            return Ok(Token { value: TokenValue::Float(Float { value, width }), location });
        }
        let (digits, signed) = match lower.strip_suffix('u') {
            Some(d) => (d, false),
            None => (lower.as_str(), true),
        };
        let value = if let Some(h) = digits.strip_prefix("0x") {
            u64::from_str_radix(h, 16)
        } else if digits.len() > 1 && digits.starts_with('0') {
            u64::from_str_radix(&digits[1..], 8)
        } else {
            digits.parse::<u64>()
        }
        .map_err(|_| (PreprocessorError::IntegerOverflow, location))?;
        if value > u64::from(u32::MAX) {
            return Err((PreprocessorError::IntegerOverflow, location));
        }
        Ok(Token { value: TokenValue::Integer(Integer { value, signed, width: 32 }), location })
    }

    /// The tokens of the rest of a directive's line.
    fn line_tokens(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            self.skip_space(false);
            if self.at >= self.src.len() || self.bytes()[self.at] == b'\n' {
                return tokens;
            }
            match self.token() {
                Ok(t) => tokens.push(t),
                Err(_) => {}
            }
        }
    }

    /// Whether only spaces lie between the line's start and `at`.
    fn at_line_start(&self, at: usize) -> bool {
        self.src[..at].rsplit('\n').next().is_none_or(|l| l.trim().is_empty())
    }
}

impl Iterator for Preprocessor<'_> {
    type Item = Result<Token, (PreprocessorError, Location)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.skip_space(true);
            if self.at >= self.src.len() {
                return None;
            }
            let start = self.at;
            if self.bytes()[start] == b'#' && self.at_line_start(start) {
                self.at += 1;
                self.skip_space(false);
                let word_start = self.at;
                while self.at < self.src.len() && (self.bytes()[self.at].is_ascii_alphanumeric() || self.bytes()[self.at] == b'_') {
                    self.at += 1;
                }
                let word = self.src[word_start..self.at].to_owned();
                let first = !self.directives;
                self.directives = true;
                let tokens = self.line_tokens();
                let location = Location { start: start as u32, end: self.at as u32 };
                let value = match word.as_str() {
                    "version" => TokenValue::Version(Version { tokens, is_first_directive: first, has_comments_before: false }),
                    "extension" => TokenValue::Extension(Extension { tokens, has_non_directive_before: false }),
                    "pragma" => TokenValue::Pragma(Pragma { tokens }),
                    // Anything else left in the text (`#line`): passed over.
                    _ => continue,
                };
                return Some(Ok(Token { value, location }));
            }
            self.directives = true;
            return Some(self.token());
        }
    }
}
