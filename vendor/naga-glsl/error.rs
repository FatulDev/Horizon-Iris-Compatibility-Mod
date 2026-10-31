use std::{
    borrow::Cow,
    string::{String, ToString},
    vec,
    vec::Vec,
};

use crate::glsl::pp::PreprocessorError;

use super::token::TokenValue;
use naga::SourceLocation;
use naga::{proc::ConstantEvaluatorError, Span};

fn join_with_comma(list: &[ExpectedToken]) -> String {
    let mut string = "".to_string();
    for (i, val) in list.iter().enumerate() {
        string.push_str(&val.to_string());
        match i {
            i if i == list.len() - 1 => {}
            i if i == list.len() - 2 => string.push_str(" or "),
            _ => string.push_str(", "),
        }
    }
    string
}

/// One of the expected tokens returned in [`InvalidToken`](ErrorKind::InvalidToken).
#[derive(Clone, Debug, PartialEq)]
pub enum ExpectedToken {
    /// A specific token was expected.
    Token(TokenValue),
    /// A type was expected.
    TypeName,
    /// An identifier was expected.
    Identifier,
    /// An integer literal was expected.
    IntLiteral,
    /// A float literal was expected.
    FloatLiteral,
    /// A boolean literal was expected.
    BoolLiteral,
    /// The end of file was expected.
    Eof,
}
impl From<TokenValue> for ExpectedToken {
    fn from(token: TokenValue) -> Self {
        ExpectedToken::Token(token)
    }
}
impl core::fmt::Display for ExpectedToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            ExpectedToken::Token(ref token) => write!(f, "{token:?}"),
            ExpectedToken::TypeName => write!(f, "a type"),
            ExpectedToken::Identifier => write!(f, "identifier"),
            ExpectedToken::IntLiteral => write!(f, "integer literal"),
            ExpectedToken::FloatLiteral => write!(f, "float literal"),
            ExpectedToken::BoolLiteral => write!(f, "bool literal"),
            ExpectedToken::Eof => write!(f, "end of file"),
        }
    }
}

/// Information about the cause of an error.
#[derive(Clone, Debug)]
pub enum ErrorKind {
    /// Whilst parsing as encountered an unexpected EOF.
    EndOfFile,
    /// The shader specified an unsupported or invalid profile.
    InvalidProfile(String),
    /// The shader requested an unsupported or invalid version.
    InvalidVersion(u64),
    /// Whilst parsing an unexpected token was encountered.
    ///
    /// A list of expected tokens is also returned.
    InvalidToken(TokenValue, Vec<ExpectedToken>),
    /// A specific feature is not yet implemented.
    ///
    /// To help prioritize work please open an issue in the github issue tracker
    /// if none exist already or react to the already existing one.
    NotImplemented(&'static str),
    /// A reference to a variable that wasn't declared was used.
    UnknownVariable(String),
    /// A reference to a type that wasn't declared was used.
    UnknownType(String),
    /// A reference to a non existent member of a type was made.
    UnknownField(String),
    /// An unknown layout qualifier was used.
    ///
    /// If the qualifier does exist please open an issue in the github issue tracker
    /// if none exist already or react to the already existing one to help
    /// prioritize work.
    UnknownLayoutQualifier(String),
    /// Unsupported matrix of the form matCx2
    ///
    /// Our IR expects matrices of the form matCx2 to have a stride of 8 however
    /// matrices in the std140 layout have a stride of at least 16.
    UnsupportedMatrixWithTwoRowsInStd140 { columns: u8 },
    /// Unsupported matrix of the form f16matCxR
    ///
    /// Our IR expects matrices of the form f16matCxR to have a stride of 4/8/8 depending on row-count,
    /// however matrices in the std140 layout have a stride of at least 16.
    UnsupportedF16MatrixInStd140 { columns: u8, rows: u8 },
    /// A variable with the same name already exists in the current scope.
    VariableAlreadyDeclared(String),
    /// A semantic error was detected in the shader.
    SemanticError(Cow<'static, str>),
    /// An error was returned by the preprocessor.
    PreprocessorError(PreprocessorError),
    /// The parser entered an illegal state and exited
    ///
    /// This obviously is a bug and as such should be reported in the github issue tracker
    InternalError(&'static str),
}


impl core::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EndOfFile => write!(f, "Unexpected end of file"),
            Self::InvalidProfile(p) => write!(f, "Invalid profile: {p}"),
            Self::InvalidVersion(v) => write!(f, "Invalid version: {v}"),
            Self::InvalidToken(found, expected) => write!(f, "Expected {}, found {found:?}", join_with_comma(expected)),
            Self::NotImplemented(w) => write!(f, "Not implemented: {w}"),
            Self::UnknownVariable(n) => write!(f, "Unknown variable: {n}"),
            Self::UnknownType(n) => write!(f, "Unknown type: {n}"),
            Self::UnknownField(n) => write!(f, "Unknown field: {n}"),
            Self::UnknownLayoutQualifier(n) => write!(f, "Unknown layout qualifier: {n}"),
            Self::UnsupportedMatrixWithTwoRowsInStd140 { columns } => write!(f, "unsupported matrix of the form matCx2 (in this case mat{columns}x2) in std140 block layout"),
            Self::UnsupportedF16MatrixInStd140 { columns, rows } => write!(f, "unsupported matrix of the form f16matCxR (in this case f16mat{columns}x{rows}) in std140 block layout"),
            Self::VariableAlreadyDeclared(n) => write!(f, "Variable already declared: {n}"),
            Self::SemanticError(e) => write!(f, "{e}"),
            Self::PreprocessorError(e) => write!(f, "{e:?}"),
            Self::InternalError(e) => write!(f, "Internal error: {e}"),
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for Error {}

impl From<ConstantEvaluatorError> for ErrorKind {
    fn from(err: ConstantEvaluatorError) -> Self {
        ErrorKind::SemanticError(err.to_string().into())
    }
}

/// Error returned during shader parsing.
#[derive(Clone, Debug)]
pub struct Error {
    /// Holds the information about the error itself.
    pub kind: ErrorKind,
    /// Holds information about the range of the source code where the error happened.
    pub meta: Span,
}

impl Error {
    /// Returns a [`SourceLocation`] for the error message.
    pub fn location(&self, source: &str) -> Option<SourceLocation> {
        Some(self.meta.location(source))
    }
}

/// A collection of errors returned during shader parsing.
#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct ParseErrors {
    pub errors: Vec<Error>,
}

impl ParseErrors {
    pub fn emit_to_string(&self, source: &str) -> String {
        let mut out = String::new();
        for e in &self.errors {
            let line = e.meta.to_range().map_or(0, |r| source[..r.start.min(source.len())].matches('\n').count() + 1);
            out.push_str(&format!("line {line}: {}\n", e.kind));
        }
        out
    }
}

impl core::fmt::Display for ParseErrors {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.errors.iter().try_for_each(|e| write!(f, "{e:?}"))
    }
}

impl core::error::Error for ParseErrors {}

impl From<Vec<Error>> for ParseErrors {
    fn from(errors: Vec<Error>) -> Self {
        Self { errors }
    }
}
