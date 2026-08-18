//! Variable expansion for profile path values.
//!
//! Every unrecognised form is an error rather than a literal, so adding a
//! variable later cannot silently reinterpret a grant written today.

use crate::error::{SandboxError, SandlockError};

/// The closed vocabulary. Values are resolved by sandlock, never looked up
/// in the environment by name.
const VARS: &[&str] = &["HOME"];

fn invalid(msg: String) -> SandlockError {
    SandlockError::Sandbox(SandboxError::Invalid(msg))
}

/// Resolve `${HOME}`.
///
/// The environment wins over passwd because the sandboxed program resolves
/// its own `~` through `$HOME`: a passwd-derived grant would cover a
/// directory the program never opens while denying the one it does.
pub fn resolve_home() -> Result<String, SandlockError> {
    let env = std::env::var("HOME").ok();
    let uid = nix::unistd::Uid::current();
    let passwd = nix::unistd::User::from_uid(uid)
        .ok()
        .flatten()
        .map(|u| u.dir.to_string_lossy().into_owned());
    resolve_home_from(env.as_deref(), passwd.as_deref())
}

/// Split out from `resolve_home` so the precedence rule is testable without
/// mutating the process environment.
fn resolve_home_from(env: Option<&str>, passwd: Option<&str>) -> Result<String, SandlockError> {
    for candidate in [env, passwd].into_iter().flatten() {
        if is_usable_home(candidate) {
            return Ok(candidate.to_string());
        }
    }
    Err(invalid(
        "cannot resolve ${HOME}: $HOME is unset, not absolute, or is the \
         filesystem root, and this uid has no passwd entry with a usable home \
         directory"
            .to_string(),
    ))
}

/// `/` is rejected alongside the relative and empty cases: it is nobody's home,
/// and expanding it would turn `write = ["${HOME}"]` into a grant over the
/// entire filesystem, which is the one thing a sandbox must never hand out by
/// accident.
fn is_usable_home(dir: &str) -> bool {
    dir.starts_with('/') && !dir.trim_end_matches('/').is_empty()
}

/// Expand `${HOME}` in one profile path value.
pub fn expand(value: &str, home: &str) -> Result<String, SandlockError> {
    if value.starts_with('~') {
        return Err(invalid(format!(
            "{value:?}: tilde is not expanded in profiles; write ${{HOME}} instead"
        )));
    }
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];
        if let Some(tail) = after.strip_prefix('{') {
            let end = tail
                .find('}')
                .ok_or_else(|| invalid(format!("{value:?}: unterminated ${{")))?;
            out.push_str(lookup(&tail[..end], home, value)?);
            rest = &tail[end + 1..];
        } else {
            return Err(invalid(bare_dollar_message(value, after)));
        }
    }
    out.push_str(rest);
    Ok(out)
}

fn lookup<'a>(name: &str, home: &'a str, value: &str) -> Result<&'a str, SandlockError> {
    if !is_well_formed(name) {
        return Err(invalid(format!(
            "{value:?}: malformed variable name ${{{name}}}; names match [A-Za-z_][A-Za-z0-9_]*"
        )));
    }
    if name == "HOME" {
        return Ok(home);
    }
    let suggestion = VARS
        .iter()
        .find(|v| v.eq_ignore_ascii_case(name))
        .map(|v| format!("; did you mean ${{{v}}}?"))
        .unwrap_or_default();
    Err(invalid(format!(
        "{value:?}: unknown variable ${{{name}}}{suggestion}; supported: {}",
        VARS.iter()
            .map(|v| format!("${{{v}}}"))
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

fn is_well_formed(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn bare_dollar_message(value: &str, after: &str) -> String {
    let name: String = after
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        format!("{value:?}: bare $ is not allowed; a literal $ cannot appear in a profile path")
    } else {
        format!("{value:?}: bare $ is not a variable; write ${{{name}}} for a variable")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        home: String,
        case: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        input: String,
        home: Option<String>,
        expect: Option<String>,
        error: Option<String>,
    }

    #[test]
    fn fixture_cases() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/profile_expansion.toml"
        );
        let text = std::fs::read_to_string(path).expect("read fixture");
        let fixture: Fixture = toml::from_str(&text).expect("parse fixture");

        for c in &fixture.case {
            let home = c.home.as_deref().unwrap_or(&fixture.home);
            let got = expand(&c.input, home);
            match (&c.expect, &c.error) {
                (Some(want), None) => {
                    let got = got.unwrap_or_else(|e| panic!("{:?}: unexpected error {e}", c.input));
                    assert_eq!(&got, want, "input {:?}", c.input);
                }
                (None, Some(want)) => {
                    let err = got.expect_err(&format!("{:?}: expected an error", c.input));
                    let msg = err.to_string();
                    assert!(
                        msg.contains(want),
                        "input {:?}: error {msg:?} does not contain {want:?}",
                        c.input
                    );
                }
                _ => panic!("{:?}: case needs exactly one of expect/error", c.input),
            }
        }
    }

    #[test]
    fn resolve_home_prefers_absolute_env() {
        assert_eq!(
            resolve_home_from(Some("/env/home"), Some("/passwd/home")).unwrap(),
            "/env/home"
        );
    }

    #[test]
    fn resolve_home_falls_back_to_passwd() {
        // Unset, empty, and relative all fail the absolute test, so passwd wins.
        for env in [None, Some(""), Some("relative/home")] {
            assert_eq!(
                resolve_home_from(env, Some("/passwd/home")).unwrap(),
                "/passwd/home",
                "env {env:?}"
            );
        }
    }

    #[test]
    fn resolve_home_refuses_the_filesystem_root() {
        // `/` is absolute but is nobody's home, and expanding it would turn
        // `write = ["${HOME}"]` into a grant over the whole filesystem.
        for env in [Some("/"), Some("//")] {
            assert_eq!(
                resolve_home_from(env, Some("/passwd/home")).unwrap(),
                "/passwd/home",
                "env {env:?}"
            );
            assert!(resolve_home_from(env, Some("/")).is_err(), "env {env:?}");
        }
    }

    #[test]
    fn resolve_home_keeps_a_trailing_slash_home() {
        // Only the all-slashes case is meaningless; `/root/` is a real home.
        assert_eq!(resolve_home_from(Some("/root/"), None).unwrap(), "/root/");
    }

    #[test]
    fn resolve_home_errors_when_neither_is_absolute() {
        let err = resolve_home_from(None, None).unwrap_err().to_string();
        assert!(err.contains("cannot resolve ${HOME}"), "error was {err:?}");
        assert!(resolve_home_from(Some("rel"), Some("also-rel")).is_err());
    }

    #[test]
    fn resolve_home_on_this_host_is_absolute() {
        let home = resolve_home().expect("resolve home");
        assert!(home.starts_with('/'), "home {home:?} must be absolute");
    }
}
