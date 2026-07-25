//! Backend-agnostic recovery of preserved COW-branch storage.
//!
//! When a transaction (or a plain [`Sandbox`](crate::sandbox::Sandbox) whose
//! branch action is [`Keep`](crate::sandbox::BranchAction::Keep)) cannot
//! reclaim its change set — a commit that could not take the workdir lock, a
//! merge that failed partway, or work deliberately kept for inspection — the
//! change set is left on disk instead of thrown away. This module is the
//! backend-neutral entry point for finding and reading that preserved work; it
//! deliberately does not name the COW backend that produced it.
//!
//! Recovery is broader than transactions: a plain `Sandbox` with
//! [`BranchAction::Keep`](crate::sandbox::BranchAction::Keep) also preserves
//! work, which is why this lives in its own module rather than under
//! `transaction`.
//!
//! # A running merge looks like an interrupted one
//!
//! [`list_preserved`] reports every preserved branch under a storage base,
//! including a merge that is *still running*: a commit writes its
//! [`PreserveReason::MergeInterrupted`] marker before the first destructive
//! step. The marker's [`pid`](PreservedBranch::pid) is what separates the two —
//! a sweep that *acts* on a branch, rather than only reporting it, must check
//! that pid is not a live process first.
//!
//! # Durability of the default storage
//!
//! With no explicit `fs_storage`, preserved work lands in a stable per-user base
//! (`$XDG_RUNTIME_DIR/sandlock-cow` when available, otherwise a securely-created
//! `$TMPDIR/sandlock-cow-<uid>`), so [`list_preserved`] on that base spans a
//! user's dead pids. `$XDG_RUNTIME_DIR` is nonetheless **session-scoped**:
//! `systemd-logind` removes `/run/user/<uid>` on last-session-exit and it is a
//! size-limited tmpfs. A daemon or any cross-session recovery MUST therefore set
//! an explicit, durable, disk-backed `fs_storage` rather than rely on the
//! default.

pub use crate::cow::seccomp::{list_preserved, read_preserved, PreserveReason, PreservedBranch};
