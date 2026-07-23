// Each test here spawns a full `sandlock` process.
// Running all tests in parallel exhausts kernel resources on most hosts.
// Limit parallelism when running this suite:
//   cargo test -p sandlock-cli --test learn_test -- --test-threads=4

use std::process::Command;

fn sandlock_bin() -> Command {
    let cmd = Command::new(env!("CARGO_BIN_EXE_sandlock"));
    cmd
}

/// Drop `-r /lib64` from a CLI argument list when the host has no `/lib64`
/// (RISC-V glibc and musl put the loader under `/lib`, with no `/lib64` at
/// all). `-r` maps to a mandatory `fs_read`, so requiring `/lib64` on such a
/// host aborts confinement; this mirrors `fs_read_if_exists` at the CLI layer.
/// On hosts that have `/lib64` (x86-64) the arguments pass through unchanged.
fn args_for_host(args: &[&str]) -> Vec<String> {
    let has_lib64 = std::path::Path::new("/lib64").exists();
    let mut out: Vec<String> = Vec::with_capacity(args.len());
    for a in args {
        if *a == "/lib64" && !has_lib64 {
            if out.last().map(|s| s == "-r").unwrap_or(false) {
                out.pop();
            }
            continue;
        }
        out.push((*a).to_string());
    }
    out
}

// ── Filesystem reads ──────────────────────────────────────────────────────────

/// Reads are recorded in the profile.
#[test]
fn test_learn_captures_fs_read() {
    let output = sandlock_bin()
        .args(["learn", "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(
        read_line.contains("/etc/hostname"),
        "expected /etc/hostname under read = [...], got: {read_line}",
    );
}

/// When the workload binary is a symlink, the real binary path is captured via process maps.
/// Skipped on hosts where python3 resolves directly (not a symlink).
#[test]
fn test_learn_captures_real_binary_path_via_maps() {
    let python3 = std::process::Command::new("which")
        .arg("python3")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if python3.is_empty() { return; }
    let real = match std::fs::canonicalize(&python3) {
        Ok(p) => p,
        Err(_) => return,
    };
    if real.to_str() == Some(python3.as_str()) {
        eprintln!("skipped: python3 is not a symlink on this host");
        return;
    }
    let real_str = real.to_str().unwrap().to_string();

    let output = sandlock_bin()
        .args(["learn", "--", "python3", "-c", "pass"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(
        read_line.contains(&real_str),
        "real binary {real_str} not in reads (only visible via r-xp maps): {read_line}",
    );
}

// ── Filesystem writes ─────────────────────────────────────────────────────────

/// Writes to existing files are recorded in the profile. COW ensures real file content is unchanged.
#[test]
fn test_learn_captures_fs_write() {
    let tmp1 = tempfile::NamedTempFile::new().expect("tempfile");
    let tmp2 = tempfile::Builder::new().tempdir_in("/var/tmp").expect("tempdir");
    let tmp2_file = tmp2.path().join("sandlock-learn-write2.txt");
    let path1 = tmp1.path().to_str().unwrap().to_owned();
    let path2 = tmp2_file.to_str().unwrap().to_owned();
    let cmd = format!("echo x > {path1} && echo y > {path2}");
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &cmd])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains(&path1), "expected {path1} under write = [...], got: {write_line}");
    assert!(write_line.contains(&path2) || write_line.contains(tmp2.path().to_str().unwrap()),
        "expected {path2} (or its parent) under write = [...], got: {write_line}");
    // COW: real file must not have been modified
    let content = std::fs::read_to_string(&path1).unwrap_or_default();
    assert!(content.is_empty(), "COW isolation failed: real file was written to");
}

/// Creates to non-existent files collapse to the parent directory.
/// COW ensures the real filesystem is not modified during learn.
#[test]
fn test_learn_new_file_collapses_to_parent() {
    let dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let path = dir.path().join("write-test.txt");
    let path_str = path.to_str().unwrap();
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &format!("echo x > {path_str}")])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(
        write_line.contains("/var/tmp"),
        "expected /var/tmp under write = [...], got: {write_line}",
    );
    assert!(!path.exists(), "COW isolation failed: real filesystem was modified");
}

/// Directory creates and deletions are captured. COW ensures the real filesystem is not modified.
#[test]
fn test_learn_captures_dir_create_and_delete() {
    let base = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let dir = base.path().join("newdir");
    let dir_str = dir.to_str().unwrap();
    let file = tempfile::NamedTempFile::new_in(base.path()).expect("tempfile");
    let file_path = file.path().to_str().unwrap().to_owned();
    let cmd = format!("mkdir {dir_str} && rm {file_path}");
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &cmd])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains("/var/tmp"),
        "expected /var/tmp in write = [...], got: {write_line}");
    assert!(!dir.exists(), "COW isolation failed: dir was created on real FS");
    assert!(std::path::Path::new(&file_path).exists(), "COW isolation failed: file was deleted on real FS");
}

/// Renames capture both source and destination parents. COW ensures the real filesystem is not modified.
#[test]
fn test_learn_captures_rename() {
    let src = tempfile::NamedTempFile::new_in("/var/tmp").expect("tempfile in /var/tmp");
    let src_path = src.path().to_str().unwrap().to_owned();
    let dst = "/tmp/sandlock-learn-rename-dst-test";
    let cmd = format!("mv {src_path} {dst}");
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &cmd])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains("/var/tmp"), "expected /var/tmp (src parent) in: {write_line}");
    assert!(write_line.contains("/tmp"), "expected /tmp (dst parent) in: {write_line}");
    assert!(src.path().exists(), "COW isolation failed: src was deleted on real FS");
    assert!(!std::path::Path::new(dst).exists(), "COW isolation failed: dst was created on real FS");
}

/// Symlink creates are captured by the link path, not the target.
/// COW ensures the real filesystem is not modified.
#[test]
fn test_learn_captures_symlink() {
    let dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let link = dir.path().join("link");
    let link_str = link.to_str().unwrap();
    let cmd = format!("ln -s hostname {link_str}");
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &cmd])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains("/var/tmp"), "expected /var/tmp (link parent) in: {write_line}");
    assert!(!link.exists(), "COW isolation failed: symlink created on real FS");
}

/// Hard link creates are captured by destination; source appears in reads.
/// COW ensures the real filesystem is not modified.
#[test]
fn test_learn_captures_hardlink() {
    let src = tempfile::NamedTempFile::new_in("/var/tmp").expect("tempfile in /var/tmp");
    let src_path = src.path().to_str().unwrap().to_owned();
    let dst_dir = tempfile::TempDir::new_in("/tmp").expect("tempdir in /tmp");
    let dst = dst_dir.path().join("hardlink");
    let dst_str = dst.to_str().unwrap();
    let cmd = format!("ln {src_path} {dst_str}");
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &cmd])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(write_line.contains("/tmp"), "expected /tmp (dst parent) in: {write_line}");
    assert!(!write_line.contains("/var/tmp"), "src parent /var/tmp wrongly recorded as write in: {write_line}");
    assert!(read_line.contains(&src_path), "expected src {src_path} in reads: {read_line}");
    assert!(!dst.exists(), "COW isolation failed: hardlink created on real FS");
}

/// Truncate is captured as a write on the file path itself. COW ensures the real file is unchanged.
#[test]
fn test_learn_captures_truncate() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), "original content").expect("write");
    let path = tmp.path().to_str().unwrap().to_owned();
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", &format!("truncate -s 0 {path}")])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains(&path),
        "expected file path {path} in write = [...], got: {write_line}");
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(content, "original content", "COW isolation failed: real file was truncated");
}

// ── Network ───────────────────────────────────────────────────────────────────

/// TCP connections are recorded under [network] allow.
#[test]
fn test_learn_captures_net_connect() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let _t = std::thread::spawn(move || { let _ = listener.accept(); });

    let script = format!(
        "import socket; s=socket.socket(); s.connect(('127.0.0.1',{port})); s.close()"
    );
    let output = sandlock_bin()
        .args(["learn", "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected = format!("127.0.0.1:{port}");
    let net_line = stdout.lines().find(|l| l.starts_with("allow = [")).unwrap_or("");
    assert!(
        net_line.contains(&expected),
        "expected {expected} under [network] allow = [...], got: {net_line}",
    );
}

/// IPv6 endpoints are recorded with brackets: tcp://[::1]:port.
/// Skipped on hosts without IPv6 loopback.
#[test]
fn test_learn_captures_ipv6_connect() {
    let listener = match std::net::TcpListener::bind("[::1]:0") {
        Ok(l) => l,
        Err(_) => { eprintln!("skipped: IPv6 loopback unavailable"); return; }
    };
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || { let _ = listener.accept(); });

    let script = format!(
        "import socket; s=socket.socket(socket.AF_INET6); s.connect(('::1',{port})); s.close()"
    );
    let output = sandlock_bin()
        .args(["learn", "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected = format!("tcp://[::1]:{port}");
    let net_line = stdout.lines().find(|l| l.starts_with("allow = [")).unwrap_or("");
    assert!(net_line.contains(&expected),
        "expected {expected} under [network] allow = [...], got: {net_line}");
}

/// Bind calls are recorded under [network] allow_bind.
#[test]
fn test_learn_captures_bind() {
    let script = concat!(
        "import socket\n",
        "s = socket.socket()\n",
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n",
        "s.bind(('127.0.0.1', 19876))\n",
        "s.close()\n",
    );
    let output = sandlock_bin()
        .args(["learn", "--", "python3", "-c", script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(),
        "sandlock learn failed: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let bind_line = stdout.lines().find(|l| l.starts_with("allow_bind = [")).unwrap_or("");
    assert!(
        bind_line.contains("19876"),
        "expected 19876 in allow_bind, got: {bind_line}",
    );
}

/// UDP destinations via sendto and sendmsg are both recorded under [network] allow.
#[test]
fn test_learn_captures_udp() {
    let sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let port = sock.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let mut buf = [0u8; 16];
        let _ = sock.recv_from(&mut buf);
        let _ = sock.recv_from(&mut buf);
    });

    let script = format!(
        "import socket\n\
         s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n\
         s.sendto(b'hi',('127.0.0.1',{port}))\n\
         s.sendmsg([b'hi'],[],0,('127.0.0.1',{port}))\n\
         s.close()"
    );
    let output = sandlock_bin()
        .args(["learn", "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let expected = format!("udp://127.0.0.1:{port}");
    assert!(
        stdout.contains(&expected),
        "expected {expected} in network output, got:\n{stdout}",
    );
}

// ── Resource limits ───────────────────────────────────────────────────────────

/// Resource peaks are recorded under [limits].
#[test]
fn test_learn_captures_resource_limits() {
    let output = sandlock_bin()
        .args(["learn", "--", "sleep", "0.2"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("memory = \""), "expected memory limit in learn output, got:\n{stdout}");
    assert!(stdout.contains("processes = "), "expected processes limit in learn output, got:\n{stdout}");
    assert!(stdout.contains("open_files = "), "expected open_files limit in learn output, got:\n{stdout}");
}

// ── Path collapse and dedup ───────────────────────────────────────────────────

/// Dedup removes a path when an ancestor is already in the read set.
#[test]
fn test_read_dedup_removes_leaf_when_ancestor_present() {
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", "ls /etc > /dev/null && cat /etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(read_line.contains("/etc"), "expected /etc in reads, got: {read_line}");
    assert!(!read_line.contains("/etc/hostname"),
        "expected /etc/hostname removed by dedup (covered by /etc), got: {read_line}");
}

/// N-threshold collapse replaces individual files with their parent directory.
#[test]
fn test_collapse_threshold_reads() {
    let output = sandlock_bin()
        .args(["learn", "--collapse", "--", "sh", "-c",
               "cat /usr/bin/cat /usr/bin/sh /usr/bin/ls /usr/bin/env"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(
        read_line.contains("/usr/bin\"") || read_line.contains("/usr/bin,"),
        "expected /usr/bin collapsed in reads, got: {read_line}",
    );
    assert!(!read_line.contains("/usr/bin/cat"),
        "expected individual /usr/bin files removed after collapse, got: {read_line}");
}

/// --collapse-prefix forces collapse regardless of the file count threshold.
#[test]
fn test_collapse_prefix_forces_collapse() {
    let output = sandlock_bin()
        .args(["learn", "--collapse-prefix", "/usr/bin", "--", "cat", "/usr/bin/cat", "/usr/bin/sh"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(
        read_line.contains("/usr/bin\"") || read_line.contains("/usr/bin,"),
        "expected /usr/bin collapsed in reads, got: {read_line}",
    );
    assert!(!read_line.contains("/usr/bin/cat"),
        "expected /usr/bin/cat removed after prefix collapse, got: {read_line}");
}

/// Guarded paths are not collapsed by the N-threshold; individual files are kept.
#[test]
fn test_collapse_guarded_not_collapsed_by_threshold() {
    let output = sandlock_bin()
        .args(["learn", "--collapse", "--", "sh", "-c",
               "cat /etc/hostname /etc/hosts /etc/resolv.conf /etc/passwd"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(read_line.contains("/etc/hostname"),
        "expected individual /etc files kept (guarded, not collapsed), got: {read_line}");
}

/// --collapse-prefix on a guarded path requires --force-sensitive-collapse.
#[test]
fn test_collapse_prefix_guarded_requires_force() {
    let output = sandlock_bin()
        .args(["learn", "--collapse-prefix", "/etc", "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(!output.status.success(), "expected failure without --force-sensitive-collapse");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--force-sensitive-collapse"),
        "expected hint about --force-sensitive-collapse in stderr, got: {stderr}");
}

/// --collapse-prefix on a guarded path with --force-sensitive-collapse collapses and emits a warning.
#[test]
fn test_collapse_prefix_guarded_with_force() {
    let output = sandlock_bin()
        .args(["learn", "--collapse-prefix", "/etc", "--force-sensitive-collapse",
               "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let read_line = stdout.lines().find(|l| l.starts_with("read = [")).unwrap_or("");
    assert!(
        read_line.contains("/etc\"") || read_line.contains("/etc,"),
        "expected /etc in reads after forced collapse, got: {read_line}",
    );
    assert!(!read_line.contains("/etc/hostname"),
        "expected /etc/hostname removed after collapse, got: {read_line}");
    assert!(stderr.contains("guarded directory"),
        "expected guarded directory warning, got: {stderr}");
}

/// Write collapse to filesystem root is skipped entirely.
#[test]
fn test_write_collapse_skips_root() {
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", "echo x > /sandlock_learn_root_test_$$"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(!write_line.contains("\"/\""), "write list must not contain \"/\", got: {write_line}");
    assert!(stderr.contains("filesystem root"),
        "expected 'filesystem root' warning in stderr, got: {stderr}");
}

/// Write collapse to a sensitive path emits a warning and observed-vs-granted diff.
#[test]
fn test_write_collapse_warns_sensitive() {
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", "echo x > /etc/sandlock_learn_sensitive_test_$$"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(
        output.status.success(),
        "sandlock learn failed: stderr={}",
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(write_line.contains("/etc"), "expected /etc in write list, got: {write_line}");
    assert!(stderr.contains("guarded directory"),
        "expected 'guarded directory' warning in stderr, got: {stderr}");
    assert!(stderr.contains("unobserved siblings now writable under"),
        "expected observed-vs-granted diff in stderr, got: {stderr}");
}

/// Write collapse to a protected path is skipped entirely.
#[test]
fn test_write_collapse_skips_protected() {
    let output = sandlock_bin()
        .args(["learn", "--", "sh", "-c", "echo x > /root/sandlock_learn_protected_test_$$"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(output.status.success(), "stderr={}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let write_line = stdout.lines().find(|l| l.starts_with("write = [")).unwrap_or("");
    assert!(!write_line.contains("/root\"") && !write_line.contains("/root,"),
        "write list must not contain /root (protected), got: {write_line}");
    assert!(stderr.contains("protected path"),
        "expected 'protected path' error in stderr, got: {stderr}");
}

// ── Merge and canonicalization ────────────────────────────────────────────────

/// --merge unions observations from a new run into an existing profile.
#[test]
fn test_learn_merge_unions_profiles() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let learn1 = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn1.status.success(),
        "first learn failed: {}", String::from_utf8_lossy(&learn1.stderr));

    let learn2 = sandlock_bin()
        .args(["learn", "--merge", &profile_path, "--", "cat", "/etc/os-release"])
        .output()
        .expect("failed to run sandlock learn --merge");
    assert!(learn2.status.success(),
        "merge learn failed: {}", String::from_utf8_lossy(&learn2.stderr));

    let merged = std::fs::read_to_string(&profile_path).expect("read merged profile");
    assert!(merged.contains("/etc/hostname") || merged.contains("/etc"),
        "merged profile missing /etc/hostname: {merged}");
    assert!(merged.contains("/etc/os-release") || merged.contains("/etc"),
        "merged profile missing /etc/os-release: {merged}");
    assert!(merged.contains("hostname"),
        "merged profile must keep [program] from first run: {merged}");
}

/// Paths accessed through symlinks are recorded as their canonical real path.
#[test]
fn test_learn_symlink_path_canonicalized() {
    let real_dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir");
    let file = real_dir.path().join("data.txt");
    std::fs::write(&file, "hello").expect("write file");
    let real_path = file.to_str().unwrap().to_owned();

    let link = format!("/tmp/sandlock-e7-link-{}", std::process::id());
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(real_dir.path(), &link).expect("create symlink");
    let via_link = format!("{link}/data.txt");

    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "cat", &via_link])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let profile_content = std::fs::read_to_string(&profile_path).expect("read profile");
    assert!(
        profile_content.contains(real_dir.path().to_str().unwrap()),
        "profile must contain canonical path {}, got:\n{profile_content}",
        real_dir.path().display()
    );

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "cat", &real_path])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run via real path failed (canonicalization not applied): {}",
        String::from_utf8_lossy(&run.stderr));

    let _ = std::fs::remove_file(&link);
}
