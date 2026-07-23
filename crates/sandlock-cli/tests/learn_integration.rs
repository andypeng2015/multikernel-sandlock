// Each test here spawns a full `sandlock` process.
// Running all tests in parallel exhausts kernel resources on most hosts.
// Limit parallelism when running this suite:
//   cargo test -p sandlock-cli --test learn_integration -- --test-threads=4

use std::process::Command;

fn sandlock_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_sandlock"))
}

/// Learn → run with a read-only workload.
#[test]
fn test_learn_then_run() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: stderr={}", String::from_utf8_lossy(&learn.stderr));

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run failed: stderr={}", String::from_utf8_lossy(&run.stderr));
    assert!(!String::from_utf8_lossy(&run.stdout).trim().is_empty(),
        "expected output from cat /etc/hostname");
}

/// Write path: COW isolates during learn; run creates the file for real.
#[test]
fn test_learn_then_run_write() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let write_dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let write_path = write_dir.path().join("run-write-test.txt");
    let write_path_str = write_path.to_str().unwrap();

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "sh", "-c", &format!("echo hello > {write_path_str}")])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: {}", String::from_utf8_lossy(&learn.stderr));
    assert!(!write_path.exists(), "COW isolation failed during learn");

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "sh", "-c", &format!("echo hello > {write_path_str}")])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(std::fs::read_to_string(&write_path).unwrap_or_default().trim(), "hello",
        "file not written during run");
    let _ = std::fs::remove_file(&write_path);
}

/// Write then read in the same run: both grants coexist without Landlock conflicts.
#[test]
fn test_learn_then_run_write_and_read() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let write_dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let file = write_dir.path().join("rw-test.txt");
    let file_str = file.to_str().unwrap();
    let script = format!("echo hello > {file_str} && cat {file_str}");

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "sh", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "sh", "-c", &script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "hello",
        "expected workload output");
    let _ = std::fs::remove_file(&file);
}

/// TCP connect: learned endpoint is allowed during run.
#[test]
fn test_learn_then_run_network() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || { let _ = listener.accept(); let _ = listener.accept(); });

    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let script = format!("import socket; s=socket.socket(); s.connect(('127.0.0.1',{port})); s.close()");

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(), "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
}

/// Bind: learned allow_bind port is permitted during run.
#[test]
fn test_learn_then_run_bind() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let script = concat!(
        "import socket\n",
        "s = socket.socket()\n",
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n",
        "s.bind(('127.0.0.1', 19878))\n",
        "s.close()\n",
    );

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "python3", "-c", script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(), "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "python3", "-c", script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
}

/// Collapsed profile: directory grant covers files not individually observed during learn.
#[test]
fn test_learn_then_run_collapse() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    // Learn touches several /usr/bin files; --collapse folds them into /usr/bin.
    let learn = sandlock_bin()
        .args(["learn", "--collapse", "-o", &profile_path, "--", "sh", "-c",
               "cat /usr/bin/cat /usr/bin/sh /usr/bin/ls /usr/bin/env"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(), "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    // Run accesses /usr/bin/true which was not individually observed, the collapsed grant covers it.
    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "cat", "/usr/bin/true"])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run failed with collapsed profile: {}", String::from_utf8_lossy(&run.stderr));
}

/// Merged profile covers paths from both learn runs.
#[test]
fn test_learn_then_run_merge() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    let learn1 = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "cat", "/etc/hostname"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn1.status.success(), "learn1 failed: {}", String::from_utf8_lossy(&learn1.stderr));

    let learn2 = sandlock_bin()
        .args(["learn", "--merge", &profile_path, "--", "cat", "/etc/os-release"])
        .output()
        .expect("failed to run sandlock learn --merge");
    assert!(learn2.status.success(), "learn2 failed: {}", String::from_utf8_lossy(&learn2.stderr));

    // Run needs both paths from both learn sessions.
    // Use `cat` directly, the profile's [program].exec is /usr/bin/cat (from learn1).
    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "cat",
               "/etc/hostname", "/etc/os-release"])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run with merged profile failed: {}", String::from_utf8_lossy(&run.stderr));
}
