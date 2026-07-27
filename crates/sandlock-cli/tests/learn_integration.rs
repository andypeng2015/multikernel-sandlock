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

/// AF_UNIX bind: socket parent observed during learn, bind succeeds during run.
#[test]
fn test_learn_then_run_unix_bind() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let sock = dir.path().join("rndtrip.sock");
    let sock_str = sock.to_str().unwrap();
    let script = format!(
        "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); \
         s.bind('{sock_str}'); s.close()"
    );

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    // bind() during learn is executed on-behalf by the supervisor (outside the
    // COW overlay), so the socket file exists on the real FS after learn.
    // Remove it so run can bind to the same path without EADDRINUSE.
    let _ = std::fs::remove_file(&sock);

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run failed (unix bind parent not in writes?): {}",
        String::from_utf8_lossy(&run.stderr));
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

/// Learned memory limit reflects actual anonymous allocation, not just the floor.
#[test]
fn test_learn_then_run_memory_limit() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    // Allocate 20 MiB, learned limit must exceed the 16M floor.
    let script = "import time; x = bytearray(20 * 1024 * 1024); time.sleep(0.05)";
    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "python3", "-c", script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(), "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let profile_toml = std::fs::read_to_string(&profile_path).expect("read profile");
    let mem_line = profile_toml.lines().find(|l| l.starts_with("memory")).unwrap();
    let mib: u64 = mem_line.trim_end_matches('"').split('"').nth(1)
        .and_then(|s| s.trim_end_matches('M').parse().ok()).unwrap_or(0);
    assert!(mib >= 25, "expected memory >= 25M (20M + 25% headroom), got {mib}M");

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "python3", "-c", script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
}

/// Learned process limit reflects observed child processes.
#[test]
fn test_learn_then_run_process_limit() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    // sh forks cat as a child, observed peak is 2 processes.
    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "sh", "-c", "cat /etc/hostname & wait"])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(), "learn failed: {}", String::from_utf8_lossy(&learn.stderr));

    let profile_toml = std::fs::read_to_string(&profile_path).expect("read profile");
    let proc_line = profile_toml.lines().find(|l| l.starts_with("processes")).unwrap();
    let procs: u32 = proc_line.split('=').nth(1)
        .and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    assert!(procs >= 4, "expected processes >= 4 (2 observed * 2), got {procs}");

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "sh", "-c", "cat /etc/hostname & wait"])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(), "run failed: {}", String::from_utf8_lossy(&run.stderr));
}

/// mknod: FIFO created during learn round-trips to run.
#[test]
fn test_learn_then_run_mknod() {
    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();
    let dir = tempfile::TempDir::new_in("/var/tmp").expect("tempdir in /var/tmp");
    let fifo = dir.path().join("rndtrip.fifo");
    let fifo_str = fifo.to_str().unwrap();
    let script = format!("import os; os.mkfifo('{fifo_str}')");

    let learn = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn.status.success(),
        "learn failed: {}", String::from_utf8_lossy(&learn.stderr));
    assert!(!fifo.exists(), "COW isolation failed during learn");

    let run = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--", "python3", "-c", &script])
        .output()
        .expect("failed to run sandlock run");
    assert!(run.status.success(),
        "run failed (mknod not in profile writes?): {}", String::from_utf8_lossy(&run.stderr));
    assert!(fifo.exists(), "FIFO not created during run");
    let _ = std::fs::remove_file(&fifo);
}

/// Merged profile covers paths from both learn runs.
#[test]
fn test_learn_then_run_merge() {
    // Two separate workdirs so paths don't collapse into a shared ancestor.
    let dir_a = tempfile::TempDir::new_in("/var/tmp").expect("tempdir a");
    let dir_b = tempfile::TempDir::new_in("/var/tmp").expect("tempdir b");
    let file_a = dir_a.path().join("input.txt");
    let file_b = dir_b.path().join("input.txt");
    let secret = dir_a.path().join("secret.txt");
    std::fs::write(&file_a, "data-a").expect("write file_a");
    std::fs::write(&file_b, "data-b").expect("write file_b");
    std::fs::write(&secret, "sensitive").expect("write secret");

    let profile = tempfile::NamedTempFile::new().expect("tempfile");
    let profile_path = profile.path().to_str().unwrap().to_owned();

    // Phase 1: learn access to dir_a/input.txt.
    let learn1 = sandlock_bin()
        .args(["learn", "-o", &profile_path, "--", "cat", file_a.to_str().unwrap()])
        .output()
        .expect("failed to run sandlock learn");
    assert!(learn1.status.success(), "learn1 failed: {}", String::from_utf8_lossy(&learn1.stderr));

    // Operator hardens the profile: deny the secret file in dir_a.
    // Insert deny into the existing [filesystem] section.
    let existing = std::fs::read_to_string(&profile_path).expect("read profile");
    let deny_line = format!("deny = [\"{}\"]", secret.display());
    let hardened = existing.replace("[filesystem]", &format!("[filesystem]\n{deny_line}"));
    std::fs::write(&profile_path, &hardened).expect("write hardened profile");

    // Phase 2: merge, learn access to dir_b/input.txt.
    let learn2 = sandlock_bin()
        .args(["learn", "--merge", &profile_path, "--", "cat", file_b.to_str().unwrap()])
        .output()
        .expect("failed to run sandlock learn --merge");
    assert!(learn2.status.success(), "learn2 failed: {}", String::from_utf8_lossy(&learn2.stderr));

    // Both files are readable, paths from both learn runs are in the profile.
    let run_ok = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--",
               "cat", file_a.to_str().unwrap(), file_b.to_str().unwrap()])
        .output()
        .expect("failed to run sandlock run");
    assert!(run_ok.status.success(),
        "run with merged profile failed: {}", String::from_utf8_lossy(&run_ok.stderr));

    // The deny rule survived the merge: the secret is blocked.
    let run_deny = sandlock_bin()
        .args(["run", "--profile-file", &profile_path, "--",
               "cat", secret.to_str().unwrap()])
        .output()
        .expect("failed to run sandlock run (deny check)");
    assert!(!run_deny.status.success(),
        "secret should be blocked by deny rule after merge");
}
