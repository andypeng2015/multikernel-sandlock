use std::io;
use super::{Checkpoint, ProcessState, MemorySegment, MemoryMap, FdInfo};
use crate::sandbox::Sandbox;
use crate::error::{SandlockError, SandboxRuntimeError};

// ---------------------------------------------------------------------------
// ptrace helpers -- PTRACE_SEIZE (doesn't auto-SIGSTOP like ATTACH)
// ---------------------------------------------------------------------------

pub(crate) fn ptrace_seize(pid: i32) -> io::Result<()> {
    let ret = unsafe {
        libc::ptrace(libc::PTRACE_SEIZE as libc::c_uint, pid, 0, 0)
    };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    // PTRACE_INTERRUPT stops the tracee without SIGSTOP side effects
    let ret = unsafe {
        libc::ptrace(libc::PTRACE_INTERRUPT as libc::c_uint, pid, 0, 0)
    };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    // Wait for the ptrace-stop
    let mut status: i32 = 0;
    unsafe {
        libc::waitpid(pid, &mut status, 0);
    }
    Ok(())
}

pub(crate) fn ptrace_detach(pid: i32) -> io::Result<()> {
    let ret = unsafe { libc::ptrace(libc::PTRACE_DETACH, pid, 0, 0) };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(crate) fn ptrace_getregs(pid: i32) -> io::Result<Vec<u64>> {
    #[cfg(target_arch = "x86_64")]
    {
        // user_regs_struct is 27 u64 fields on x86_64 (216 bytes)
        let mut regs = vec![0u64; 27];
        let ret = unsafe { libc::ptrace(libc::PTRACE_GETREGS, pid, 0, regs.as_mut_ptr()) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(regs)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Linux arm64 exposes general-purpose registers through
        // PTRACE_GETREGSET/NT_PRSTATUS. user_pt_regs is:
        // x0-x30, sp, pc, pstate (34 u64 values).
        const NT_PRSTATUS: libc::c_int = 1;
        let mut regs = vec![0u64; 34];
        let mut iov = libc::iovec {
            iov_base: regs.as_mut_ptr() as *mut libc::c_void,
            iov_len: regs.len() * std::mem::size_of::<u64>(),
        };
        let ret = unsafe {
            libc::ptrace(
                libc::PTRACE_GETREGSET,
                pid,
                NT_PRSTATUS as usize as *mut libc::c_void,
                &mut iov as *mut libc::iovec as *mut libc::c_void,
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        regs.truncate(iov.iov_len / std::mem::size_of::<u64>());
        Ok(regs)
    }

    #[cfg(target_arch = "riscv64")]
    {
        // Linux riscv64 exposes general-purpose registers through
        // PTRACE_GETREGSET/NT_PRSTATUS. struct user_regs_struct is:
        // pc, ra, sp, gp, tp, t0-t2, s0-s1, a0-a7, s2-s11, t3-t6
        // (32 u64 values; x0 is hardwired zero and not stored).
        const NT_PRSTATUS: libc::c_int = 1;
        let mut regs = vec![0u64; 32];
        let mut iov = libc::iovec {
            iov_base: regs.as_mut_ptr() as *mut libc::c_void,
            iov_len: regs.len() * std::mem::size_of::<u64>(),
        };
        let ret = unsafe {
            libc::ptrace(
                libc::PTRACE_GETREGSET,
                pid,
                NT_PRSTATUS as usize as *mut libc::c_void,
                &mut iov as *mut libc::iovec as *mut libc::c_void,
            )
        };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        regs.truncate(iov.iov_len / std::mem::size_of::<u64>());
        Ok(regs)
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        let _ = pid;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "checkpoint register capture is not implemented on this architecture",
        ))
    }
}

// ---------------------------------------------------------------------------
// FPU/extended register capture via PTRACE_GETREGSET
// ---------------------------------------------------------------------------

fn ptrace_getregset_bytes(pid: i32, set: libc::c_int, max: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; max];
    let mut iov = libc::iovec {
        iov_base: buf.as_mut_ptr() as *mut libc::c_void,
        iov_len: buf.len(),
    };
    let ret = unsafe {
        libc::ptrace(
            libc::PTRACE_GETREGSET,
            pid,
            set as usize as *mut libc::c_void,
            &mut iov as *mut libc::iovec as *mut libc::c_void,
        )
    };
    if ret < 0 {
        return Err(io::Error::last_os_error());
    }
    buf.truncate(iov.iov_len.min(buf.len()));
    Ok(buf)
}

fn ptrace_getfpregs(pid: i32) -> io::Result<Vec<u8>> {
    // NT_PRFPREG = 2; NT_X86_XSTATE = 0x202.
    //
    // The buffer must be able to hold the CPU's *whole* user xstate. The kernel
    // fills what fits and reports how much it wrote, so a buffer that is merely
    // "big enough for the CPUs we had in mind" yields a silently truncated
    // capture: 8 KiB covers AVX-512 (~2.7 KiB) but not AMX, whose tile data
    // alone is 8 KiB. That truncation is invisible here and lethal at restore,
    // where the image is framed for the signal frame's `xrstor` as if complete.
    // 64 KiB is far above any current CPU's `CPUID.(EAX=0DH,ECX=0):EBX`.
    #[cfg(target_arch = "x86_64")]
    { ptrace_getregset_bytes(pid, 0x202, 65536).or_else(|_| ptrace_getregset_bytes(pid, 2, 512)) }
    #[cfg(not(target_arch = "x86_64"))]
    { ptrace_getregset_bytes(pid, 2, 4096) }
}

// ---------------------------------------------------------------------------
// /proc parsing
// ---------------------------------------------------------------------------

pub(crate) fn parse_proc_maps(pid: i32) -> io::Result<Vec<MemoryMap>> {
    let content = std::fs::read_to_string(format!("/proc/{}/maps", pid))?;
    let mut maps = Vec::new();
    for line in content.lines() {
        // Format: start-end perms offset dev inode [pathname]
        let parts: Vec<&str> = line.splitn(6, ' ').collect();
        if parts.len() < 5 {
            continue;
        }
        let addrs: Vec<&str> = parts[0].split('-').collect();
        if addrs.len() != 2 {
            continue;
        }
        let start = u64::from_str_radix(addrs[0], 16).unwrap_or(0);
        let end = u64::from_str_radix(addrs[1], 16).unwrap_or(0);
        let perms = parts[1].to_string();
        let offset = u64::from_str_radix(parts[2], 16).unwrap_or(0);
        let path = if parts.len() >= 6 {
            let p = parts[5].trim();
            if p.is_empty() {
                None
            } else {
                Some(p.to_string())
            }
        } else {
            None
        };
        maps.push(MemoryMap {
            start,
            end,
            perms,
            offset,
            path,
        });
    }
    Ok(maps)
}

// ---------------------------------------------------------------------------
// Memory capture -- process_vm_readv (scatter-gather, no file I/O)
// ---------------------------------------------------------------------------

/// Largest single region capture will dump.
///
/// A resource guard, not a supported-size statement: the buffer for a region is
/// allocated whole, so one pathological mapping could otherwise exhaust the
/// supervisor. It does not bound the image, which is the sum of every dumped
/// region. Raise it if real workloads need larger regions; exceeding it is a
/// refusal to checkpoint, never a partial image.
const MAX_REGION_BYTES: usize = 256 * 1024 * 1024;

/// Whether a region's bytes have to travel inside the image, because restore
/// cannot obtain them from anywhere else.
///
/// The bytes are needed unless some file still holds them at restore time:
///
/// * Kernel special mappings are provided by the kernel, never rebuilt.
/// * A read-only region backed by a real file is remapped from that file, which
///   is cheaper and shares pages.
/// * A *shared* writable region backed by a real file wrote through to it, so
///   remapping from that file recovers the current contents.
///
/// Everything else is dumped, including shared **anonymous** regions. Those have
/// no backing file, so their contents exist only in the process; skipping them
/// loses the data outright. They come back private, since the restore plan has
/// no shared arm, which trades sharing for contents. A region backed by a memfd
/// or a "(deleted)" path counts as anonymous here: the path is real in `/proc`
/// but nothing can reopen it, and that includes the program text of a workload
/// exec'd from a memfd.
fn must_dump(map: &MemoryMap) -> bool {
    if map.is_special() {
        return false;
    }
    let unreopenable = map
        .path
        .as_deref()
        .map_or(false, |p| p.starts_with("/memfd:") || p.ends_with(" (deleted)"));
    if !map.writable() && !unreopenable {
        return false;
    }
    let reopenable_file =
        map.path.as_deref().map_or(false, |p| p.starts_with('/')) && !unreopenable;
    if !map.private() && reopenable_file {
        return false;
    }
    true
}

fn capture_memory(pid: i32, maps: &[MemoryMap]) -> Result<Vec<MemorySegment>, SandlockError> {
    let refuse = |msg: String| {
        SandlockError::Runtime(SandboxRuntimeError::Child(format!(
            "cannot checkpoint: {msg}"
        )))
    };
    let mut segments = Vec::new();

    for map in maps {
        if !must_dump(map) {
            continue;
        }
        // Past this point the region's bytes ARE the image. Nothing else can
        // supply them at restore: with no captured segment and no reopenable
        // absolute path, `restore_blob::build_memory_plan` omits the region
        // altogether, so the restored process comes up with that mapping simply
        // absent and dies at whatever unrelated point first touches it. Skipping
        // quietly here buys a checkpoint that reports success and cannot work,
        // which is strictly worse than refusing to take one.
        let size = (map.end - map.start) as usize;
        if size > MAX_REGION_BYTES {
            return Err(refuse(format!(
                "region {:#x}-{:#x} is {} MiB, over the {} MiB per-region capture limit \
                 (MAX_REGION_BYTES); its contents exist nowhere else, so the image would \
                 be missing that memory",
                map.start,
                map.end,
                size / (1024 * 1024),
                MAX_REGION_BYTES / (1024 * 1024),
            )));
        }

        let mut data = vec![0u8; size];

        let local_iov = libc::iovec {
            iov_base: data.as_mut_ptr() as *mut libc::c_void,
            iov_len: size,
        };
        let remote_iov = libc::iovec {
            iov_base: map.start as *mut libc::c_void,
            iov_len: size,
        };

        let ret = unsafe {
            libc::process_vm_readv(
                pid as libc::pid_t,
                &local_iov as *const libc::iovec,
                1,
                &remote_iov as *const libc::iovec,
                1,
                0,
            )
        };

        if ret != size as isize {
            let why = if ret < 0 {
                io::Error::last_os_error().to_string()
            } else {
                format!("read {ret} of {size} bytes")
            };
            return Err(refuse(format!(
                "reading region {:#x}-{:#x}: {why}",
                map.start, map.end,
            )));
        }
        segments.push(MemorySegment {
            start: map.start,
            data,
        });
    }
    Ok(segments)
}

// ---------------------------------------------------------------------------
// FD table capture
// ---------------------------------------------------------------------------

fn capture_fd_table(pid: i32) -> io::Result<Vec<FdInfo>> {
    let fd_dir = format!("/proc/{}/fd", pid);
    let mut fds = Vec::new();

    for entry in std::fs::read_dir(&fd_dir)? {
        let entry = entry?;
        let fd_str = entry.file_name().into_string().unwrap_or_default();
        let fd: i32 = match fd_str.parse() {
            Ok(f) => f,
            Err(_) => continue,
        };

        let path = std::fs::read_link(entry.path())
            .map(|p| p.display().to_string())
            .unwrap_or_default();

        // Parse fdinfo for flags and offset
        let (flags, offset) = parse_fdinfo(pid, fd).unwrap_or((0, 0));

        fds.push(FdInfo {
            fd,
            path,
            flags,
            offset,
        });
    }

    fds.sort_by_key(|f| f.fd);
    Ok(fds)
}

fn parse_fdinfo(pid: i32, fd: i32) -> io::Result<(i32, u64)> {
    let content = std::fs::read_to_string(format!("/proc/{}/fdinfo/{}", pid, fd))?;
    let mut flags = 0i32;
    let mut pos = 0u64;
    for line in content.lines() {
        if let Some(val) = line.strip_prefix("flags:\t") {
            flags = i32::from_str_radix(val.trim(), 8).unwrap_or(0);
        }
        if let Some(val) = line.strip_prefix("pos:\t") {
            pos = val.trim().parse().unwrap_or(0);
        }
    }
    Ok((flags, pos))
}

// ---------------------------------------------------------------------------
// Main capture function
// ---------------------------------------------------------------------------

/// Capture a checkpoint from a running, stopped sandbox.
/// The sandbox must already be frozen (SIGSTOP'd and fork-held).
pub(crate) fn capture(pid: i32, policy: &Sandbox) -> Result<Checkpoint, SandlockError> {
    // Credential-injection rules hold supervisor-only secrets that are
    // deliberately not serialized (`#[serde(skip)]`). A checkpoint image would
    // therefore restore with no injection rules and send every request
    // uncredentialed — reject rather than silently drop them.
    if !policy.inject.is_empty() {
        return Err(SandlockError::Runtime(SandboxRuntimeError::Child(
            "checkpoint is not supported with credential injection (--http-auth); \
             the injected secrets cannot be serialized into the image"
                .into(),
        )));
    }

    // Seize via ptrace (PTRACE_SEIZE + PTRACE_INTERRUPT -- doesn't auto-SIGSTOP)
    ptrace_seize(pid).map_err(|e| {
        SandlockError::Runtime(SandboxRuntimeError::Child(format!("ptrace seize: {}", e)))
    })?;

    // Everything up to the detach reads the frozen process, and any of it can
    // fail (a region too large to dump, an unreadable mapping, a /proc read
    // racing the workload). Gather it all first and detach unconditionally:
    // returning early here would leave the workload stopped under a tracer that
    // is walking away, which turns a failed checkpoint into a hung sandbox.
    let captured = (|| {
        let regs = ptrace_getregs(pid).map_err(|e| {
            SandlockError::Runtime(SandboxRuntimeError::Child(format!("ptrace getregs: {}", e)))
        })?;
        // FP state is best-effort: an image without it still restores.
        let fpregs = ptrace_getfpregs(pid).unwrap_or_default();
        let maps =
            parse_proc_maps(pid).map_err(|e| SandlockError::Runtime(SandboxRuntimeError::Io(e)))?;
        let memory_data = capture_memory(pid, &maps)?;
        let fd_table = capture_fd_table(pid)
            .map_err(|e| SandlockError::Runtime(SandboxRuntimeError::Io(e)))?;
        Ok::<_, SandlockError>((regs, fpregs, maps, memory_data, fd_table))
    })();

    let detached = ptrace_detach(pid);
    // Surface a capture failure first: it is the reason the caller is here, and
    // a detach error on top of it is noise.
    let (regs, fpregs, maps, memory_data, fd_table) = captured?;
    detached.map_err(|e| {
        SandlockError::Runtime(SandboxRuntimeError::Child(format!("ptrace detach: {}", e)))
    })?;

    // Capture cwd and exe from /proc
    let cwd = std::fs::read_link(format!("/proc/{}/cwd", pid))
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let exe = std::fs::read_link(format!("/proc/{}/exe", pid))
        .map(|p| p.display().to_string())
        .unwrap_or_default();

    Ok(Checkpoint {
        name: String::new(),
        policy: policy.clone(),
        process_state: ProcessState {
            pid,
            cwd,
            exe,
            regs,
            fpregs,
            memory_maps: maps,
            memory_data,
        },
        fd_table,
        cow_snapshot: None,
        app_state: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// `ptrace_getregs` captures a full register file with a plausible,
    /// non-zero program counter from a live, seized child on the host
    /// architecture. This exercises the architecture-specific register
    /// capture path without requiring a full sandbox launch (no Landlock).
    #[test]
    fn ptrace_getregs_captures_program_counter() {
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep child");
        let pid = child.id() as i32;

        let result = (|| -> io::Result<Vec<u64>> {
            ptrace_seize(pid)?;
            let regs = ptrace_getregs(pid)?;
            ptrace_detach(pid)?;
            Ok(regs)
        })();

        let _ = child.kill();
        let _ = child.wait();

        let regs = result.expect("register capture should succeed on this architecture");

        // Architecture-specific register-file width.
        #[cfg(target_arch = "x86_64")]
        assert_eq!(regs.len(), 27, "x86_64 user_regs_struct is 27 u64");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(regs.len(), 34, "aarch64 user_pt_regs is 34 u64");
        #[cfg(target_arch = "riscv64")]
        assert_eq!(regs.len(), 32, "riscv64 user_regs_struct is 32 u64");

        // The program counter must be a non-zero userspace address; its index
        // into the register file differs per architecture.
        #[cfg(target_arch = "x86_64")]
        let pc = regs[16]; // rip
        #[cfg(target_arch = "aarch64")]
        let pc = regs[32]; // pc, after x0-x30 and sp
        #[cfg(target_arch = "riscv64")]
        let pc = regs[0]; // pc is first in riscv user_regs_struct

        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        assert!(pc != 0, "captured program counter should be non-zero, got {:#x}", pc);
    }

    #[test]
    fn ptrace_getfpregs_captures_nonempty_state() {
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id() as i32;
        let res = (|| -> io::Result<Vec<u8>> {
            ptrace_seize(pid)?;
            let fp = ptrace_getfpregs(pid)?;
            ptrace_detach(pid)?;
            Ok(fp)
        })();
        let _ = child.kill();
        let _ = child.wait();
        let fp = res.expect("fpreg capture should succeed");
        assert!(!fp.is_empty(), "captured FP/extended register blob should be non-empty");
    }

    /// Full capture -> save -> load roundtrip against a live child. `capture()`
    /// only ptraces and reads `/proc`, so this exercises the architecture-specific
    /// register arm plus the on-disk save/load format end to end WITHOUT a sandbox
    /// launch (no Landlock) -- the coverage the sandbox-launch integration test
    /// cannot provide on kernels below the required Landlock ABI.
    fn region(perms: &str, path: Option<&str>) -> MemoryMap {
        MemoryMap {
            start: 0x1000,
            end: 0x2000,
            perms: perms.into(),
            offset: 0,
            path: path.map(Into::into),
        }
    }

    #[test]
    fn shared_anonymous_regions_are_dumped_like_private_ones() {
        // Nothing backs these, so their contents live only in the process. They
        // return private, trading sharing for the data itself.
        assert!(must_dump(&region("rw-s", None)), "shared anonymous");
        assert!(must_dump(&region("rw-s", Some("/memfd:scratch (deleted)"))), "shared memfd");
        assert!(must_dump(&region("rw-s", Some("/SYSV00000000 (deleted)"))), "SysV shm");
        assert!(must_dump(&region("rw-p", None)), "private anonymous");
    }

    #[test]
    fn regions_a_file_still_holds_are_left_to_be_remapped() {
        // Dumping these would bloat every image with bytes restore can read off
        // disk, and could push a checkpoint over the per-region ceiling.
        assert!(!must_dump(&region("rw-s", Some("/data/db"))),
            "a shared writable file mapping wrote through to the file");
        assert!(!must_dump(&region("r--p", Some("/lib/libc.so"))),
            "a read-only file mapping is remapped from the file");
        assert!(!must_dump(&region("r-xp", Some("/bin/app"))), "program text");
        assert!(!must_dump(&region("r--s", Some("/lib/libc.so"))), "shared read-only file");
    }

    #[test]
    fn kernel_special_mappings_are_never_dumped() {
        for name in ["[vdso]", "[vvar]", "[vvar_vclock]", "[vsyscall]"] {
            assert!(!must_dump(&region("rw-p", Some(name))), "{name}");
        }
    }

    #[test]
    fn a_dirtied_private_file_mapping_is_dumped() {
        // Its contents no longer match the file on disk, so the file cannot
        // supply them at restore.
        assert!(must_dump(&region("rw-p", Some("/lib/libc.so"))));
    }

    /// A region capture cannot dump must refuse the checkpoint, and must still
    /// let the workload go.
    ///
    /// Skipping the region instead would produce an image that reports success
    /// and cannot work: with no captured bytes and no reopenable path, restore
    /// omits the mapping entirely and the resumed process dies wherever it first
    /// touches that memory, a long way from the cause.
    #[test]
    fn capture_refuses_a_region_it_cannot_dump_and_still_detaches() {
        // The child maps past the limit and parks. The pages are never touched,
        // so this costs address space rather than memory.
        let child = unsafe { libc::fork() };
        if child == 0 {
            unsafe {
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    MAX_REGION_BYTES + 4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if p == libc::MAP_FAILED {
                    libc::_exit(1);
                }
                loop {
                    libc::pause();
                }
            }
        }
        assert!(child > 0, "fork");
        unsafe { libc::usleep(100_000) }; // let the child finish its mmap

        let policy = Sandbox::builder().build().expect("build policy");
        let result = capture(child, &policy);

        // Read the child's state while it is still alive: a capture that bailed
        // out without detaching would leave it in 't' (tracing stop) forever.
        let state = std::fs::read_to_string(format!("/proc/{child}/stat"))
            .ok()
            .and_then(|s| s.rsplit(") ").next().and_then(|t| t.split(' ').next()).map(String::from));

        unsafe {
            libc::kill(child, libc::SIGKILL);
            let mut s = 0;
            libc::waitpid(child, &mut s, 0);
        }

        let err = result.expect_err("a region over the dump limit must refuse").to_string();
        assert!(
            err.contains("MAX_REGION_BYTES"),
            "the refusal should name the limit that caused it: {err}"
        );
        // Anything but a stop ('t' tracing stop, 'T' stopped) means the tracer
        // let go; whether the child is scheduled or sleeping is up to the kernel.
        let state = state.expect("read child state");
        assert!(
            state != "t" && state != "T",
            "a refused capture must still detach, leaving the workload runnable; state {state}"
        );
    }

    /// End-to-end: a live MAP_SHARED|MAP_ANONYMOUS region's contents reach the
    /// image. Classification alone cannot show this, because the region has to
    /// survive `/proc/<pid>/maps` parsing (where it appears as `rw-s` with no
    /// path) as well as the dump decision.
    #[test]
    fn shared_anonymous_contents_reach_the_image() {
        const PAT: u8 = 0x3C;

        // The kernel picks the address and the child reports it back. Naming one
        // here would be a bet on the architecture's virtual address layout: the
        // high addresses that are free on x86_64 do not exist on riscv64, whose
        // Sv39 user space ends at 256 GiB, so a MAP_FIXED there fails and the
        // child dies before the test can look at it.
        let mut pipefd = [0i32; 2];
        assert_eq!(unsafe { libc::pipe(pipefd.as_mut_ptr()) }, 0);
        let (r, w) = (pipefd[0], pipefd[1]);

        let child = unsafe { libc::fork() };
        if child == 0 {
            unsafe {
                libc::close(r);
                let p = libc::mmap(
                    std::ptr::null_mut(),
                    4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if p == libc::MAP_FAILED {
                    libc::_exit(1);
                }
                let mut i = 0usize;
                while i < 4096 {
                    *(p as *mut u8).add(i) = PAT;
                    i += 1;
                }
                let addr = p as u64;
                libc::write(w, &addr as *const u64 as *const libc::c_void, 8);
                loop {
                    libc::pause();
                }
            }
        }
        assert!(child > 0, "fork");
        unsafe { libc::close(w) };

        let mut addr_buf = [0u8; 8];
        let n = unsafe { libc::read(r, addr_buf.as_mut_ptr() as *mut libc::c_void, 8) };
        let addr = u64::from_le_bytes(addr_buf);

        let policy = Sandbox::builder().build().expect("build policy");
        let captured = capture(child, &policy);

        unsafe {
            libc::kill(child, libc::SIGKILL);
            let mut s = 0;
            libc::waitpid(child, &mut s, 0);
            libc::close(r);
        }

        assert_eq!(n, 8, "the child must report where the kernel mapped its region");
        let cp = captured.expect("capture");
        let seg = cp
            .process_state
            .memory_data
            .iter()
            .find(|s| s.start == addr)
            .expect("the shared anonymous region must be in the image");
        assert!(
            seg.data.len() >= 4096 && seg.data[..4096].iter().all(|&b| b == PAT),
            "its contents must be captured, not just its address"
        );
    }

    #[test]
    fn capture_save_load_roundtrips() {
        let mut child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep child");
        let pid = child.id() as i32;

        let policy = Sandbox::builder().build().expect("build policy");
        let captured = capture(pid, &policy);

        let _ = child.kill();
        let _ = child.wait();

        let cp = captured.expect("capture should succeed on this architecture");
        assert!(!cp.process_state.regs.is_empty(), "captured registers");
        assert!(!cp.process_state.memory_maps.is_empty(), "captured memory maps");
        assert!(!cp.fd_table.is_empty(), "captured fd table");

        // Save to a temp dir, load it back, and confirm the round-trip is faithful.
        let dir = std::env::temp_dir()
            .join(format!("sandlock-cp-roundtrip-{}", std::process::id()));
        cp.save(&dir).expect("save checkpoint");
        let loaded = Checkpoint::load(&dir).expect("load checkpoint");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(loaded.process_state.regs, cp.process_state.regs, "regs roundtrip");
        assert_eq!(loaded.process_state.fpregs, cp.process_state.fpregs, "fpregs roundtrip");
        assert_eq!(
            loaded.process_state.memory_data.len(),
            cp.process_state.memory_data.len(),
            "memory segment count roundtrip"
        );
        assert_eq!(loaded.fd_table.len(), cp.fd_table.len(), "fd count roundtrip");
        assert_eq!(loaded.process_state.pid, cp.process_state.pid, "pid roundtrip");
        assert!(!loaded.process_state.exe.is_empty(), "exe path captured");
    }
}
