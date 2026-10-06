use serde::{Deserialize, Serialize};

const DEFAULT_MEMORY_LIMIT_RATIO: f64 = 0.8;
const MIN_MEMORY_LIMIT_BYTES: u64 = 100 * 1024 * 1024;

/// Smallest log file size in MB that config validation accepts.
pub const MIN_LOG_FILE_SIZE_MB: u32 = 128;
pub(crate) const DEFAULT_LOG_FILE_SIZE_MB: u32 = 2048;
pub(crate) const DEFAULT_LOG_OVERDUE_DAYS: u32 = 7;
pub(crate) const DEFAULT_LOG_DIR: &str = "./logs";
pub(crate) const DEFAULT_LOG_BASENAME: &str = "finch.log";

pub(crate) const DEFAULT_WAL_FLUSH_EVERY_DOCS: u32 = 1;
pub(crate) const DEFAULT_WAL_FSYNC_EVERY_DOCS: u32 = 0;

/// Lowest severity that the log writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum LogLevel {
    /// Debug messages and above.
    Debug = 0,
    /// Information messages and above.
    Info = 1,
    /// Warnings and above.
    #[default]
    Warn = 2,
    /// Errors only.
    Error = 3,
    /// Errors only; the log treats it as `Error`.
    Fatal = 4,
}

fn read_to_string(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn normalize_cgroup_memory_limit(v: u64) -> Option<u64> {
    // Some cgroup implementations use extremely large sentinel values to mean
    // "unlimited". Treat those as unknown so we can fall back to physical RAM.
    if v == 0 || v > (1u64 << 60) {
        return None;
    }
    Some(v)
}

#[cfg(unix)]
fn detect_physical_memory_bytes_unix() -> Option<u64> {
    // POSIX-ish physical memory detection via sysconf. Works on Linux and macOS.
    // If sysconf isn't available or returns -1, we fall back to other methods.
    // SAFETY: sysconf takes integer names only and has no memory preconditions.
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if pages <= 0 || page_size <= 0 {
            return None;
        }
        (pages as u64).checked_mul(page_size as u64)
    }
}

#[cfg(windows)]
fn detect_physical_memory_bytes_windows() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: `st` is a zeroed plain C struct with `dwLength` set, as GlobalMemoryStatusEx requires.
    unsafe {
        let mut st: MEMORYSTATUSEX = std::mem::zeroed();
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut st as *mut MEMORYSTATUSEX) == 0 {
            return None;
        }
        Some(st.ullTotalPhys)
    }
}

/// Best-effort memory limit detection.
///
/// Priority:
/// 1) cgroup memory limit (Linux containers)
/// 2) OS-reported physical memory
/// 3) `/proc/meminfo` total as a Unix fallback
pub fn detect_memory_limit_bytes() -> Option<u64> {
    // cgroup v2
    if let Some(s) = read_to_string("/sys/fs/cgroup/memory.max") {
        if s != "max" {
            if let Ok(v) = s.parse::<u64>() {
                if let Some(v) = normalize_cgroup_memory_limit(v) {
                    return Some(v);
                }
            }
        }
    }
    // cgroup v1
    if let Some(s) = read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        if let Ok(v) = s.parse::<u64>() {
            if let Some(v) = normalize_cgroup_memory_limit(v) {
                return Some(v);
            }
        }
    }

    #[cfg(unix)]
    if let Some(v) = detect_physical_memory_bytes_unix() {
        return Some(v);
    }

    #[cfg(windows)]
    if let Some(v) = detect_physical_memory_bytes_windows() {
        return Some(v);
    }

    // /proc fallback
    read_to_string("/proc/meminfo").and_then(|meminfo| meminfo_total_bytes(&meminfo))
}

fn meminfo_total_bytes(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("MemTotal:") else {
            continue;
        };
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() >= 2 && parts[1] == "kB" {
            if let Ok(kb) = parts[0].parse::<u64>() {
                return Some(kb.saturating_mul(1024));
            }
        }
    }
    None
}

fn detect_cpu_limit() -> Option<u32> {
    // cgroup v2: "quota period" (e.g. "200000 100000"); "max" means unlimited.
    if let Some(s) = read_to_string("/sys/fs/cgroup/cpu.max") {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] != "max" {
            if let (Ok(quota), Ok(period)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                if quota > 0 && period > 0 {
                    let v = quota.div_ceil(period);
                    return Some((v as u32).max(1));
                }
            }
        }
    }
    // cgroup v1
    let quota = read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")?
        .parse::<i64>()
        .ok()?;
    let period = read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us")?
        .parse::<i64>()
        .ok()?;
    if quota > 0 && period > 0 {
        let v = (quota as u64).div_ceil(period as u64);
        return Some((v as u32).max(1));
    }
    None
}

/// Global configuration data.
///
/// Finch uses these values as process-wide defaults. Most knobs can also be
/// overridden per-operation via options structs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GlobalConfigData {
    /// Soft memory budget in bytes (advisory; at least 100 MiB).
    pub memory_limit_bytes: u64,

    /// Log level (used only when `RUST_LOG` is not set).
    pub log_level: LogLevel,
    /// Whether to log to a rotating file under `log_dir`.
    pub log_to_file: bool,
    /// Log directory (used when `log_to_file=true`).
    pub log_dir: String,
    /// Base log filename (used when `log_to_file=true`).
    pub log_basename: String,
    /// Rotate log when the active file exceeds this size (MB).
    pub log_file_size_mb: u32,
    /// Remove rotated logs older than this many days.
    pub log_overdue_days: u32,

    /// Default query parallelism (Rayon global thread-pool size).
    pub query_thread_count: u32,

    /// Match ratio, from 0 to 1, above which a range filter scans the forward store
    /// instead of the inverted index. A larger value keeps the inverted index more often.
    pub invert_to_forward_scan_ratio: f32,

    /// Match ratio, from 0 to 1, below which a search scans only the doc ids that the
    /// inverted index matched. A larger value selects this scan more often.
    pub brute_force_by_keys_ratio: f32,

    /// Default optimize (compaction) parallelism.
    pub optimize_thread_count: u32,

    /// WAL: flush buffered records to the OS every N docs (0 disables flushing).
    ///
    /// Note: this controls userspace buffering. It does not guarantee durability
    /// unless paired with `wal_fsync_every_docs`.
    pub wal_flush_every_docs: u32,

    /// WAL: fsync (`sync_data`) every N docs (0 disables fsync).
    ///
    /// Larger values improve throughput but can lose recent writes on crash.
    pub wal_fsync_every_docs: u32,
}

impl Default for GlobalConfigData {
    fn default() -> Self {
        let n = detect_cpu_limit().unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
                .max(1)
        });

        let memory_limit_bytes = detect_memory_limit_bytes()
            .map(|v| ((v as f64) * DEFAULT_MEMORY_LIMIT_RATIO) as u64)
            .unwrap_or(MIN_MEMORY_LIMIT_BYTES);
        GlobalConfigData {
            memory_limit_bytes,
            log_level: LogLevel::Warn,
            log_to_file: false,
            log_dir: DEFAULT_LOG_DIR.to_string(),
            log_basename: DEFAULT_LOG_BASENAME.to_string(),
            log_file_size_mb: DEFAULT_LOG_FILE_SIZE_MB,
            log_overdue_days: DEFAULT_LOG_OVERDUE_DAYS,
            query_thread_count: n,
            invert_to_forward_scan_ratio: 0.9,
            brute_force_by_keys_ratio: 0.1,
            optimize_thread_count: n,
            wal_flush_every_docs: DEFAULT_WAL_FLUSH_EVERY_DOCS,
            wal_fsync_every_docs: DEFAULT_WAL_FSYNC_EVERY_DOCS,
        }
    }
}
