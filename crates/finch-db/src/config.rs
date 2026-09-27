//! Global configuration for finch-db.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use finch_types::{GlobalConfigData, LogLevel, Status, ZResult};

static GLOBAL_CONFIG: OnceLock<GlobalConfigData> = OnceLock::new();
static DEFAULT_CONFIG: OnceLock<GlobalConfigData> = OnceLock::new();
static QUERY_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static OPTIMIZE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

const MIN_MEMORY_LIMIT_BYTES: u64 = 100 * 1024 * 1024;
const MIN_LOG_FILE_SIZE_MB: u32 = finch_types::config::MIN_LOG_FILE_SIZE_MB;

/// Initialize global logging/tracing (idempotent).
pub fn init_logging() {
    let cfg = global_config();

    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level = match cfg.log_level {
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error | LogLevel::Fatal => "error",
        };
        tracing_subscriber::EnvFilter::new(level)
    });

    if cfg.log_to_file {
        // file logging uses a basename that results in files
        // matching `<log_basename>.*` (glog-style). Finch writes a single stream
        // of logs, but uses a `.INFO` suffix for the active file so downstream
        // checks like `glob("<basename>.*")` behave the same.
        let basename = if cfg.log_basename.ends_with(".INFO")
            || cfg.log_basename.ends_with(".WARNING")
            || cfg.log_basename.ends_with(".ERROR")
            || cfg.log_basename.ends_with(".FATAL")
        {
            cfg.log_basename.clone()
        } else {
            format!("{}.INFO", cfg.log_basename)
        };
        let writer = RotatingFileMakeWriter::new(
            Path::new(&cfg.log_dir),
            &basename,
            cfg.log_file_size_mb,
            cfg.log_overdue_days,
        );
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    }
}

/// Initialize global config (idempotent).
///
/// If called more than once in a process, subsequent calls are a no-op and
/// return `Ok(())` without re-validating `config`.
///
/// Note: Finch uses dedicated Rayon thread pools (query + optimize) instead of
/// Rayon's global pool so config thread counts are honored regardless of other
/// crates initializing Rayon first.
pub fn initialize_global_config(config: GlobalConfigData) -> ZResult<()> {
    // repeated initialization is allowed and becomes a no-op.
    if GLOBAL_CONFIG.get().is_some() {
        return Ok(());
    }

    validate_global_config(&config)?;
    validate_thread_pool_init_order(&config)?;

    if GLOBAL_CONFIG.set(config).is_err() {
        // Another thread won the race to initialize. Treat as success.
        return Ok(());
    }
    // Best-effort: initialize tracing based on the configured log settings.
    init_logging();
    ensure_rayon_initialized();
    Ok(())
}

/// Get the active global config (initializes to defaults on first use).
pub fn global_config() -> &'static GlobalConfigData {
    GLOBAL_CONFIG
        .get()
        .unwrap_or_else(|| DEFAULT_CONFIG.get_or_init(GlobalConfigData::default))
}

pub fn validate_global_config(config: &GlobalConfigData) -> ZResult<()> {
    if config.memory_limit_bytes < MIN_MEMORY_LIMIT_BYTES {
        return Err(Status::invalid_argument(format!(
            "memory_limit_bytes must be greater than {}",
            MIN_MEMORY_LIMIT_BYTES
        )));
    }

    if let Some(limit) = finch_types::config::detect_memory_limit_bytes() {
        if config.memory_limit_bytes > limit {
            return Err(Status::invalid_argument(format!(
                "memory_limit_bytes must be less than {}",
                limit
            )));
        }
    }

    if config.log_to_file {
        if config.log_dir.trim().is_empty() {
            return Err(Status::invalid_argument(
                "log_dir cannot be empty when set to FileLogger",
            ));
        }
        if config.log_basename.trim().is_empty() {
            return Err(Status::invalid_argument(
                "log_file basename cannot be empty when set to FileLogger",
            ));
        }
        if config.log_file_size_mb < MIN_LOG_FILE_SIZE_MB {
            return Err(Status::invalid_argument(format!(
                "log file_size must be greater than {} when set to FileLogger",
                MIN_LOG_FILE_SIZE_MB
            )));
        }
        if config.log_overdue_days == 0 {
            return Err(Status::invalid_argument(
                "log_overdue_days must be greater than 0 when set to FileLogger",
            ));
        }
    }

    if config.query_thread_count == 0 {
        return Err(Status::invalid_argument("query_thread_count must be > 0"));
    }
    if config.optimize_thread_count == 0 {
        return Err(Status::invalid_argument(
            "optimize_thread_count must be > 0",
        ));
    }
    // WAL knobs are best-effort and accept 0 to disable.
    // No additional validation required beyond type bounds.
    if !(0.0..=1.0).contains(&config.invert_to_forward_scan_ratio) {
        return Err(Status::invalid_argument(
            "invert_to_forward_scan_ratio must be between 0 and 1",
        ));
    }
    if !(0.0..=1.0).contains(&config.brute_force_by_keys_ratio) {
        return Err(Status::invalid_argument(
            "brute_force_by_keys_ratio must be between 0 and 1",
        ));
    }
    Ok(())
}

/// Finch's dedicated query thread pool.
///
/// Finch avoids depending on Rayon's global thread pool because it can be
/// initialized by other crates before Finch's config is set (and then the
/// configured thread count cannot be applied). Finch uses a dedicated pool
/// for query operations.
pub fn query_pool() -> &'static rayon::ThreadPool {
    QUERY_POOL.get_or_init(|| {
        let cfg = global_config().clone();
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.query_thread_count as usize)
            .build()
            .expect("failed to build finch query thread pool")
    })
}

/// Finch's dedicated optimize thread pool.
pub fn optimize_pool() -> &'static rayon::ThreadPool {
    OPTIMIZE_POOL.get_or_init(|| {
        let cfg = global_config().clone();
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.optimize_thread_count as usize)
            .build()
            .expect("failed to build finch optimize thread pool")
    })
}

/// Ensure Finch's internal thread pools are initialized (best-effort).
pub fn ensure_rayon_initialized() {
    let _ = query_pool();
    let _ = optimize_pool();
}

fn validate_thread_pool_init_order(config: &GlobalConfigData) -> ZResult<()> {
    if let Some(pool) = QUERY_POOL.get() {
        let threads = pool.current_num_threads();
        if threads != config.query_thread_count as usize {
            return Err(Status::invalid_argument(format!(
                "query thread pool was initialized with {} threads before global config; expected {}. Call init_global_config() before any queries.",
                threads, config.query_thread_count
            )));
        }
    }
    if let Some(pool) = OPTIMIZE_POOL.get() {
        let threads = pool.current_num_threads();
        if threads != config.optimize_thread_count as usize {
            return Err(Status::invalid_argument(format!(
                "optimize thread pool was initialized with {} threads before global config; expected {}. Call init_global_config() before any optimize/DDL work.",
                threads, config.optimize_thread_count
            )));
        }
    }
    Ok(())
}

// ── Rotating file writer ─────────────────────────────────────────────────────

struct RotatingFileState {
    dir: PathBuf,
    basename: String,
    max_bytes: u64,
    overdue_days: u32,
    file: io::BufWriter<std::fs::File>,
    active_path: PathBuf,
    approx_written: u64,
}

impl RotatingFileState {
    fn open_active(
        dir: &Path,
        basename: &str,
    ) -> io::Result<(io::BufWriter<std::fs::File>, PathBuf)> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join(basename);
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok((io::BufWriter::new(f), path))
    }

    fn cleanup_overdue(&self) {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return;
        };
        let cutoff = SystemTime::now()
            .checked_sub(Duration::from_secs(
                86400u64.saturating_mul(self.overdue_days as u64),
            ))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let prefix = format!("{}.", self.basename);
        for ent in rd.flatten() {
            let Ok(ft) = ent.file_type() else {
                continue;
            };
            if !ft.is_file() {
                continue;
            }
            let name = ent.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with(&prefix) {
                continue;
            }
            let Ok(meta) = ent.metadata() else {
                continue;
            };
            let Ok(mtime) = meta.modified() else {
                continue;
            };
            if mtime < cutoff {
                let _ = std::fs::remove_file(ent.path());
            }
        }
    }

    fn maybe_rotate(&mut self) -> io::Result<()> {
        if self.approx_written < self.max_bytes {
            return Ok(());
        }

        self.file.flush()?;
        let cur_size = self
            .active_path
            .metadata()
            .map(|m| m.len())
            .unwrap_or(self.approx_written);
        if cur_size < self.max_bytes {
            self.approx_written = cur_size;
            return Ok(());
        }

        // Rotate: rename active file to basename.<unix_ts_ms>
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let rotated_name = format!("{}.{}", self.basename, ts);
        let rotated_path = self.dir.join(rotated_name);

        // Drop current file handle before rename (Windows compatibility).
        self.file.flush()?;
        let dummy_path = std::env::temp_dir().join(format!("finch_log_dummy_{}", ts));
        let dummy = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(dummy_path)?;
        let old_file = std::mem::replace(&mut self.file, io::BufWriter::new(dummy));
        drop(old_file);

        let _ = std::fs::rename(&self.active_path, rotated_path);

        let (new_file, _) = Self::open_active(&self.dir, &self.basename)?;
        self.file = new_file;

        self.approx_written = 0;
        self.cleanup_overdue();
        Ok(())
    }
}

#[derive(Clone)]
struct RotatingFileMakeWriter {
    state: std::sync::Arc<std::sync::Mutex<RotatingFileState>>,
}

impl RotatingFileMakeWriter {
    fn new(dir: &Path, basename: &str, max_mb: u32, overdue_days: u32) -> Self {
        let max_bytes = (max_mb as u64).saturating_mul(1024 * 1024);
        let (file, active_path) =
            RotatingFileState::open_active(dir, basename).unwrap_or_else(|_| {
                // As a last resort, fall back to stderr by writing to a temp file.
                let tmp = std::env::temp_dir().join("finch.log");
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&tmp)
                    .expect("failed to open fallback finch.log");
                (io::BufWriter::new(f), tmp)
            });

        let state = RotatingFileState {
            dir: dir.to_path_buf(),
            basename: basename.to_string(),
            max_bytes: max_bytes.max(1),
            overdue_days: overdue_days.max(1),
            file,
            active_path,
            approx_written: 0,
        };
        state.cleanup_overdue();
        RotatingFileMakeWriter {
            state: std::sync::Arc::new(std::sync::Mutex::new(state)),
        }
    }
}

struct RotatingFileWriterGuard {
    state: std::sync::Arc<std::sync::Mutex<RotatingFileState>>,
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for RotatingFileMakeWriter {
    type Writer = RotatingFileWriterGuard;

    fn make_writer(&'a self) -> Self::Writer {
        RotatingFileWriterGuard {
            state: self.state.clone(),
        }
    }
}

impl Write for RotatingFileWriterGuard {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| io::Error::other("log writer poisoned"))?;
        let n = st.file.write(buf)?;
        st.approx_written = st.approx_written.saturating_add(n as u64);
        let _ = st.maybe_rotate();
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut st = self
            .state
            .lock()
            .map_err(|_| io::Error::other("log writer poisoned"))?;
        st.file.flush()?;
        let _ = st.maybe_rotate();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_global_config_is_idempotent_and_skips_revalidation() {
        // First init: best effort (may already be initialized by another test).
        let _ = initialize_global_config(GlobalConfigData::default());

        // once initialized, subsequent init calls are a no-op and
        // should not fail validation even if provided an invalid config.
        let invalid = GlobalConfigData {
            query_thread_count: 0,
            ..GlobalConfigData::default()
        };
        initialize_global_config(invalid).expect("second init should be a no-op");
    }
}
