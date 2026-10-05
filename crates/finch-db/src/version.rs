//! Manifest-based version manager for collection metadata

use finch_proto::manifest::{CollectionSchema as MSchema, Manifest, SegmentMeta};
use finch_types::{CollectionSchema, Status, ZResult};
use parking_lot::{Mutex, RwLock};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MANIFEST_PREFIX: &str = "manifest";
const MANIFEST_TMP: &str = "manifest.tmp";
/// On-disk format of the collection's indexes; open rejects any other value.
pub const FORMAT_VERSION: u32 = 1;

/// Snapshot of collection state
#[derive(Debug, Clone)]
pub struct Version {
    pub schema: CollectionSchema,
    pub persisted_segments: Vec<PersistedSegmentVersion>,
    pub writing_segment_id: Option<u32>,
    pub id_map_suffix: u32,
    pub delete_suffix: u32,
    pub next_segment_id: u32,
    pub enable_mmap: bool,
    /// Index format the persisted segments were written in; see `FORMAT_VERSION`.
    pub format_version: u32,
}

#[derive(Debug, Clone)]
pub struct PersistedSegmentVersion {
    pub segment_id: u32,
    pub min_doc_id: u64,
    pub max_doc_id: u64,
    pub doc_count: u64,
    pub indexed_vector_fields: Vec<String>,
    /// Directory generation per indexed vector field; absent means 0.
    pub vector_index_generations: BTreeMap<String, u32>,
}

impl PersistedSegmentVersion {
    pub fn vector_index_generation(&self, field: &str) -> u32 {
        self.vector_index_generations
            .get(field)
            .copied()
            .unwrap_or(0)
    }
}

impl Version {
    pub fn new(schema: CollectionSchema, enable_mmap: bool) -> Self {
        Version {
            schema,
            persisted_segments: Vec::new(),
            writing_segment_id: Some(0),
            id_map_suffix: 0,
            delete_suffix: 0,
            next_segment_id: 1,
            enable_mmap,
            format_version: FORMAT_VERSION,
        }
    }
}

pub struct VersionManager {
    path: PathBuf,
    current: RwLock<Arc<Version>>,
    next_manifest_id: Mutex<u64>,
}

impl VersionManager {
    /// Create a new collection manifest
    pub fn create(path: &Path, schema: CollectionSchema, enable_mmap: bool) -> ZResult<Self> {
        let version = Version::new(schema, enable_mmap);
        let manager = VersionManager {
            path: path.to_path_buf(),
            current: RwLock::new(Arc::new(version.clone())),
            next_manifest_id: Mutex::new(0),
        };
        manager.flush(&version)?;
        Ok(manager)
    }

    /// Load an existing collection manifest
    pub fn load(path: &Path) -> ZResult<Self> {
        let (version, next_manifest_id) = load_latest_valid_version(path)?;
        Ok(VersionManager {
            path: path.to_path_buf(),
            current: RwLock::new(Arc::new(version)),
            next_manifest_id: Mutex::new(next_manifest_id),
        })
    }

    pub fn current(&self) -> Arc<Version> {
        self.current.read().clone()
    }

    /// Atomically update the manifest: write to .tmp then rename
    pub fn flush(&self, version: &Version) -> ZResult<()> {
        let manifest = encode_manifest(version)?;
        let buf = manifest
            .encode()
            .map_err(|e| Status::io_error(format!("manifest encode error: {}", e)))?;

        let mut next_id_guard = self.next_manifest_id.lock();
        let next_id = *next_id_guard;

        let tmp_path = self.path.join(MANIFEST_TMP);
        let manifest_path = manifest_path_for_id(&self.path, next_id);

        {
            let mut file =
                fs::File::create(&tmp_path).map_err(|e| Status::io_error(e.to_string()))?;
            file.write_all(&buf)
                .and_then(|()| file.sync_all())
                .map_err(|e| Status::io_error(format!("manifest write failed: {}", e)))?;
        }

        fs::rename(&tmp_path, &manifest_path).map_err(|e| Status::io_error(e.to_string()))?;
        // The new manifest must be durable before the previous one goes away or callers act on it.
        sync_dir(&self.path)?;

        *next_id_guard = next_id + 1;
        *self.current.write() = Arc::new(version.clone());

        // Keep only the newest manifest file; load picks the newest, so a leftover older one is harmless.
        if next_id > 0 {
            let _ = fs::remove_file(manifest_path_for_id(&self.path, next_id - 1));
        }
        Ok(())
    }

    pub fn update(&self, version: Version) -> ZResult<()> {
        self.flush(&version)?;
        *self.current.write() = Arc::new(version);
        Ok(())
    }
}

fn sync_dir(path: &Path) -> ZResult<()> {
    let dir = match fs::File::open(path) {
        Ok(dir) => dir,
        Err(e) => {
            if cfg!(windows) && e.kind() == std::io::ErrorKind::PermissionDenied {
                return Ok(());
            }
            return Err(Status::io_error(e.to_string()));
        }
    };
    if let Err(e) = dir.sync_all() {
        if cfg!(windows) && e.kind() == std::io::ErrorKind::PermissionDenied {
            return Ok(());
        }
        return Err(Status::io_error(e.to_string()));
    }
    Ok(())
}

fn encode_manifest(v: &Version) -> ZResult<Manifest> {
    let schema: MSchema = v.schema.clone().into();

    let persisted: Vec<SegmentMeta> = v
        .persisted_segments
        .iter()
        .map(|seg| SegmentMeta {
            segment_id: seg.segment_id,
            persisted_blocks: Vec::new(),
            writing_forward_block: None,
            indexed_vector_fields: seg.indexed_vector_fields.clone(),
            vector_index_generations: seg.vector_index_generations.clone(),
            min_doc_id: seg.min_doc_id,
            max_doc_id: seg.max_doc_id,
            doc_count: seg.doc_count,
        })
        .collect();

    Ok(Manifest {
        version: v.format_version,
        schema: Some(schema),
        enable_mmap: v.enable_mmap,
        persisted_segment_metas: persisted,
        writing_segment_meta: v.writing_segment_id.map(|id| SegmentMeta {
            segment_id: id,
            ..Default::default()
        }),
        id_map_path_suffix: v.id_map_suffix,
        delete_snapshot_path_suffix: v.delete_suffix,
        next_segment_id: v.next_segment_id,
    })
}

fn decode_manifest(m: Manifest) -> ZResult<Version> {
    let schema = m
        .schema
        .ok_or_else(|| Status::internal("manifest missing schema"))?;
    let schema: CollectionSchema = schema.try_into()?;

    let persisted = m
        .persisted_segment_metas
        .into_iter()
        .map(|sm| PersistedSegmentVersion {
            segment_id: sm.segment_id,
            min_doc_id: sm.min_doc_id,
            max_doc_id: sm.max_doc_id,
            doc_count: sm.doc_count,
            indexed_vector_fields: sm.indexed_vector_fields,
            vector_index_generations: sm.vector_index_generations,
        })
        .collect();

    let writing_segment_id = m.writing_segment_meta.map(|sm| sm.segment_id);

    Ok(Version {
        schema,
        persisted_segments: persisted,
        writing_segment_id,
        id_map_suffix: m.id_map_path_suffix,
        delete_suffix: m.delete_snapshot_path_suffix,
        next_segment_id: m.next_segment_id,
        enable_mmap: m.enable_mmap,
        format_version: m.version,
    })
}

fn manifest_path_for_id(path: &Path, id: u64) -> PathBuf {
    path.join(format!("{MANIFEST_PREFIX}.{id}"))
}

fn load_latest_valid_version(path: &Path) -> ZResult<(Version, u64)> {
    let mut ids: Vec<u64> = Vec::new();

    let entries = fs::read_dir(path).map_err(|e| Status::io_error(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| Status::io_error(e.to_string()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(&(MANIFEST_PREFIX.to_string() + ".")) {
            continue;
        }
        let suffix = &name[MANIFEST_PREFIX.len() + 1..];
        if let Ok(id) = suffix.parse::<u64>() {
            ids.push(id);
        }
    }

    if let Some(&max_id) = ids.iter().max() {
        let next_id = max_id + 1;

        // load only the latest (max-id) manifest.
        // If it is corrupt/truncated, open fails even if older manifest files exist.
        let manifest_path = manifest_path_for_id(path, max_id);
        let data = fs::read(&manifest_path).map_err(|e| Status::io_error(e.to_string()))?;
        let manifest = Manifest::decode(&data)
            .map_err(|e| Status::io_error(format!("manifest decode error: {}", e)))?;
        let version = decode_manifest(manifest)?;
        return Ok((version, next_id));
    }

    Err(Status::not_found("no manifest file found"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough for tests")
            .as_nanos();
        path.push(format!("finch-{prefix}-{}-{ts}", std::process::id()));
        fs::create_dir_all(&path).expect("test temp dir should be created");
        path
    }

    fn test_schema() -> CollectionSchema {
        CollectionSchema::new("test")
    }

    #[test]
    fn rotates_manifest_files_on_update() {
        let dir = test_dir("rotate");

        let manager =
            VersionManager::create(&dir, test_schema(), false).expect("create should work");
        assert!(dir.join("manifest.0").exists());

        let mut version = (*manager.current()).clone();
        version.next_segment_id += 1;
        manager.update(version).expect("update should work");

        assert!(!dir.join("manifest.0").exists());
        assert!(dir.join("manifest.1").exists());

        drop(manager);
        fs::remove_dir_all(&dir).expect("test temp dir should be removed");
    }

    #[test]
    fn load_uses_latest_manifest_file() {
        let dir = test_dir("latest");

        let manager =
            VersionManager::create(&dir, test_schema(), false).expect("create should work");
        let mut v1 = (*manager.current()).clone();
        v1.next_segment_id += 1;
        manager.update(v1).expect("first update should work");
        assert!(dir.join("manifest.1").exists());
        drop(manager);

        let loaded = VersionManager::load(&dir).expect("load should pick latest manifest");
        let mut v2 = (*loaded.current()).clone();
        v2.next_segment_id += 1;
        loaded.update(v2).expect("second update should work");

        assert!(!dir.join("manifest.1").exists());
        assert!(dir.join("manifest.2").exists());

        drop(loaded);
        fs::remove_dir_all(&dir).expect("test temp dir should be removed");
    }

    #[test]
    fn load_fails_on_corrupt_latest_manifest_even_if_previous_exists() {
        let dir = test_dir("corrupt_latest");

        let manager =
            VersionManager::create(&dir, test_schema(), false).expect("create should work");
        let v0 = (*manager.current()).clone();

        let mut v1 = v0.clone();
        v1.next_segment_id += 1;
        manager.update(v1).expect("update should work");
        assert!(dir.join("manifest.1").exists());

        // Simulate crash window: previous manifest still exists on disk.
        let m0 = encode_manifest(&v0).expect("encode v0");
        let buf0 = m0.encode().expect("encode bytes");
        fs::write(dir.join("manifest.0"), buf0).expect("write manifest.0");

        // Corrupt latest manifest.
        fs::write(dir.join("manifest.1"), b"\x01").expect("corrupt manifest.1");

        drop(manager);

        assert!(VersionManager::load(&dir).is_err());
        fs::remove_dir_all(&dir).expect("test temp dir should be removed");
    }

    #[test]
    fn load_survives_crash_between_rename_and_cleanup() {
        let dir = test_dir("crash_before_cleanup");

        let manager =
            VersionManager::create(&dir, test_schema(), false).expect("create should work");
        let v0 = (*manager.current()).clone();
        let mut v1 = v0.clone();
        v1.next_segment_id += 1;
        manager.update(v1.clone()).expect("update should work");
        drop(manager);

        // The crash left the previous manifest and a half-written temp file behind.
        let buf0 = encode_manifest(&v0)
            .expect("encode v0")
            .encode()
            .expect("encode bytes");
        fs::write(dir.join("manifest.0"), buf0).expect("write manifest.0");
        fs::write(dir.join(MANIFEST_TMP), b"\x01").expect("write manifest.tmp");

        let loaded = VersionManager::load(&dir).expect("load should pick the renamed manifest");
        assert_eq!(loaded.current().next_segment_id, v1.next_segment_id);

        let mut v2 = v1;
        v2.next_segment_id += 1;
        loaded
            .update(v2)
            .expect("update after recovery should work");
        assert!(dir.join("manifest.2").exists());
        assert!(!dir.join("manifest.1").exists());
        assert!(!dir.join(MANIFEST_TMP).exists());

        drop(loaded);
        fs::remove_dir_all(&dir).expect("test temp dir should be removed");
    }
}
