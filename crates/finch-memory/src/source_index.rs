//! Persistent source-level lexical index with a source-to-span registry.
//!
//! Retrieval over large imported corpora needs two capabilities the span
//! vector index alone cannot provide: exact lexical matching of entities,
//! identifiers, and phrases at whole-source granularity, and an indexed
//! lookup from a source to its ordered spans so retrieved chunks can be
//! hydrated into complete parent evidence without scanning the store.
//!
//! The index stores no source text and no vectors: postings hold term
//! statistics keyed by the term itself, and the registry holds span
//! identifiers. Span text is hydrated from the memory store at query time.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use finch_types::{Status, ZResult};
use serde::{Deserialize, Serialize};

use crate::retrieval::lexical_terms;

pub const SOURCE_LEXICAL_INDEX_FORMAT: &str = "finch-source-lexical-index-v5";

const MANIFEST_FILE: &str = "manifest.json";
const SOURCES_FILE: &str = "sources.jsonl";
const TERMS_FILE: &str = "terms.bin";
const POSTINGS_FILE: &str = "postings.bin";

// offset u64, posting count u32, term length u32; the term bytes follow.
const TERM_RECORD_HEADER_BYTES: usize = 8 + 4 + 4;
const POSTING_RECORD_BYTES: usize = 4 + 2;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

#[derive(Debug, Serialize, Deserialize)]
struct SourceLexicalManifest {
    format: String,
    source_count: usize,
    span_count: usize,
    total_doc_len: u64,
    term_count: usize,
    posting_count: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceRow {
    id: String,
    #[serde(rename = "dir")]
    directory: String,
    len: u32,
    spans: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SourceLexicalHit {
    pub source_id: String,
    pub score: f32,
}

struct SourceEntry {
    id: String,
    directory: String,
    doc_len: u32,
    span_ids: Vec<String>,
}

#[derive(Clone, Copy)]
struct TermEntry {
    offset: u64,
    count: u32,
}

/// Builder that accumulates one entry per source and writes the on-disk index.
#[derive(Default)]
pub struct SourceLexicalIndexBuilder {
    sources: Vec<SourceRow>,
    postings: HashMap<String, Vec<(u32, u16)>>,
    total_doc_len: u64,
    span_count: usize,
}

impl SourceLexicalIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one source document: its title, directory, body, and ordered span ids.
    pub fn add_source(
        &mut self,
        source_id: &str,
        title: &str,
        directory: &str,
        body: &str,
        span_ids: Vec<String>,
    ) -> ZResult<()> {
        let ordinal = u32::try_from(self.sources.len()).map_err(|_| {
            Status::invalid_argument("source lexical index supports at most 2^32 sources")
        })?;
        let mut term_freq = HashMap::<String, u32>::new();
        let mut doc_len = 0u32;
        for term in lexical_terms(title).into_iter().chain(lexical_terms(body)) {
            doc_len = doc_len.saturating_add(1);
            *term_freq.entry(term).or_insert(0) += 1;
        }
        for (term, tf) in term_freq {
            self.postings
                .entry(term)
                .or_default()
                .push((ordinal, u16::try_from(tf).unwrap_or(u16::MAX)));
        }
        self.total_doc_len += u64::from(doc_len);
        self.span_count += span_ids.len();
        self.sources.push(SourceRow {
            id: source_id.to_string(),
            directory: directory.to_string(),
            len: doc_len,
            spans: span_ids,
        });
        Ok(())
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Write the index directory. Any existing directory content is replaced.
    pub fn write(self, dir: &Path) -> ZResult<()> {
        if self.sources.is_empty() {
            return Err(Status::invalid_argument(
                "source lexical index requires at least one source",
            ));
        }
        let staging = staging_dir(dir)?;
        let _ = fs::remove_dir_all(&staging);
        fs::create_dir_all(&staging).map_err(Status::from)?;

        let mut sources_writer =
            BufWriter::new(File::create(staging.join(SOURCES_FILE)).map_err(Status::from)?);
        for row in &self.sources {
            serde_json::to_writer(&mut sources_writer, row)
                .map_err(|error| Status::internal(error.to_string()))?;
            sources_writer.write_all(b"\n").map_err(Status::from)?;
        }
        sources_writer.flush().map_err(Status::from)?;

        let mut terms = self.postings.into_iter().collect::<Vec<_>>();
        terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut terms_writer =
            BufWriter::new(File::create(staging.join(TERMS_FILE)).map_err(Status::from)?);
        let mut postings_writer =
            BufWriter::new(File::create(staging.join(POSTINGS_FILE)).map_err(Status::from)?);
        let mut offset = 0u64;
        let mut posting_count = 0u64;
        let term_count = terms.len();
        for (term, mut postings) in terms {
            postings.sort_by_key(|(ordinal, _)| *ordinal);
            let term_len = u32::try_from(term.len())
                .map_err(|_| Status::invalid_argument("source lexical term is too long"))?;
            terms_writer
                .write_all(&offset.to_le_bytes())
                .map_err(Status::from)?;
            terms_writer
                .write_all(&(postings.len() as u32).to_le_bytes())
                .map_err(Status::from)?;
            terms_writer
                .write_all(&term_len.to_le_bytes())
                .map_err(Status::from)?;
            terms_writer
                .write_all(term.as_bytes())
                .map_err(Status::from)?;
            for (ordinal, tf) in &postings {
                postings_writer
                    .write_all(&ordinal.to_le_bytes())
                    .map_err(Status::from)?;
                postings_writer
                    .write_all(&tf.to_le_bytes())
                    .map_err(Status::from)?;
            }
            posting_count += postings.len() as u64;
            offset += (postings.len() * POSTING_RECORD_BYTES) as u64;
        }
        terms_writer.flush().map_err(Status::from)?;
        postings_writer.flush().map_err(Status::from)?;

        let manifest = SourceLexicalManifest {
            format: SOURCE_LEXICAL_INDEX_FORMAT.to_string(),
            source_count: self.sources.len(),
            span_count: self.span_count,
            total_doc_len: self.total_doc_len,
            term_count,
            posting_count,
        };
        let manifest_file = File::create(staging.join(MANIFEST_FILE)).map_err(Status::from)?;
        serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)
            .map_err(|error| Status::internal(error.to_string()))?;

        let _ = fs::remove_dir_all(dir);
        fs::rename(&staging, dir).map_err(Status::from)
    }
}

fn staging_dir(dir: &Path) -> ZResult<PathBuf> {
    let name = dir.file_name().ok_or_else(|| {
        Status::invalid_argument("source lexical index path has no directory name")
    })?;
    let mut staged = name.to_os_string();
    staged.push(".staging");
    Ok(dir.with_file_name(staged))
}

/// Read handle over a written source lexical index.
pub struct SourceLexicalIndex {
    sources: Vec<SourceEntry>,
    ordinal_by_source: HashMap<String, u32>,
    ordinals_by_directory: HashMap<String, Vec<u32>>,
    terms: HashMap<String, TermEntry>,
    postings: Mutex<File>,
    total_doc_len: u64,
}

impl SourceLexicalIndex {
    pub fn open(dir: &Path) -> ZResult<Self> {
        let manifest: SourceLexicalManifest = serde_json::from_reader(BufReader::new(
            File::open(dir.join(MANIFEST_FILE)).map_err(Status::from)?,
        ))
        .map_err(|error| Status::invalid_argument(error.to_string()))?;
        if manifest.format != SOURCE_LEXICAL_INDEX_FORMAT {
            return Err(Status::invalid_argument(format!(
                "unsupported source lexical index format {}",
                manifest.format
            )));
        }

        let mut sources = Vec::with_capacity(manifest.source_count);
        let mut ordinal_by_source = HashMap::with_capacity(manifest.source_count);
        let mut ordinals_by_directory = HashMap::<String, Vec<u32>>::new();
        let reader = BufReader::new(File::open(dir.join(SOURCES_FILE)).map_err(Status::from)?);
        for line in reader.lines() {
            let line = line.map_err(Status::from)?;
            if line.trim().is_empty() {
                continue;
            }
            let row: SourceRow = serde_json::from_str(&line)
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            let ordinal = u32::try_from(sources.len())
                .map_err(|_| Status::invalid_argument("source lexical index is too large"))?;
            ordinal_by_source.insert(row.id.clone(), ordinal);
            ordinals_by_directory
                .entry(row.directory.clone())
                .or_default()
                .push(ordinal);
            sources.push(SourceEntry {
                id: row.id,
                directory: row.directory,
                doc_len: row.len,
                span_ids: row.spans,
            });
        }
        if sources.len() != manifest.source_count {
            return Err(Status::invalid_argument(format!(
                "source lexical index has {} sources, manifest expects {}",
                sources.len(),
                manifest.source_count
            )));
        }

        let mut terms = HashMap::with_capacity(manifest.term_count);
        let mut terms_reader =
            BufReader::new(File::open(dir.join(TERMS_FILE)).map_err(Status::from)?);
        let mut header = [0u8; TERM_RECORD_HEADER_BYTES];
        loop {
            match terms_reader.read_exact(&mut header) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(Status::from(error)),
            }
            let offset = u64::from_le_bytes(header[0..8].try_into().unwrap()); // 8-byte range of a fixed-size header
            let count = u32::from_le_bytes(header[8..12].try_into().unwrap()); // 4-byte range of a fixed-size header
            let term_len = u32::from_le_bytes(header[12..16].try_into().unwrap()); // 4-byte range of a fixed-size header
            let mut term = vec![0u8; term_len as usize];
            terms_reader.read_exact(&mut term).map_err(Status::from)?;
            let term = String::from_utf8(term)
                .map_err(|error| Status::invalid_argument(error.to_string()))?;
            terms.insert(term, TermEntry { offset, count });
        }

        let postings = File::open(dir.join(POSTINGS_FILE)).map_err(Status::from)?;
        Ok(Self {
            sources,
            ordinal_by_source,
            ordinals_by_directory,
            terms,
            postings: Mutex::new(postings),
            total_doc_len: manifest.total_doc_len,
        })
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Ordered span ids for one source, if the source is indexed.
    pub fn span_ids_for_source(&self, source_id: &str) -> Option<&[String]> {
        let ordinal = *self.ordinal_by_source.get(source_id)?;
        Some(&self.sources[ordinal as usize].span_ids)
    }

    /// Directory recorded for one source, if the source is indexed.
    pub fn directory_of(&self, source_id: &str) -> Option<&str> {
        let ordinal = *self.ordinal_by_source.get(source_id)?;
        Some(&self.sources[ordinal as usize].directory)
    }

    /// Source ids in a directory, in index insertion order.
    pub fn sources_in_directory(&self, directory: &str) -> Vec<&str> {
        self.ordinals_by_directory
            .get(directory)
            .into_iter()
            .flatten()
            .map(|ordinal| self.sources[*ordinal as usize].id.as_str())
            .collect()
    }

    /// BM25 search over whole sources. Terms present in more than
    /// `max_df_ratio` of all sources are skipped as non-discriminative.
    pub fn search(
        &self,
        query_text: &str,
        k: usize,
        max_df_ratio: f32,
    ) -> ZResult<Vec<SourceLexicalHit>> {
        if k == 0 || self.sources.is_empty() {
            return Ok(Vec::new());
        }
        let mut query_terms = lexical_terms(query_text);
        query_terms.sort_unstable();
        query_terms.dedup();
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let source_count = self.sources.len() as f32;
        let avg_len = (self.total_doc_len as f32 / source_count).max(1.0);
        let max_df = (max_df_ratio.clamp(0.0, 1.0) * source_count).max(1.0);
        let mut scores = HashMap::<u32, f32>::new();
        for term in query_terms {
            let Some(entry) = self.terms.get(&term) else {
                continue;
            };
            if entry.count as f32 > max_df {
                continue;
            }
            let postings = self.read_postings(entry)?;
            let df = postings.len() as f32;
            let idf = ((source_count - df + 0.5) / (df + 0.5) + 1.0).ln();
            for (ordinal, tf) in postings {
                let Some(source) = self.sources.get(ordinal as usize) else {
                    continue;
                };
                let tf = f32::from(tf);
                let doc_len = source.doc_len as f32;
                let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / avg_len);
                *scores.entry(ordinal).or_insert(0.0) += idf * (tf * (BM25_K1 + 1.0)) / denom;
            }
        }

        let mut hits = scores.into_iter().collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        hits.truncate(k);
        Ok(hits
            .into_iter()
            .map(|(ordinal, score)| SourceLexicalHit {
                source_id: self.sources[ordinal as usize].id.clone(),
                score,
            })
            .collect())
    }

    fn read_postings(&self, entry: &TermEntry) -> ZResult<Vec<(u32, u16)>> {
        let mut bytes = vec![0u8; entry.count as usize * POSTING_RECORD_BYTES];
        {
            let mut file = self
                .postings
                .lock()
                .map_err(|_| Status::internal("source lexical postings lock is poisoned"))?;
            file.seek(SeekFrom::Start(entry.offset))
                .map_err(Status::from)?;
            file.read_exact(&mut bytes).map_err(Status::from)?;
        }
        Ok(bytes
            .chunks_exact(POSTING_RECORD_BYTES)
            .map(|record| {
                (
                    u32::from_le_bytes(record[0..4].try_into().unwrap()), // chunks_exact yields full records
                    u16::from_le_bytes(record[4..6].try_into().unwrap()), // chunks_exact yields full records
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index(dir: &Path) -> SourceLexicalIndex {
        let mut builder = SourceLexicalIndexBuilder::new();
        builder
            .add_source(
                "doc_alpha",
                "",
                "projects/kestrel",
                "Kestrel deployment runbook. The kestrel rollout freezes traffic during failover.",
                vec!["span_a0".to_string(), "span_a1".to_string()],
            )
            .unwrap();
        builder
            .add_source(
                "doc_beta",
                "",
                "departments/finance",
                "Quarterly finance summary. Revenue grew while the deployment budget stayed flat.",
                vec!["span_b0".to_string()],
            )
            .unwrap();
        builder
            .add_source(
                "doc_gamma",
                "",
                "teams/people",
                "Weekly team notes. The team discussed hiring, onboarding, and the offsite.",
                vec!["span_c0".to_string()],
            )
            .unwrap();
        builder.write(dir).unwrap();
        SourceLexicalIndex::open(dir).unwrap()
    }

    #[test]
    fn search_ranks_exact_term_matches_first() {
        let dir = std::env::temp_dir().join(format!("finch-source-index-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let index = build_index(&dir);
        assert_eq!(index.source_count(), 3);

        let hits = index.search("kestrel failover", 10, 1.0).unwrap();
        assert_eq!(hits[0].source_id, "doc_alpha");

        let hits = index.search("finance revenue budget", 10, 1.0).unwrap();
        assert_eq!(hits[0].source_id, "doc_beta");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn title_and_body_terms_have_equal_weight() {
        let dir =
            std::env::temp_dir().join(format!("finch-source-index-title-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut builder = SourceLexicalIndexBuilder::new();
        builder
            .add_source(
                "title_match",
                "Kestrel",
                "",
                "Operations brief",
                vec!["title_span".to_string()],
            )
            .unwrap();
        builder
            .add_source(
                "body_match",
                "",
                "",
                "Kestrel operations brief",
                vec!["body_span".to_string()],
            )
            .unwrap();
        builder.write(&dir).unwrap();
        let index = SourceLexicalIndex::open(&dir).unwrap();

        let hits = index.search("kestrel", 10, 1.0).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].score, hits[1].score);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn registry_returns_ordered_span_ids() {
        let dir =
            std::env::temp_dir().join(format!("finch-source-index-reg-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let index = build_index(&dir);
        assert_eq!(
            index.span_ids_for_source("doc_alpha").unwrap(),
            &["span_a0".to_string(), "span_a1".to_string()]
        );
        assert!(index.span_ids_for_source("doc_missing").is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_metadata_round_trips_and_lists_sources_in_insertion_order() {
        let dir =
            std::env::temp_dir().join(format!("finch-source-index-dir-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut builder = SourceLexicalIndexBuilder::new();
        for (source_id, directory) in [
            ("design_brief", "projects/orbit"),
            ("release_notes", "projects/orbit"),
            ("team_guide", "teams/people"),
        ] {
            builder
                .add_source(
                    source_id,
                    "",
                    directory,
                    "product knowledge",
                    vec![format!("{source_id}_span")],
                )
                .unwrap();
        }
        builder.write(&dir).unwrap();
        let index = SourceLexicalIndex::open(&dir).unwrap();

        assert_eq!(index.directory_of("design_brief"), Some("projects/orbit"));
        assert_eq!(index.directory_of("missing"), None);
        assert_eq!(
            index.sources_in_directory("projects/orbit"),
            vec!["design_brief", "release_notes"]
        );
        assert!(index.sources_in_directory("projects/missing").is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn high_document_frequency_terms_are_skipped() {
        let dir =
            std::env::temp_dir().join(format!("finch-source-index-df-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let index = build_index(&dir);
        // "the" appears in every source; with a strict ratio it contributes nothing.
        let hits = index.search("the", 10, 0.5).unwrap();
        assert!(hits.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_terms_keep_separate_posting_lists() {
        let dir =
            std::env::temp_dir().join(format!("finch-source-index-terms-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut builder = SourceLexicalIndexBuilder::new();
        builder
            .add_source("doc_x", "", "", "kestrel kestrel", Vec::new())
            .unwrap();
        builder
            .add_source("doc_y", "", "", "osprey", Vec::new())
            .unwrap();
        builder.write(&dir).unwrap();
        let index = SourceLexicalIndex::open(&dir).unwrap();

        // Postings are keyed by the term bytes, not a hash of them, so two terms can never merge.
        assert_eq!(index.terms.len(), 2);
        for (term, ordinal, tf, source_id) in
            [("kestrel", 0, 2, "doc_x"), ("osprey", 1, 1, "doc_y")]
        {
            let postings = index.read_postings(&index.terms[term]).unwrap();
            assert_eq!(postings, vec![(ordinal, tf)]);
            let hits = index.search(term, 10, 1.0).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].source_id, source_id);
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
