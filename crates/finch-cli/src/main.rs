//! `finch`: commands that let an agent search and read a store of documents from its shell.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::LazyLock;

use finch_memory::{
    ingest_episode, lexical_terms, ActorKind, ChunkOptions, EpisodeInput, IndexedSource,
    MemoryScope, MemoryStore, PostgresStore, SourceKind, SourceLexicalIndex,
    SourceLexicalIndexBuilder, SourceSearch, SourceSpanSelection, Visibility,
};
use finch_types::{HnswIndexParams, MetricType, Status, StatusCode, ZResult};
use regex::{Regex, RegexBuilder};

const HELP: &str = "\
finch: search and read a store of documents.

  finch add DIR [--index-only]
      Adds each text file under DIR as one document and writes the keyword index.
      --index-only writes only the index, for documents that the store holds.
  finch context \"QUESTION\" [--sources N] [--pool N] [--chars N]
      Prints the best passages of the documents that match QUESTION. Read it first.
      With a rerank command, it reranks the best N documents (default 50) first.
  finch find \"QUERY\" [--source DIR] [--after YYYY-MM-DD] [--before YYYY-MM-DD]
             [--sort date] [--limit N] [--offset N] [--exact]
      Lists the documents that match QUERY: id, date, folder, title, best line.
      It ranks by keywords and by meaning. --exact uses keywords only.
  finch show ID [--grep TERM] [--lines A-B]
      Prints one document. --grep prints only the paragraphs that contain TERM.
  finch similar ID [--limit N]
      Lists documents with the same rare words as ID: other versions, events or customers.
  finch grep REGEX [--source DIR] [--limit N]
      Finds exact text in the documents that best match the words of REGEX.
  finch check ID \"TEXT\"
      Prints the lines of ID that contain TEXT, or the closest lines.

Settings:
  FINCH_POSTGRES_URL   Postgres server with pgvector. This setting is necessary.
  FINCH_STORE          Store name, a Postgres schema. The default is finch_documents.
  FINCH_INDEX          Directory of the keyword index. The default is ./finch-index.
  FINCH_SPACE          Memory space of the documents. The default is memory.
  FINCH_EMBED_COMMAND  Shell command that embeds text. It reads a JSON array of strings
                       and writes a JSON array of vectors. Without it, find uses keywords only.
  FINCH_RERANK_COMMAND Shell command that scores documents for context. It reads
                       {\"query\": Q, \"documents\": [...]} and writes a JSON array of scores.
";

const CONTEXT_SOURCES: usize = 10;
const CONTEXT_POOL: usize = 50;
const CONTEXT_SPANS_PER_SOURCE: usize = 3;
const CONTEXT_CHARS: usize = 24_000;
const FIND_LIMIT: usize = 20;
const FILTERED_CANDIDATES: usize = 5_000;
const SHOW_LINES: usize = 300;
const SIMILAR_TERMS: usize = 24;
const GREP_CANDIDATES: usize = 2_000;
const GREP_LIMIT: usize = 30;
const TEXT_BATCH: usize = 100;
const EMBED_BATCH: usize = 64;
const STORE_BATCH_DOCUMENTS: usize = 256;
const TITLE_CHARS: usize = 120;

static DATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\D)((?:19|20)\d{2}-\d{2}-\d{2})(?:\D|$)").unwrap()); // a fixed valid pattern
static HEADER_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Za-z][A-Za-z_-]{0,30}):\s?(.*)$").unwrap()); // a fixed valid pattern

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("finch: {}", error.message());
            std::process::exit(1);
        }
    }
}

fn run(args: &[String]) -> ZResult<String> {
    let Some((command, rest)) = args.split_first() else {
        return Ok(HELP.to_string());
    };
    let options = Options::parse(rest)?;
    let settings = Settings::from_env()?;
    match command.as_str() {
        "add" => add(&settings, &options),
        "context" => context(&Memory::open(settings)?, &options),
        "find" => find(&Memory::open(settings)?, &options),
        "show" => show(&Memory::open(settings)?, &options),
        "similar" => similar(&Memory::open(settings)?, &options),
        "grep" => grep(&Memory::open(settings)?, &options),
        "check" => check(&Memory::open(settings)?, &options),
        _ => Ok(HELP.to_string()),
    }
}

/// Positional words and `--name value` flags; `--exact` and `--index-only` take no value.
struct Options {
    words: Vec<String>,
    flags: BTreeMap<String, String>,
}

impl Options {
    fn parse(args: &[String]) -> ZResult<Self> {
        let mut words = Vec::new();
        let mut flags = BTreeMap::new();
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            let Some(name) = arg.strip_prefix("--") else {
                words.push(arg.clone());
                continue;
            };
            let value = match name {
                "exact" | "index-only" => String::new(),
                _ => args
                    .next()
                    .cloned()
                    .ok_or_else(|| Status::invalid_argument(format!("--{name} needs a value")))?,
            };
            flags.insert(name.to_string(), value);
        }
        Ok(Self { words, flags })
    }

    fn text(&self) -> String {
        self.words.join(" ")
    }

    fn flag(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    fn number(&self, name: &str, default: usize) -> ZResult<usize> {
        self.flag(name).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| Status::invalid_argument(format!("--{name} needs a number")))
        })
    }

    fn date(&self, name: &str) -> ZResult<Option<i64>> {
        self.flag(name)
            .map(|value| {
                parse_date(value).ok_or_else(|| {
                    Status::invalid_argument(format!("--{name} needs a date as YYYY-MM-DD"))
                })
            })
            .transpose()
    }
}

struct Settings {
    store: PostgresStore,
    index_dir: PathBuf,
    scope: MemoryScope,
    embed_command: Option<String>,
    rerank_command: Option<String>,
}

impl Settings {
    fn from_env() -> ZResult<Self> {
        let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
        let url = var("FINCH_POSTGRES_URL")
            .ok_or_else(|| Status::invalid_argument("set FINCH_POSTGRES_URL"))?;
        Ok(Self {
            store: PostgresStore::new(
                url,
                var("FINCH_STORE").unwrap_or_else(|| "finch_documents".to_string()),
            ),
            index_dir: PathBuf::from(var("FINCH_INDEX").unwrap_or_else(|| "finch-index".into())),
            scope: MemoryScope::new(var("FINCH_SPACE").unwrap_or_else(|| "memory".to_string())),
            embed_command: var("FINCH_EMBED_COMMAND"),
            rerank_command: var("FINCH_RERANK_COMMAND"),
        })
    }

    /// Embeddings of `texts`, one per text, from `FINCH_EMBED_COMMAND`.
    fn embed(&self, texts: &[String]) -> ZResult<Vec<Vec<f32>>> {
        let command = self
            .embed_command
            .as_deref()
            .ok_or_else(|| Status::invalid_argument("set FINCH_EMBED_COMMAND"))?;
        let vectors: Vec<Vec<f32>> = json_command("embed", command, &serde_json::json!(texts))?;
        if vectors.len() != texts.len() {
            return Err(Status::invalid_argument(format!(
                "embed command wrote {} vectors for {} texts",
                vectors.len(),
                texts.len()
            )));
        }
        Ok(vectors)
    }

    /// Scores of `documents` for `query`, one per document, from `FINCH_RERANK_COMMAND`.
    fn rerank(&self, command: &str, query: &str, documents: &[String]) -> ZResult<Vec<f32>> {
        let input = serde_json::json!({ "query": query, "documents": documents });
        let scores: Vec<f32> = json_command("rerank", command, &input)?;
        if scores.len() != documents.len() {
            return Err(Status::invalid_argument(format!(
                "rerank command wrote {} scores for {} documents",
                scores.len(),
                documents.len()
            )));
        }
        Ok(scores)
    }
}

/// Runs the shell command `command`, writes `input` as JSON to it, and parses the JSON it writes.
fn json_command<T: serde::de::DeserializeOwned>(
    name: &str,
    command: &str,
    input: &serde_json::Value,
) -> ZResult<T> {
    let mut child = Command::new("sh")
        .args(["-c", command])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(Status::from)?;
    let input = serde_json::to_vec(input).map_err(|e| Status::internal(e.to_string()))?;
    child
        .stdin
        .take()
        .ok_or_else(|| Status::internal(format!("{name} command has no input")))?
        .write_all(&input)
        .map_err(Status::from)?;
    let output = child.wait_with_output().map_err(Status::from)?;
    if !output.status.success() {
        return Err(Status::internal(format!(
            "{name} command failed with {}",
            output.status
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        Status::invalid_argument(format!("{name} command wrote no valid JSON: {error}"))
    })
}

/// An open store with its keyword index.
struct Memory {
    settings: Settings,
    store: MemoryStore,
    index: SourceLexicalIndex,
}

impl Memory {
    fn open(settings: Settings) -> ZResult<Self> {
        let store = MemoryStore::open(&settings.store)?;
        let index = SourceLexicalIndex::open(&settings.index_dir).map_err(|error| {
            Status::invalid_argument(format!(
                "no keyword index in {}: {}; run finch add",
                settings.index_dir.display(),
                error.message()
            ))
        })?;
        Ok(Self {
            settings,
            store,
            index,
        })
    }

    fn texts(&self, ids: &[String]) -> ZResult<BTreeMap<String, String>> {
        let mut texts = BTreeMap::new();
        for batch in ids.chunks(TEXT_BATCH) {
            texts.extend(self.store.source_texts(
                &self.settings.scope,
                &self.index,
                batch,
                None,
            )?);
        }
        Ok(texts)
    }

    fn text(&self, id: &str) -> ZResult<String> {
        self.texts(&[id.to_string()])?
            .remove(id)
            .ok_or_else(|| Status::not_found(format!("no document {id}")))
    }

    /// The `k` best documents for `query`, by keywords and, with `by_meaning` and an embed
    /// command, by meaning.
    fn ranked(&self, query: &str, k: usize, by_meaning: bool) -> ZResult<Vec<String>> {
        let query_embedding = match &self.settings.embed_command {
            Some(_) if by_meaning => self.settings.embed(&[query.to_string()])?.pop(),
            _ => None,
        };
        Ok(self
            .store
            .search_sources(
                &self.settings.scope,
                SourceSearch {
                    query_text: query,
                    query_embedding,
                    index: &self.index,
                    k,
                    scan_limit: k.max(200),
                    at_ms: None,
                },
            )?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    /// The document line: id, date, folder, and title.
    fn heading(&self, id: &str) -> String {
        let info = self.index.source_info(id);
        format!(
            "{id}  {}  {}  | {}",
            info.and_then(|info| info.date_ms)
                .map_or_else(|| "----------".to_string(), format_date),
            info.map_or("", |info| info.directory),
            info.map_or("", |info| info.title)
        )
    }

    /// Whether `id` passes the --source, --after, and --before filters.
    fn passes(&self, id: &str, options: &Options) -> ZResult<bool> {
        let Some(info) = self.index.source_info(id) else {
            return Ok(false);
        };
        if let Some(source) = options.flag("source") {
            let source = source.trim_end_matches('/');
            let inside =
                info.directory == source || info.directory.starts_with(&format!("{source}/"));
            if !inside {
                return Ok(false);
            }
        }
        let (after, before) = (options.date("after")?, options.date("before")?);
        if after.is_none() && before.is_none() {
            return Ok(true);
        }
        Ok(info.date_ms.is_some_and(|date| {
            after.is_none_or(|after| date >= after) && before.is_none_or(|before| date <= before)
        }))
    }

    /// Prints `ids` from `offset`, at most `limit`, each with its best line for `terms`.
    fn list(
        &self,
        ids: &[String],
        terms: &[String],
        offset: usize,
        limit: usize,
    ) -> ZResult<String> {
        if ids.is_empty() {
            return Ok("no documents found\n".to_string());
        }
        let shown = ids
            .iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();
        let texts = self.texts(&shown)?;
        let mut output = String::new();
        for (n, id) in shown.iter().enumerate() {
            let line = texts.get(id).map_or("", |text| best_line(text, terms));
            output.push_str(&format!(
                "{:>3}. {}\n       {}\n",
                offset + n + 1,
                self.heading(id),
                clip(line, 220)
            ));
        }
        let rest = ids.len().saturating_sub(offset + shown.len());
        if rest > 0 {
            output.push_str(&format!(
                "... {rest} more: add --offset {}\n",
                offset + shown.len()
            ));
        }
        Ok(output)
    }
}

fn add(settings: &Settings, options: &Options) -> ZResult<String> {
    let root = PathBuf::from(options.text());
    if !root.is_dir() {
        return Err(Status::invalid_argument("finch add needs a directory"));
    }
    let index_only = options.flag("index-only").is_some();
    let mut store = if index_only {
        Some(MemoryStore::open(&settings.store)?)
    } else {
        None
    };
    let mut builder = SourceLexicalIndexBuilder::new();
    let mut pending = Vec::new();
    let mut skipped = 0usize;
    for path in text_files(&root)? {
        let Ok(text) = std::fs::read_to_string(&path) else {
            skipped += 1;
            continue;
        };
        let relative = path.strip_prefix(&root).unwrap_or(&path);
        let document = Document::read(relative, text);
        let sequence_no = builder.source_count() as i64;
        let spans = ingest_episode(
            EpisodeInput {
                id: Some(document.id.clone()),
                scope: settings.scope.clone(),
                visibility: Visibility::Shared,
                policy_tags: Vec::new(),
                source_kind: SourceKind::Import,
                actor: ActorKind::External,
                sequence_no,
                event_time_ms: document.date_ms,
                valid_from_ms: None,
                valid_to_ms: None,
                raw_text: document.text.clone(),
                blob_ref: Some(relative.display().to_string()),
                mime_type: Some("text/plain".to_string()),
                causal_parent_ids: Vec::new(),
                metadata_json: Some(
                    serde_json::json!({"path": relative.display().to_string(), "title": document.title})
                        .to_string(),
                ),
            },
            sequence_no,
            &ChunkOptions::default(),
        )
        .spans;
        builder.add_source(IndexedSource {
            source_id: &document.id,
            title: &document.title,
            directory: &document.directory,
            body: &document.text,
            date_ms: document.date_ms,
            span_ids: spans.iter().map(|span| span.id.clone()).collect(),
        })?;
        if !index_only {
            pending.extend(spans);
            if builder.source_count().is_multiple_of(STORE_BATCH_DOCUMENTS) {
                store_spans(settings, &mut store, std::mem::take(&mut pending))?;
            }
        }
    }
    let count = builder.source_count();
    if count == 0 {
        return Err(Status::invalid_argument("no text files found"));
    }
    if !index_only {
        store_spans(settings, &mut store, pending)?;
        if let Some(store) = &store {
            store.ensure_span_hnsw_index(HnswIndexParams::new(MetricType::Cosine), None)?;
        }
    }
    builder.write(&settings.index_dir)?;
    Ok(format!(
        "added {count} documents; skipped {skipped} files that are not UTF-8 text\n"
    ))
}

/// Embeds `spans` and writes them. Opens the store first, or creates it at the embedding size.
fn store_spans(
    settings: &Settings,
    store: &mut Option<MemoryStore>,
    spans: Vec<finch_memory::SpanRecord>,
) -> ZResult<()> {
    let mut records = Vec::with_capacity(spans.len());
    for batch in spans.chunks(EMBED_BATCH) {
        let texts = batch
            .iter()
            .map(|span| span.text.clone())
            .collect::<Vec<_>>();
        records.extend(batch.iter().cloned().zip(settings.embed(&texts)?));
    }
    let Some((_, first)) = records.first() else {
        return Ok(());
    };
    if store.is_none() {
        *store = Some(match MemoryStore::open(&settings.store) {
            Ok(store) => store,
            Err(error) if error.code() == StatusCode::NotFound => {
                MemoryStore::create(&settings.store, first.len(), false)?
            }
            Err(error) => return Err(error),
        });
    }
    store
        .as_ref()
        .ok_or_else(|| Status::internal("store is not open"))?
        .append_vector_spans(&records)
}

/// Text files under `root`, in path order, without hidden files and directories.
fn text_files(root: &Path) -> ZResult<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).map_err(Status::from)? {
            let path = entry.map_err(Status::from)?.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('.'))
            {
                continue;
            }
            if path.is_dir() {
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// One file as a document: its id, title, folder, and date come from header lines such as
/// `title: ...` at the top of the file, else from the file name and the text.
struct Document {
    id: String,
    title: String,
    directory: String,
    date_ms: Option<i64>,
    text: String,
}

impl Document {
    fn read(relative: &Path, file_text: String) -> Self {
        let (header, text) = split_header(&file_text);
        let field = |names: &[&str]| {
            names.iter().find_map(|name| {
                header
                    .iter()
                    .find(|(key, value)| key == name && !value.is_empty())
                    .map(|(_, value)| value.clone())
            })
        };
        let directory = relative
            .parent()
            .map(|parent| parent.display().to_string())
            .unwrap_or_default();
        let stem = relative
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_default();
        let id = field(&["id", "document_id"])
            .unwrap_or_else(|| relative.with_extension("").display().to_string());
        let title = field(&["title", "subject"]).unwrap_or_else(|| {
            let first = text
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .unwrap_or("");
            let folder = Path::new(&directory)
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default();
            // A short first line such as a channel name says little; the file name says more.
            if first.split_whitespace().count() <= 3 || first == folder {
                format!("{first} — {}", stem.replace(['-', '_'], " "))
            } else {
                first.to_string()
            }
        });
        let date_ms = field(&[
            "date",
            "created",
            "created_at",
            "timestamp",
            "sent",
            "updated",
        ])
        .and_then(|value| first_date(&value))
        .or_else(|| first_date(&stem))
        .or_else(|| first_date(&text.lines().take(20).collect::<Vec<_>>().join("\n")));
        Self {
            id,
            title: clip(&title, TITLE_CHARS).to_string(),
            directory,
            date_ms,
            text: text.to_string(),
        }
    }
}

/// The `name: value` lines at the top of a text, with lowercase names, and the body after the
/// blank line that ends them. Without that blank line, the lines are body text, so `Ada: hi` at
/// the top of a chat stays in the chat.
fn split_header(text: &str) -> (Vec<(String, String)>, &str) {
    let mut fields = Vec::new();
    let mut offset = 0;
    for row in text.split_inclusive('\n') {
        offset += row.len();
        if row.trim().is_empty() {
            return (fields, &text[offset..]);
        }
        let Some(captures) = HEADER_LINE.captures(row.trim_end()) else {
            break;
        };
        fields.push((captures[1].to_lowercase(), captures[2].trim().to_string()));
    }
    (Vec::new(), text)
}

fn context(memory: &Memory, options: &Options) -> ZResult<String> {
    let question = options.text();
    if question.trim().is_empty() {
        return Err(Status::invalid_argument("finch context needs a question"));
    }
    let sources = options.number("sources", CONTEXT_SOURCES)?;
    let rerank = memory.settings.rerank_command.as_deref();
    let pool = match rerank {
        Some(_) => options.number("pool", CONTEXT_POOL)?.max(sources),
        None => sources,
    };
    let mut selections = memory
        .ranked(&question, pool, options.flag("exact").is_none())?
        .into_iter()
        .map(|source_id| SourceSpanSelection {
            source_id,
            max_spans: CONTEXT_SPANS_PER_SOURCE,
        })
        .collect::<Vec<_>>();
    let passages = memory.store.search_selected_sources(
        &memory.settings.scope,
        &question,
        &selections,
        &[],
        &memory.index,
        None,
    )?;
    let mut passages_of = BTreeMap::<&str, Vec<&str>>::new();
    for hit in &passages {
        passages_of
            .entry(hit.span.source_id.as_str())
            .or_default()
            .push(hit.span.text.trim());
    }
    let passages_of = |source_id: &str| passages_of.get(source_id).cloned().unwrap_or_default();
    if let Some(command) = rerank {
        let documents = selections
            .iter()
            .map(|selection| {
                let heading = memory.heading(&selection.source_id);
                format!(
                    "{heading}\n{}",
                    passages_of(&selection.source_id).join("\n...\n")
                )
            })
            .collect::<Vec<_>>();
        let scores = memory.settings.rerank(command, &question, &documents)?;
        selections = best_first(&scores, sources)
            .into_iter()
            .map(|index| selections[index].clone())
            .collect();
    }
    let mut output = String::new();
    for selection in &selections {
        output.push_str(&format!("## {}\n", memory.heading(&selection.source_id)));
        for passage in passages_of(&selection.source_id) {
            output.push_str(passage);
            output.push_str("\n...\n");
        }
        output.push('\n');
    }
    let max_chars = options.number("chars", CONTEXT_CHARS)?;
    Ok(match output.char_indices().nth(max_chars) {
        Some((end, _)) => format!("{}\n[cut at {max_chars} characters]\n", &output[..end]),
        None => output,
    })
}

/// The indexes of the `n` highest `scores`, best first; ties keep their order.
fn best_first(scores: &[f32], n: usize) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(n);
    order
}

fn find(memory: &Memory, options: &Options) -> ZResult<String> {
    let query = options.text();
    if query.trim().is_empty() {
        return Err(Status::invalid_argument("finch find needs a query"));
    }
    let (offset, limit) = (
        options.number("offset", 0)?,
        options.number("limit", FIND_LIMIT)?,
    );
    let filtered = ["source", "after", "before"]
        .iter()
        .any(|name| options.flag(name).is_some());
    let k = if filtered || options.flag("sort").is_some() {
        FILTERED_CANDIDATES
    } else {
        offset + limit
    };
    let mut ids = Vec::new();
    for id in memory.ranked(&query, k, options.flag("exact").is_none())? {
        if memory.passes(&id, options)? {
            ids.push(id);
        }
    }
    if options.flag("sort") == Some("date") {
        // Undated documents go last.
        ids.sort_by_key(|id| {
            let date = memory.index.source_info(id).and_then(|info| info.date_ms);
            (date.is_none(), date)
        });
    }
    memory.list(&ids, &lexical_terms(&query), offset, limit)
}

fn show(memory: &Memory, options: &Options) -> ZResult<String> {
    let id = options.text();
    let text = memory.text(&id)?;
    let lines = text.lines().collect::<Vec<_>>();
    let mut output = format!("# {}\n{} lines\n\n", memory.heading(&id), lines.len());
    if let Some(term) = options.flag("grep") {
        output.push_str(&matching_paragraphs(&lines, term)?);
        return Ok(output);
    }
    let (first, last) = match options.flag("lines") {
        Some(range) => parse_range(range)?,
        None => (1, SHOW_LINES),
    };
    let start = first.saturating_sub(1).min(lines.len());
    let end = last.min(lines.len()).max(start);
    output.push_str(&lines[start..end].join("\n"));
    output.push('\n');
    if end < lines.len() {
        output.push_str(&format!(
            "\n... lines {first}-{end} of {}: add --lines {}-{}\n",
            lines.len(),
            end + 1,
            end + SHOW_LINES
        ));
    }
    Ok(output)
}

/// The paragraphs that contain `term`; a long paragraph shows only its matching lines and two
/// lines around each.
fn matching_paragraphs(lines: &[&str], term: &str) -> ZResult<String> {
    let pattern = RegexBuilder::new(&regex::escape(term))
        .case_insensitive(true)
        .build()
        .map_err(|error| Status::invalid_argument(error.to_string()))?;
    let mut output = String::new();
    let mut paragraph = Vec::<(usize, &str)>::new();
    for (n, line) in lines.iter().chain(std::iter::once(&"")).enumerate() {
        if !line.trim().is_empty() {
            paragraph.push((n + 1, line));
            continue;
        }
        let hits = paragraph
            .iter()
            .enumerate()
            .filter(|(_, (_, text))| pattern.is_match(text))
            .map(|(k, _)| k)
            .collect::<Vec<_>>();
        if !hits.is_empty() {
            let keep = if paragraph.len() <= 15 {
                (0..paragraph.len()).collect::<BTreeSet<_>>()
            } else {
                hits.iter()
                    .flat_map(|k| k.saturating_sub(2)..(k + 3).min(paragraph.len()))
                    .collect()
            };
            for k in keep {
                output.push_str(&format!("{:>5}  {}\n", paragraph[k].0, paragraph[k].1));
            }
            output.push('\n');
        }
        paragraph.clear();
    }
    if output.is_empty() {
        output.push_str(&format!("no paragraph contains \"{term}\"\n"));
    }
    Ok(output)
}

fn similar(memory: &Memory, options: &Options) -> ZResult<String> {
    let id = options.text();
    let text = memory.text(&id)?;
    let total = memory.index.source_count() as f64;
    let mut term_counts = BTreeMap::<String, usize>::new();
    for term in lexical_terms(&text) {
        *term_counts.entry(term).or_insert(0) += 1;
    }
    // Rare words name the event, version, or customer; words in many documents say nothing.
    let mut weighted = term_counts
        .into_iter()
        .filter_map(|(term, count)| {
            let df = memory.index.document_frequency(&term) as f64;
            (df > 1.0 && df < total / 20.0)
                .then(|| ((1.0 + (count as f64).ln()) * (total / df).ln(), term))
        })
        .collect::<Vec<_>>();
    weighted.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut terms = weighted
        .into_iter()
        .take(SIMILAR_TERMS)
        .map(|(_, term)| term)
        .collect::<Vec<_>>();
    if let Some(info) = memory.index.source_info(&id) {
        terms.extend(lexical_terms(info.title));
    }
    let limit = options.number("limit", 15)?;
    let ids = memory
        .index
        .search(&terms.join(" "), limit + 1, 1.0)?
        .into_iter()
        .map(|hit| hit.source_id)
        .filter(|other| *other != id)
        .take(limit)
        .collect::<Vec<_>>();
    Ok(format!(
        "documents like {}:\n{}",
        memory.heading(&id),
        memory.list(&ids, &terms, 0, limit)?
    ))
}

fn grep(memory: &Memory, options: &Options) -> ZResult<String> {
    let pattern = options.text();
    let regex = RegexBuilder::new(&pattern)
        .case_insensitive(true)
        .build()
        .map_err(|error| Status::invalid_argument(error.to_string()))?;
    let words = lexical_terms(&pattern.replace(|c: char| !c.is_alphanumeric(), " "));
    if words.is_empty() {
        return Err(Status::invalid_argument(
            "finch grep needs a word in the pattern",
        ));
    }
    let candidates = memory
        .index
        .search(&words.join(" "), GREP_CANDIDATES, 1.0)?
        .into_iter()
        .map(|hit| hit.source_id)
        .filter(|id| memory.passes(id, options).unwrap_or(false))
        .collect::<Vec<_>>();
    let texts = memory.texts(&candidates)?;
    let mut matches = candidates
        .iter()
        .enumerate()
        .filter_map(|(rank, id)| {
            let text = texts.get(id)?;
            let lines = text.lines().filter(|line| regex.is_match(line));
            let count = lines.clone().count();
            let first = lines.map(str::trim).next()?;
            Some((count, rank, id, first))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let limit = options.number("limit", GREP_LIMIT)?;
    let mut output = format!(
        "{} documents match, among the {} best keyword matches\n",
        matches.len(),
        candidates.len()
    );
    for (count, _, id, first) in matches.iter().take(limit) {
        output.push_str(&format!(
            "{count:>4}x {}\n       {}\n",
            memory.heading(id),
            clip(first, 220)
        ));
    }
    if matches.len() > limit {
        output.push_str(&format!(
            "... {} more: add --limit {} or narrow with --source\n",
            matches.len() - limit,
            limit + GREP_LIMIT
        ));
    }
    Ok(output)
}

fn check(memory: &Memory, options: &Options) -> ZResult<String> {
    let Some((id, quoted)) = options.words.split_first() else {
        return Err(Status::invalid_argument("usage: finch check ID \"TEXT\""));
    };
    let wanted = quoted.join(" ");
    let wanted = wanted.trim().trim_matches(['"', '\'']);
    if wanted.is_empty() {
        return Err(Status::invalid_argument("usage: finch check ID \"TEXT\""));
    }
    let text = memory.text(id)?;
    let lower = wanted.to_lowercase();
    let hits = text
        .lines()
        .enumerate()
        .filter(|(_, line)| line.to_lowercase().contains(&lower))
        .collect::<Vec<_>>();
    if !hits.is_empty() {
        let mut output = format!("FOUND in {id} ({} lines):\n", hits.len());
        for (n, line) in hits.iter().take(5) {
            output.push_str(&format!("{:>5}  {}\n", n + 1, clip(line.trim(), 300)));
        }
        return Ok(output);
    }
    let terms = lexical_terms(wanted).into_iter().collect::<BTreeSet<_>>();
    let mut near = text
        .lines()
        .enumerate()
        .map(|(n, line)| {
            let shared = lexical_terms(line)
                .into_iter()
                .collect::<BTreeSet<_>>()
                .intersection(&terms)
                .count();
            (shared, n, line)
        })
        .filter(|(shared, _, _)| *shared > 0)
        .collect::<Vec<_>>();
    near.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut output = format!("NOT FOUND in {id}: \"{wanted}\". Closest lines:\n");
    for (_, n, line) in near.iter().take(3) {
        output.push_str(&format!("{:>5}  {}\n", n + 1, clip(line.trim(), 300)));
    }
    Ok(output)
}

/// The line of `text` that holds the most distinct words of `terms`, else its first line.
fn best_line<'a>(text: &'a str, terms: &[String]) -> &'a str {
    let wanted = terms.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let first = text.lines().map(str::trim).find(|line| !line.is_empty());
    let mut best = (first.unwrap_or(""), 0);
    for line in text.lines() {
        let shared = lexical_terms(line)
            .iter()
            .filter(|term| wanted.contains(term.as_str()))
            .collect::<BTreeSet<_>>()
            .len();
        if shared > best.1 {
            best = (line.trim(), shared);
        }
    }
    best.0
}

fn clip(text: &str, max_chars: usize) -> &str {
    text.char_indices()
        .nth(max_chars)
        .map_or(text, |(end, _)| &text[..end])
}

fn parse_range(range: &str) -> ZResult<(usize, usize)> {
    let parsed = range
        .split_once('-')
        .and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?)));
    match parsed {
        Some((first, last)) if first >= 1 && last >= first => Ok((first, last)),
        _ => Err(Status::invalid_argument(
            "--lines needs a range such as 1-300",
        )),
    }
}

/// The first `YYYY-MM-DD` date in `text`, in Unix ms at midnight UTC.
fn first_date(text: &str) -> Option<i64> {
    DATE.captures_iter(text)
        .find_map(|found| parse_date(&found[1]))
}

fn parse_date(text: &str) -> Option<i64> {
    let mut parts = text.trim().splitn(3, '-');
    let year = parts.next()?.parse::<i64>().ok()?;
    let month = parts.next()?.parse::<i64>().ok()?;
    let day = parts.next()?.get(..2)?.parse::<i64>().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400_000)
}

fn format_date(ms: i64) -> String {
    let (year, month, day) = civil_from_days(ms.div_euclid(86_400_000));
    format!("{year:04}-{month:02}-{day:02}")
}

// Howard Hinnant's day-count algorithms for the proleptic Gregorian calendar.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_first_keeps_the_highest_scores_and_breaks_ties_by_order() {
        assert_eq!(best_first(&[0.2, 0.9, 0.5, 0.9], 3), vec![1, 3, 2]);
        assert_eq!(best_first(&[0.1], 5), vec![0]);
    }

    #[test]
    fn dates_round_trip_and_reject_bad_values() {
        for date in ["1970-01-01", "2000-02-29", "2026-10-27"] {
            assert_eq!(format_date(parse_date(date).unwrap()), date);
        }
        assert_eq!(parse_date("2026-13-01"), None);
        assert_eq!(
            first_date("notes from 2025-03-04 and 2024-01-01"),
            parse_date("2025-03-04")
        );
    }

    #[test]
    fn document_reads_header_fields_then_falls_back_to_the_file() {
        let doc = Document::read(
            Path::new("jira/ops/ticket.txt"),
            "id: T-1\ntitle: Disk full on db-3\ncreated: 2026-05-01T10:00\n\nbody".to_string(),
        );
        assert_eq!(
            (doc.id.as_str(), doc.title.as_str(), doc.directory.as_str()),
            ("T-1", "Disk full on db-3", "jira/ops")
        );
        assert_eq!(doc.date_ms, parse_date("2026-05-01"));
        assert_eq!(doc.text, "body");

        let doc = Document::read(
            Path::new("slack/eng/2026-02-03-standup.txt"),
            "eng\nAda: the rollout of version 4 starts Monday".to_string(),
        );
        assert_eq!(doc.id, "slack/eng/2026-02-03-standup");
        assert_eq!(
            split_header("Ada: hi\nBo: ok"),
            (Vec::new(), "Ada: hi\nBo: ok")
        );
        let (fields, body) = split_header("id: x\npath: a/b\n\nfirst line\n");
        assert_eq!((fields.len(), body), (2, "first line\n"));
        assert_eq!(doc.title, "eng — 2026 02 03 standup");
        assert_eq!(doc.date_ms, parse_date("2026-02-03"));
    }

    #[test]
    fn matching_paragraphs_keep_short_paragraphs_whole_and_trim_long_ones() {
        let long = (0..20).map(|n| format!("row {n}")).collect::<Vec<_>>();
        let mut lines = vec!["alpha one", "beta two", ""];
        lines.extend(long.iter().map(String::as_str));
        lines[10] = "row KESTREL";
        let output = matching_paragraphs(&lines, "kestrel").unwrap();
        assert!(!output.contains("alpha"));
        assert!(
            output.contains("row KESTREL") && output.contains("row 5") && !output.contains("row 0")
        );
        assert!(matching_paragraphs(&lines, "beta")
            .unwrap()
            .contains("alpha one"));
    }
}
