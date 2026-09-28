use super::*;
use finch_db::crc32c_hash;
use finch_types::Doc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;

const STATE_MUTATION_JOURNAL: &str = ".state-mutation-journal.json";

#[derive(Serialize, Deserialize)]
struct DocumentBeforeImage {
    pk: String,
    doc: Option<Doc>,
}

#[derive(Serialize, Deserialize)]
struct CollectionChanges {
    name: String,
    documents: Vec<DocumentBeforeImage>,
}

/// The journal header; a journal written before records were appended holds every change here.
#[derive(Serialize, Deserialize)]
struct StateMutationJournal {
    scope: MemoryScope,
    collections: Vec<CollectionChanges>,
}

/// The running mutation's journal file, open for appends, and the rows it already holds.
pub(crate) struct OpenStateMutationJournal {
    file: fs::File,
    captured: BTreeMap<String, BTreeSet<String>>,
}

impl MemoryStore {
    pub(crate) fn begin_state_mutation_journal(&self, scope: &MemoryScope) -> ZResult<()> {
        let path = self.path.join(STATE_MUTATION_JOURNAL);
        let journal = StateMutationJournal {
            scope: scope.clone(),
            collections: Vec::new(),
        };
        let bytes = serde_json::to_vec(&journal).map_err(json_error)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .map_err(|error| {
                Status::io_error(format!(
                    "cannot start state mutation journal at {}: {error}",
                    path.display()
                ))
            })?;
        if let Err(error) = file.write_all(&bytes).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&path);
            return Err(Status::io_error(error.to_string()));
        }
        sync_directory(&self.path)?;
        *self.state_mutation_journal.lock() = Some(OpenStateMutationJournal {
            file,
            captured: BTreeMap::new(),
        });
        Ok(())
    }

    pub(crate) fn capture_state_mutation_documents(
        &self,
        collection_name: &str,
        collection: &Collection,
        pks: impl IntoIterator<Item = String>,
    ) -> ZResult<()> {
        let mut journal = self.state_mutation_journal.lock();
        let Some(journal) = journal.as_mut() else {
            return Err(Status::internal(format!(
                "state write to {collection_name} outside a journaled state mutation"
            )));
        };
        let captured = journal
            .captured
            .entry(collection_name.to_string())
            .or_default();
        let mut missing = pks
            .into_iter()
            .filter(|pk| !captured.contains(pk))
            .collect::<Vec<_>>();
        missing.sort();
        missing.dedup();
        if missing.is_empty() {
            return Ok(());
        }
        let existing = collection.fetch(missing.clone())?;
        let changes = CollectionChanges {
            name: collection_name.to_string(),
            documents: missing
                .iter()
                .map(|pk| DocumentBeforeImage {
                    doc: existing.get(pk).map(|doc| (**doc).clone()),
                    pk: pk.clone(),
                })
                .collect(),
        };
        append_record(
            &mut journal.file,
            &serde_json::to_vec(&changes).map_err(json_error)?,
        )?;
        captured.extend(missing);
        Ok(())
    }

    pub(crate) fn capture_state_mutation_docs(
        &self,
        collection_name: &str,
        collection: &Collection,
        docs: &[Doc],
    ) -> ZResult<()> {
        #[cfg(test)]
        if !docs.is_empty() {
            count_state_write_toward_crash();
        }
        self.capture_state_mutation_documents(
            collection_name,
            collection,
            docs.iter().map(|doc| doc.pk.clone()),
        )
    }

    pub(crate) fn commit_state_mutation_journal(&self) -> ZResult<()> {
        self.state_mutation_journal.lock().take();
        remove_journal(&self.path.join(STATE_MUTATION_JOURNAL))
    }

    pub(crate) fn recover_pending_state_mutation(&self) -> ZResult<()> {
        self.state_mutation_journal.lock().take();
        #[cfg(test)]
        if FAIL_NEXT_ROLLBACK.with(|fail| fail.replace(false)) {
            return Err(Status::io_error("injected rollback failure"));
        }
        let path = self.path.join(STATE_MUTATION_JOURNAL);
        if !path.exists() {
            return Ok(());
        }
        let journal = read_journal(&path)?;
        if self.claims.options().read_only {
            return Err(Status::invalid_argument(format!(
                "pending state mutation requires writable recovery at {}",
                path.display()
            )));
        }
        let collections = self.state_mutation_collections();
        for changes in journal.collections {
            let collection = journal_collection(&collections, &changes.name)?;
            let mut restore = Vec::new();
            let mut remove = Vec::new();
            for before in changes.documents {
                match before.doc {
                    Some(doc) => restore.push(doc),
                    None => remove.push(before.pk),
                }
            }
            for chunk in remove.chunks(1024) {
                ensure_delete_statuses(collection.delete(chunk.to_vec())?)?;
            }
            for chunk in restore.chunks(1024) {
                ensure_statuses(collection.upsert(chunk.to_vec())?)?;
            }
        }
        remove_journal(&path)
    }

    fn state_mutation_collections(&self) -> Vec<(&'static str, &Collection)> {
        vec![
            (ENTITIES_COLLECTION, self.entities.as_ref()),
            (ENTITY_ALIASES_COLLECTION, self.entity_aliases.as_ref()),
            (CLAIMS_COLLECTION, self.claims.as_ref()),
            (RULES_COLLECTION, self.rules.as_ref()),
            (CORRECTIONS_COLLECTION, self.corrections.as_ref()),
            (STATE_RECORDS_COLLECTION, self.state_records.as_ref()),
            (
                DEPENDENCY_TRACES_COLLECTION,
                self.dependency_traces.as_ref(),
            ),
            (SLOTS_COLLECTION, self.slots.as_ref()),
            (SLOT_ALIASES_COLLECTION, self.slot_aliases.as_ref()),
        ]
    }
}

#[cfg(test)]
thread_local! {
    // Test crash point: this many journaled writes succeed, then the next one panics before it runs.
    pub(crate) static STATE_WRITES_BEFORE_CRASH: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
    // Test fault: the next rollback of a state mutation fails before it reads the journal.
    pub(crate) static FAIL_NEXT_ROLLBACK: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    // Every journal record append: the bytes it wrote and the file length after it.
    pub(crate) static JOURNAL_APPENDS: std::cell::RefCell<Vec<(u64, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn count_state_write_toward_crash() {
    STATE_WRITES_BEFORE_CRASH.with(|remaining| match remaining.get() {
        Some(0) => {
            remaining.set(None);
            panic!("injected crash before a state write");
        }
        Some(writes) => remaining.set(Some(writes - 1)),
        None => {}
    });
}

/// The header, then every whole record in the order it was appended.
fn read_journal(path: &Path) -> ZResult<StateMutationJournal> {
    let bytes = fs::read(path).map_err(|error| Status::io_error(error.to_string()))?;
    let mut header = serde_json::Deserializer::from_slice(&bytes).into_iter();
    let mut journal: StateMutationJournal = header
        .next()
        .ok_or_else(|| {
            Status::io_error(format!(
                "state mutation journal {} has no header",
                path.display()
            ))
        })?
        .map_err(json_error)?;
    let mut rest = &bytes[header.byte_offset()..];
    while let Some(payload) = next_record(&mut rest, path)? {
        let changes: CollectionChanges = serde_json::from_slice(payload).map_err(json_error)?;
        match journal
            .collections
            .iter_mut()
            .find(|collection| collection.name == changes.name)
        {
            Some(collection) => collection.documents.extend(changes.documents),
            None => journal.collections.push(changes),
        }
    }
    Ok(journal)
}

/// The next whole record's payload, or `None` at the end or at a final record a crash cut short.
fn next_record<'a>(rest: &mut &'a [u8], path: &Path) -> ZResult<Option<&'a [u8]>> {
    if rest.is_empty() {
        return Ok(None);
    }
    if let Some((payload, size)) = whole_record(rest) {
        *rest = &rest[size..];
        return Ok(Some(payload));
    }
    // Each record is synced before the next starts, so only the last can be torn, and a torn
    // record has no newline yet; compact JSON payloads never contain one.
    if !rest.contains(&b'\n') {
        return Ok(None);
    }
    Err(Status::io_error(format!(
        "state mutation journal {} has a corrupt record; inspect or move it aside before reopening",
        path.display()
    )))
}

/// A record is `{length:08x}{payload}{crc32c:08x}\n`; returns the payload and the record size.
fn whole_record(bytes: &[u8]) -> Option<(&[u8], usize)> {
    let length = hex_u32(bytes.get(..8)?)? as usize;
    let payload = bytes.get(8..8 + length)?;
    let checksum = hex_u32(bytes.get(8 + length..16 + length)?)?;
    (bytes.get(16 + length) == Some(&b'\n') && checksum == crc32c_hash(payload, 0))
        .then_some((payload, 17 + length))
}

fn hex_u32(digits: &[u8]) -> Option<u32> {
    u32::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()
}

fn append_record(file: &mut fs::File, payload: &[u8]) -> ZResult<()> {
    let length = u32::try_from(payload.len())
        .map_err(|_| Status::invalid_argument("state mutation journal record too large"))?;
    let mut record = format!("{length:08x}").into_bytes();
    record.extend_from_slice(payload);
    record.extend_from_slice(format!("{:08x}\n", crc32c_hash(payload, 0)).as_bytes());
    file.write_all(&record)
        .and_then(|_| file.sync_data())
        .map_err(|error| Status::io_error(error.to_string()))?;
    #[cfg(test)]
    JOURNAL_APPENDS.with(|appends| {
        let length = file.metadata().map_or(0, |metadata| metadata.len());
        appends.borrow_mut().push((record.len() as u64, length));
    });
    Ok(())
}

fn remove_journal(path: &Path) -> ZResult<()> {
    fs::remove_file(path).map_err(|error| Status::io_error(error.to_string()))?;
    path.parent().map_or(Ok(()), sync_directory)
}

fn sync_directory(path: &Path) -> ZResult<()> {
    fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| Status::io_error(error.to_string()))
}

fn ensure_statuses(statuses: Vec<Status>) -> ZResult<()> {
    if let Some(error) = statuses.into_iter().find(|status| !status.is_ok()) {
        return Err(error);
    }
    Ok(())
}

fn ensure_delete_statuses(statuses: Vec<Status>) -> ZResult<()> {
    if let Some(error) = statuses
        .into_iter()
        .find(|status| !status.is_ok() && !status.is_not_found())
    {
        return Err(error);
    }
    Ok(())
}

fn journal_collection<'a>(
    collections: &[(&'static str, &'a Collection)],
    name: &str,
) -> ZResult<&'a Collection> {
    collections
        .iter()
        .find(|(collection_name, _)| *collection_name == name)
        .map(|(_, collection)| *collection)
        .ok_or_else(|| {
            Status::internal(format!(
                "state mutation journal references unknown collection {name}"
            ))
        })
}
