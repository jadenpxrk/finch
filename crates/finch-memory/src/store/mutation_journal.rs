use super::*;
use finch_types::Doc;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;

const STATE_MUTATION_JOURNAL: &str = ".state-mutation-journal.json";
const STATE_MUTATION_JOURNAL_TMP: &str = ".state-mutation-journal.json.tmp";

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

#[derive(Serialize, Deserialize)]
struct StateMutationJournal {
    scope: MemoryScope,
    collections: Vec<CollectionChanges>,
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
            .write(true)
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
        sync_directory(&self.path)
    }

    pub(crate) fn capture_state_mutation_documents(
        &self,
        collection_name: &str,
        collection: &Collection,
        pks: impl IntoIterator<Item = String>,
    ) -> ZResult<()> {
        let path = self.path.join(STATE_MUTATION_JOURNAL);
        if !path.exists() {
            return Ok(());
        }
        let mut journal = read_journal(&path)?;
        let position = journal
            .collections
            .iter()
            .position(|changes| changes.name == collection_name)
            .unwrap_or_else(|| {
                journal.collections.push(CollectionChanges {
                    name: collection_name.to_string(),
                    documents: Vec::new(),
                });
                journal.collections.len() - 1
            });
        let changes = &mut journal.collections[position];
        let captured = changes
            .documents
            .iter()
            .map(|before| before.pk.as_str())
            .collect::<BTreeSet<_>>();
        let mut missing = pks
            .into_iter()
            .filter(|pk| !captured.contains(pk.as_str()))
            .collect::<Vec<_>>();
        missing.sort();
        missing.dedup();
        if missing.is_empty() {
            return Ok(());
        }
        let existing = collection.fetch(missing.clone())?;
        changes
            .documents
            .extend(missing.into_iter().map(|pk| DocumentBeforeImage {
                doc: existing.get(&pk).map(|doc| (**doc).clone()),
                pk,
            }));
        write_journal(&self.path, &journal)
    }

    pub(crate) fn capture_state_mutation_docs(
        &self,
        collection_name: &str,
        collection: &Collection,
        docs: &[Doc],
    ) -> ZResult<()> {
        self.capture_state_mutation_documents(
            collection_name,
            collection,
            docs.iter().map(|doc| doc.pk.clone()),
        )
    }

    pub(crate) fn commit_state_mutation_journal(&self) -> ZResult<()> {
        remove_journal(&self.path.join(STATE_MUTATION_JOURNAL))
    }

    pub(crate) fn recover_pending_state_mutation(&self) -> ZResult<()> {
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

fn read_journal(path: &Path) -> ZResult<StateMutationJournal> {
    let bytes = fs::read(path).map_err(|error| Status::io_error(error.to_string()))?;
    serde_json::from_slice(&bytes).map_err(json_error)
}

fn write_journal(root: &Path, journal: &StateMutationJournal) -> ZResult<()> {
    let path = root.join(STATE_MUTATION_JOURNAL);
    let temporary = root.join(STATE_MUTATION_JOURNAL_TMP);
    let bytes = serde_json::to_vec(journal).map_err(json_error)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| Status::io_error(error.to_string()))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| Status::io_error(error.to_string()))?;
    fs::rename(&temporary, &path)
        .map_err(|error| Status::io_error(error.to_string()))
        .and_then(|_| sync_directory(root))
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
