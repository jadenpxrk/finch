use finch_types::ZResult;

use crate::version::Version;

use super::Collection;

impl Collection {
    pub(super) fn commit_manifest_with_prepared<T>(
        &self,
        new_version: &Version,
        prepared: T,
        publish: impl FnOnce(T) -> ZResult<()>,
        abort: impl FnOnce(T),
    ) -> ZResult<()> {
        match self.version_manager.flush(new_version) {
            Ok(()) => publish(prepared),
            Err(e) => {
                abort(prepared);
                Err(e)
            }
        }
    }
}
