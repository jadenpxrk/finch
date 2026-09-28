// Binding signatures mirror the JS argument lists, which are the wire contract.
#![allow(clippy::too_many_arguments)]

mod collection;
mod doc;
mod memory;
mod schema;
mod types;

use napi::bindgen_prelude::*;

/// A JS error whose `code` property is the numeric finch `StatusCode`.
pub(crate) fn finch_err(env: &Env, status: finch_types::Status) -> napi::Error {
    let build = || -> Result<napi::Error> {
        let mut err = env.create_error(napi::Error::from_reason(status.to_string()))?;
        err.set_named_property("code", status.code as u32)?;
        Ok(napi::Error::from(err.into_unknown()))
    };
    build().unwrap_or_else(|e| e)
}

/// Runs finch work on the libuv thread pool and settles the returned promise with its result.
pub struct BlockingTask<T>(Option<Box<dyn FnOnce() -> finch_types::ZResult<T> + Send>>);

impl<T: ToNapiValue + TypeName + Send + 'static> BlockingTask<T> {
    pub(crate) fn spawn(
        work: impl FnOnce() -> finch_types::ZResult<T> + Send + 'static,
    ) -> AsyncTask<Self> {
        AsyncTask::new(Self(Some(Box::new(work))))
    }
}

impl<T: ToNapiValue + TypeName + Send + 'static> Task for BlockingTask<T> {
    type Output = finch_types::ZResult<T>;
    type JsValue = T;

    fn compute(&mut self) -> Result<Self::Output> {
        let work = self
            .0
            .take()
            .ok_or_else(|| napi::Error::from_reason("task already ran"))?;
        Ok(work())
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> Result<T> {
        output.map_err(|status| finch_err(&env, status))
    }
}
