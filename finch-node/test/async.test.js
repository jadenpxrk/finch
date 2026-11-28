const test = require('node:test');
const assert = require('node:assert');
const finch = require('..');

// Memory stores live in Postgres; the test needs a server with pgvector.
const postgresUrl = process.env.FINCH_MEMORY_TEST_POSTGRES_URL;

test('memory writes resolve through the async store', { skip: !postgresUrl }, async () => {
  const memory = await finch.FinchMemory.create({
    url: postgresUrl,
    name: `node_async_${process.pid}`,
    embeddingDim: 4,
  });
  const written = await memory.ingestEpisode({ scope: { space: 's' }, text: 'hello world' });
  assert.ok(written.episode);
  const { hits } = await memory.search({ scope: { space: 's' }, query: 'hello', mode: 'keyword' });
  assert.strictEqual(hits.length, 1);
});

test('a wrapper call with bad input rejects its promise', async () => {
  const memory = new finch.FinchMemory({});
  for (const call of [
    () => memory.search({ scope: {}, query: 'hello' }),
    () => memory.ingestEpisode({ scope: {}, text: 'hello' }),
    () => memory.extractionBacklog({}),
  ]) {
    let promise;
    assert.doesNotThrow(() => { promise = call(); });
    await assert.rejects(promise, /scope.space is required/);
  }
  // A misspelled key must not widen the scope to the whole space.
  await assert.rejects(memory.search({ scope: { space: 's', user: 'u1' }, query: 'hello' }), /unknown keys: user/);
});

test('a queue call with a tenant reaches only the jobs of that tenant', { skip: !postgresUrl }, async () => {
  const memory = await finch.FinchMemory.create({
    url: postgresUrl,
    name: `node_queue_${process.pid}_${Date.now()}`,
    embeddingDim: 4,
  });
  for (const tenantId of ['a', 'b']) {
    await memory.ingestEpisode({ scope: { space: 's', tenantId }, text: `hello from ${tenantId}` });
  }
  const leaseAndFail = async () => {
    const { jobs } = await memory.leaseExtractionJobs({ workerId: 'w', tenantId: 'a' });
    assert.deepStrictEqual(jobs.map((job) => job.scope.tenant_id), ['a']);
    const failure = { jobId: jobs[0].id, workerId: 'w', error: 'timeout', maxAttempts: 1, tenantId: 'a' };
    assert.strictEqual((await memory.failExtractionJob(failure)).status, 'failed');
    return jobs[0].id;
  };
  const jobId = await leaseAndFail();
  await assert.rejects(memory.retryExtractionJob(jobId, 'b'), /no extraction job/);
  assert.strictEqual((await memory.retryExtractionJob(jobId, 'a')).status, 'pending');
  assert.strictEqual(await leaseAndFail(), jobId);
  assert.strictEqual((await memory.skipExtractionJob(jobId, 'a')).status, 'skipped');
});
