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
