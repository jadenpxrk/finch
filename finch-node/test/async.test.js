const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const finch = require('..');

const NOT_FOUND = 1;
const INVALID_ARGUMENT = 3;

function tempDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'finch-node-'));
}

function schema() {
  const hnswIndex = { metric: finch.MetricType.L2, m: 16, efConstruction: 200, quantize: 0 };
  return {
    name: 'async_test',
    fields: [{ name: 'v', dataType: finch.DataType.VectorFp32, dimension: 16, hnswIndex }],
  };
}

function docs(start, n) {
  return Array.from({ length: n }, (_, i) => ({
    pk: String(start + i),
    fields: { v: Array.from({ length: 16 }, (_, j) => Math.sin(start + i + j)) },
  }));
}

test('optimize runs off the JS thread while a timer keeps firing', async () => {
  const dir = tempDir();
  try {
    const col = await finch.createAndOpen(path.join(dir, 'c'), schema());
    for (let batch = 0; batch < 2; batch += 1) {
      await col.insert(docs(batch * 150, 150));
      await col.flush();
    }

    let ticks = 0;
    const timer = setInterval(() => { ticks += 1; }, 1);
    try {
      const pending = col.optimize();
      assert.ok(pending instanceof Promise, 'optimize must return a promise');
      await pending;
    } finally {
      clearInterval(timer);
    }
    assert.ok(ticks > 0, 'the timer never fired while optimize ran');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('errors carry the finch status code', async () => {
  const dir = tempDir();
  try {
    await assert.rejects(finch.open(path.join(dir, 'missing')), (err) => err.code === NOT_FOUND);

    const col = await finch.createAndOpen(path.join(dir, 'c'), schema());
    assert.throws(() => col.querySql('SELECT FROM'), (err) => err.code === INVALID_ARGUMENT);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

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
