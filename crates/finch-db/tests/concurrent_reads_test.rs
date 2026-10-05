mod common;
use common::*;

#[test]
fn test_concurrent_reads() {
    let path = temp_dir("concurrent");
    let schema = basic_schema(4);
    let col =
        Arc::new(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

    col.insert(vec![
        make_doc("d1", vec![1.0, 0.0, 0.0, 0.0], "a"),
        make_doc("d2", vec![0.0, 1.0, 0.0, 0.0], "b"),
    ])
    .unwrap();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let col_clone = col.clone();
            std::thread::spawn(move || {
                let q = VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 2);
                let results = col_clone.query(q).unwrap();
                assert!(!results.is_empty());
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_high_concurrency_queries_do_not_deadlock() {
    let path = temp_dir("concurrent_high");
    let schema = basic_schema(16);
    let col =
        Arc::new(Collection::create_and_open(&path, schema, CollectionOptions::default()).unwrap());

    // Create one persisted segment and leave some docs in the writing segment so
    // each query touches both paths.
    let docs: Vec<Doc> = (0..200)
        .map(|i| {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            make_doc(&format!("p{}", i), v, if i % 2 == 0 { "a" } else { "b" })
        })
        .collect();
    col.insert(docs).unwrap();
    col.flush().unwrap();

    let docs2: Vec<Doc> = (200..260)
        .map(|i| {
            let mut v = vec![0.0f32; 16];
            v[i % 16] = 1.0;
            make_doc(&format!("w{}", i), v, if i % 2 == 0 { "a" } else { "b" })
        })
        .collect();
    col.insert(docs2).unwrap();

    let threads = 16usize;
    let iters = 25usize;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let col_clone = col.clone();
            std::thread::spawn(move || {
                for i in 0..iters {
                    let mut q = vec![0.0f32; 16];
                    q[(t + i) % 16] = 1.0;
                    let query = VectorQuery::new("emb", q, 10).with_filter("label = 'a'");
                    let results = col_clone.query(query).unwrap();
                    assert!(!results.is_empty());
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    drop(col);
    std::fs::remove_dir_all(&path).ok();
}

#[test]
fn test_query_racing_upsert_never_returns_both_versions() {
    let path = temp_dir("query_upsert_race");
    let col =
        Collection::create_and_open(&path, basic_schema(4), CollectionOptions::default()).unwrap();
    col.insert(vec![
        make_doc("p", vec![1.0, 0.0, 0.0, 0.0], "v0"),
        make_doc("q", vec![0.0, 1.0, 0.0, 0.0], "other"),
    ])
    .unwrap();
    col.flush().unwrap();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|reader| {
            let col = col.clone();
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut duplicates = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    let query = if reader % 2 == 0 {
                        VectorQuery::new("emb", vec![1.0, 0.0, 0.0, 0.0], 10)
                    } else {
                        VectorQuery::new("", Vec::new(), 10).with_filter("label != 'other'")
                    };
                    let hits = col.query(query).unwrap();
                    if hits.iter().filter(|doc| doc.pk == "p").count() > 1 {
                        duplicates += 1;
                    }
                }
                duplicates
            })
        })
        .collect();
    for i in 1..=2000 {
        let doc = make_doc("p", vec![1.0, 0.0, 0.0, 0.0], &format!("v{i}"));
        assert!(col.upsert(vec![doc]).unwrap()[0].is_ok());
    }
    stop.store(true, Ordering::Relaxed);
    let duplicates: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    drop(col);
    std::fs::remove_dir_all(&path).ok();
    assert_eq!(
        duplicates, 0,
        "queries returned both versions of an upserted pk"
    );
}
