mod common;
use common::*;

#[test]
fn test_create_and_open_rejects_read_only() {
    let path = temp_dir("create_readonly");
    let schema = basic_schema(4);
    let opts = CollectionOptions {
        read_only: true,
        enable_mmap: false,
        index_storage: None,
        forward_storage: None,
        max_buffer_size: CollectionOptions::DEFAULT_MAX_BUFFER_SIZE,
        forward_file_format: None,
    };
    assert!(
        Collection::create_and_open(&path, schema, opts).is_err(),
        "create_and_open with read_only=true must be rejected"
    );
}
