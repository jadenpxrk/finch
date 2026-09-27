use finch_types::{Status, StatusCode};

#[test]
fn status_default_constructor_is_ok() {
    let status = Status::default();
    assert!(status.ok());
    assert_eq!(status.code(), StatusCode::Ok);
    assert_eq!(status.message(), "");
}

#[test]
fn status_constructor_with_code_and_message() {
    let msg = "Test error message";
    let status = Status::new(StatusCode::InvalidArgument, msg);
    assert!(!status.ok());
    assert_eq!(status.code(), StatusCode::InvalidArgument);
    assert_eq!(status.message(), msg);
}

#[test]
fn status_move_preserves_payload() {
    let original = Status::new(StatusCode::NotFound, "Move test");
    let moved = original;
    assert_eq!(moved.code(), StatusCode::NotFound);
    assert_eq!(moved.message(), "Move test");
}

#[test]
fn status_comparison_operators() {
    let s1 = Status::new(StatusCode::InvalidArgument, "Error 1");
    let s2 = Status::new(StatusCode::InvalidArgument, "Error 1");
    let s3 = Status::new(StatusCode::NotFound, "Error 2");
    let ok1 = Status::default();
    let ok2 = Status::default();

    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
    assert_eq!(ok1, ok2);
    assert_ne!(s1, ok1);
}

#[test]
fn status_factories_and_display() {
    let invalid = Status::invalid_argument("Invalid arg: 42");
    assert_eq!(invalid.code(), StatusCode::InvalidArgument);
    assert!(!invalid.message().is_empty());

    let not_found = Status::not_found("Not found: key");
    assert_eq!(not_found.code(), StatusCode::NotFound);
    assert!(!not_found.message().is_empty());

    let already_exists = Status::already_exists("Already exists: item");
    assert_eq!(already_exists.code(), StatusCode::AlreadyExists);
    assert!(!already_exists.message().is_empty());

    let internal_error = Status::internal("Internal error: details");
    assert_eq!(internal_error.code(), StatusCode::Internal);
    assert!(!internal_error.message().is_empty());

    let permission_denied = Status::permission_denied("Permission denied: resource");
    assert_eq!(permission_denied.code(), StatusCode::PermissionDenied);
    assert!(!permission_denied.message().is_empty());

    let ok = Status::default();
    assert!(ok.ok());
    assert_eq!(ok.code(), StatusCode::Ok);
    assert_eq!(ok.message(), "");

    let s = Status::new(StatusCode::InvalidArgument, "Stream test");
    let rendered = s.to_string();
    assert!(rendered.contains(Status::default_message(StatusCode::InvalidArgument)));
    assert!(rendered.contains("Stream test"));

    let ok_rendered = Status::default().to_string();
    assert!(ok_rendered.contains("OK"));
}
