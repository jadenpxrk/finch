//! Dates for the day words of retrieved text, so a reader does not do calendar arithmetic.

use std::borrow::Cow;

use crate::retrieval::SpanSearchHit;

const MS_PER_DAY: i64 = 86_400_000;
/// Weekday names in the order of the days that follow 1970-01-01, a Thursday.
const WEEKDAYS: [&str; 7] = [
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "monday",
    "tuesday",
    "wednesday",
];

/// Returns the hits with a date after each day word of their text, as in
/// `by Thursday [2025-07-17]`.
///
/// A day word is an English weekday name, `today`, `tonight`, `tomorrow`, or `EOD`. The date
/// counts from the day of the span's `valid_from_ms` at `utc_offset_minutes` from UTC, the
/// speaker's time zone. A weekday name means the first such day
/// on or after that day. A weekday name after `last`, `next`, `previous`, or `past` gets no date,
/// and neither does `today`, `tonight`, or `EOD` before another day word. A span without
/// `valid_from_ms` stays as it is.
pub fn date_day_words(hits: &[SpanSearchHit], utc_offset_minutes: i32) -> Vec<SpanSearchHit> {
    let offset_ms = i64::from(utc_offset_minutes) * 60_000;
    hits.iter()
        .map(|hit| {
            let mut hit = hit.clone();
            if let Some(valid_from_ms) = hit.span.valid_from_ms {
                let said_on = valid_from_ms
                    .saturating_add(offset_ms)
                    .div_euclid(MS_PER_DAY);
                hit.span.text = text_with_dates(&hit.span.text, said_on);
            }
            hit
        })
        .collect()
}

/// `text` with the date after each day word. `said_on` counts days since 1970-01-01.
fn text_with_dates(text: &str, said_on: i64) -> String {
    let words = ascii_words(text);
    let lower = words
        .iter()
        .map(|&(start, end)| text[start..end].to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut out = String::with_capacity(text.len() + 16);
    let mut copied = 0usize;
    for (index, &(_, end)) in words.iter().enumerate() {
        let previous = index.checked_sub(1).map(|i| lower[i].as_str());
        let next = lower.get(index + 1).map(String::as_str);
        let Some(day) = day_of_word(&lower[index], previous, next, said_on) else {
            continue;
        };
        let (year, month, day) = civil_date(day);
        out.push_str(&text[copied..end]);
        out.push_str(&format!(" [{year:04}-{month:02}-{day:02}]"));
        copied = end;
    }
    out.push_str(&text[copied..]);
    out
}

/// The day a day word names, as days since 1970-01-01.
fn day_of_word(
    word: &str,
    previous: Option<&str>,
    next: Option<&str>,
    said_on: i64,
) -> Option<i64> {
    let weekday = |word: &str| WEEKDAYS.iter().position(|&name| name == word);
    if let Some(weekday) = weekday(word) {
        // "last Friday" and "next Friday" need the speaker's meaning. The reader resolves them.
        let qualified = matches!(previous, Some("last" | "next" | "previous" | "past"));
        return (!qualified).then(|| said_on + (weekday as i64 - said_on).rem_euclid(7));
    }
    match word {
        "tomorrow" => Some(said_on + 1),
        // "EOD Friday" names Friday, so only the weekday gets the date.
        "today" | "tonight" | "eod" => {
            let before_day_word =
                next.is_some_and(|next| weekday(next).is_some() || next == "tomorrow");
            (!before_day_word).then_some(said_on)
        }
        _ => None,
    }
}

/// Byte ranges of the runs of ASCII letters in `text`.
fn ascii_words(text: &str) -> Vec<(usize, usize)> {
    let mut words = Vec::new();
    let mut start = None;
    for (index, byte) in text.bytes().enumerate() {
        match (byte.is_ascii_alphabetic(), start) {
            (true, None) => start = Some(index),
            (false, Some(word_start)) => {
                words.push((word_start, index));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(word_start) = start {
        words.push((word_start, text.len()));
    }
    words
}

/// Whether `quote` is in `text` when the dates that `date_day_words` adds are left out of both.
/// A quote copied from a packed context then matches the stored span, and the reverse.
pub(crate) fn quote_matches(text: &str, quote: &str) -> bool {
    let quote = without_day_word_dates(quote);
    !quote.trim().is_empty() && without_day_word_dates(text).contains(quote.as_ref())
}

/// `text` without each ` [YYYY-MM-DD]` tag.
fn without_day_word_dates(text: &str) -> Cow<'_, str> {
    const TAG_LEN: usize = " [YYYY-MM-DD]".len();
    let is_tag = |tag: &[u8]| {
        tag.len() == TAG_LEN
            && tag.starts_with(b" [")
            && tag[TAG_LEN - 1] == b']'
            && tag[2..TAG_LEN - 1].iter().enumerate().all(|(i, byte)| {
                if i == 4 || i == 7 {
                    *byte == b'-'
                } else {
                    byte.is_ascii_digit()
                }
            })
    };
    let bytes = text.as_bytes();
    let mut out = String::new();
    let mut copied = 0usize;
    let mut index = 0usize;
    while index + TAG_LEN <= bytes.len() {
        if is_tag(&bytes[index..index + TAG_LEN]) {
            out.push_str(&text[copied..index]);
            index += TAG_LEN;
            copied = index;
        } else {
            index += 1;
        }
    }
    if copied == 0 {
        return Cow::Borrowed(text);
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// The year, month, and day of a count of days since 1970-01-01, in the Gregorian calendar.
pub(crate) fn civil_date(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    (year_of_era + era * 400 + i64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MemoryScope, MemoryStatus, SourceType, SpanRecord, Visibility};

    /// 2025-07-15, a Tuesday, as days since 1970-01-01.
    const TUESDAY: i64 = 20_284;

    #[test]
    fn civil_date_matches_known_days() {
        assert_eq!(civil_date(0), (1970, 1, 1));
        assert_eq!(civil_date(TUESDAY), (2025, 7, 15));
        assert_eq!(civil_date(19_782), (2024, 2, 29));
        assert_eq!(civil_date(-1), (1969, 12, 31));
    }

    #[test]
    fn weekday_name_gets_the_first_such_day_on_or_after_the_message_day() {
        assert_eq!(
            text_with_dates("Confirm the fields by Thursday.", TUESDAY),
            "Confirm the fields by Thursday [2025-07-17]."
        );
        assert_eq!(
            text_with_dates("Tuesday works. Monday is too late.", TUESDAY),
            "Tuesday [2025-07-15] works. Monday [2025-07-21] is too late."
        );
    }

    #[test]
    fn relative_day_words_get_their_date() {
        assert_eq!(
            text_with_dates("Send it today, review tomorrow, sign off by EOD.", TUESDAY),
            "Send it today [2025-07-15], review tomorrow [2025-07-16], sign off by EOD [2025-07-15]."
        );
        assert_eq!(
            text_with_dates("Ship tomorrow.", 20_453),
            "Ship tomorrow [2026-01-01]."
        );
    }

    #[test]
    fn qualified_and_compound_day_words_keep_the_speakers_meaning() {
        assert_eq!(
            text_with_dates("Last Friday slipped; next Friday is the target.", TUESDAY),
            "Last Friday slipped; next Friday is the target."
        );
        assert_eq!(
            text_with_dates("Post it by EOD Friday.", TUESDAY),
            "Post it by EOD Friday [2025-07-18]."
        );
    }

    #[test]
    fn text_without_day_words_and_non_ascii_text_stay_whole() {
        let text = "Reunión el día 3 — sin cambios ✓, Fridays are busy";
        assert_eq!(text_with_dates(text, TUESDAY), text);
    }

    #[test]
    fn hit_without_a_valid_from_time_stays_as_it_is() {
        let hit = |valid_from_ms| SpanSearchHit {
            score: 1.0,
            span: SpanRecord {
                id: "span_a".to_string(),
                scope: MemoryScope::new("default"),
                status: MemoryStatus::Active,
                visibility: Visibility::Private,
                policy_tags: Vec::new(),
                source_type: SourceType::Episode,
                source_id: "ep_1".to_string(),
                byte_start: 0,
                byte_end: 16,
                char_start: 0,
                char_end: 16,
                token_start: None,
                token_end: None,
                span_index: 0,
                text: "Reply by Friday.".to_string(),
                text_hash: "span_a".to_string(),
                chunker_version: "test".to_string(),
                embedding_model: None,
                embedding_version: None,
                lexical_text: "reply by friday.".to_string(),
                created_at_ms: 0,
                valid_from_ms,
                valid_to_ms: None,
                provenance: Vec::new(),
            },
        };
        let dated = date_day_words(&[hit(Some(TUESDAY * MS_PER_DAY + 3_600_000)), hit(None)], 0);
        assert_eq!(dated[0].span.text, "Reply by Friday [2025-07-18].");
        assert_eq!(dated[1].span.text, "Reply by Friday.");
        // 01:00 UTC on Tuesday is Monday evening at UTC-7, so Friday is the same date.
        let pacific = date_day_words(&[hit(Some(TUESDAY * MS_PER_DAY + 3_600_000))], -420);
        assert_eq!(pacific[0].span.text, "Reply by Friday [2025-07-18].");
        let tomorrow = |offset| {
            let mut hit = hit(Some(TUESDAY * MS_PER_DAY + 3_600_000));
            hit.span.text = "Ship tomorrow.".to_string();
            date_day_words(&[hit], offset).remove(0).span.text
        };
        assert_eq!(tomorrow(0), "Ship tomorrow [2025-07-16].");
        assert_eq!(tomorrow(-420), "Ship tomorrow [2025-07-15].");
    }

    #[test]
    fn quote_matches_with_or_without_the_added_dates() {
        let stored = "Reply by Friday.";
        let packed = "Reply by Friday [2025-07-18].";
        assert!(quote_matches(stored, "by Friday [2025-07-18]."));
        assert!(quote_matches(packed, "by Friday."));
        assert!(quote_matches(packed, "by Friday [2025-07-18]."));
        assert!(!quote_matches(stored, "by Monday."));
        assert!(!quote_matches(stored, " [2025-07-18]"));
        assert_eq!(
            without_day_word_dates("Due [2025-7-18]."),
            "Due [2025-7-18]."
        );
    }
}
