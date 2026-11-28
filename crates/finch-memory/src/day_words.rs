//! Dates for the day words of retrieved text, so a reader does not do calendar arithmetic.

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
        out.push_str(&text[copied..end]);
        out.push_str(&format!(" [{}]", date_text(day)));
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

/// The text of a span that a quote must be in.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QuotedSpan<'a> {
    /// The stored text of the span, or its text in a packed context.
    pub text: &'a str,
    /// `valid_from_ms` of the span when `text` is the stored text. `None` for a packed text,
    /// which shows the dates of its day words itself.
    pub valid_from_ms: Option<i64>,
}

/// Whether `quote` is in the text of `span` when the dates that `date_day_words` adds are left
/// out. A quote copied from a packed context then matches the stored span, and the reverse.
///
/// A date in the quote must be the date that the text shows after the same day word. Where the
/// text shows none, the date must be one that `date_day_words` adds for the time of the span
/// in some time zone. The quote keeps every other date in brackets as text.
pub(crate) fn quote_matches(span: QuotedSpan<'_>, quote: &str) -> bool {
    if quote.trim().is_empty() {
        return false;
    }
    if span.text.contains(quote) {
        return true;
    }
    let (text, quote) = (Undated::of(span.text), Undated::of(quote));
    if quote.text.trim().is_empty() {
        return false;
    }
    // The local day of a time is the UTC day, the day before, or the day after.
    let said_on_days = span.valid_from_ms.map(|at_ms| {
        let day = at_ms.div_euclid(MS_PER_DAY);
        [day - 1, day, day + 1]
    });
    let date_is_right = |start: usize, (at, date): &(usize, String)| {
        let at = start + at;
        if let Some((_, shown)) = text.dates.iter().find(|(shown_at, _)| *shown_at == at) {
            return shown == date;
        }
        let days = said_on_days.iter().flatten();
        days.filter_map(|said_on| text.day_of_word_ending(at, *said_on))
            .any(|day| date_text(day) == *date)
    };
    text.text.char_indices().any(|(start, _)| {
        text.text[start..].starts_with(&quote.text)
            && quote.dates.iter().all(|date| date_is_right(start, date))
    })
}

/// A text without the dates that follow the day words that `text_with_dates` dates.
struct Undated {
    text: String,
    /// The lowercase words of the text, each with the byte position of its end in `text`.
    words: Vec<(String, usize)>,
    /// The dates left out, each with the byte position in `text` of the end of its day word.
    dates: Vec<(usize, String)>,
}

impl Undated {
    fn of(dated: &str) -> Self {
        const TAG_LEN: usize = " [YYYY-MM-DD]".len();
        let is_date = |tag: &str| {
            tag.starts_with(" [")
                && tag.ends_with(']')
                && tag[2..TAG_LEN - 1].bytes().enumerate().all(|(i, byte)| {
                    if i == 4 || i == 7 {
                        byte == b'-'
                    } else {
                        byte.is_ascii_digit()
                    }
                })
        };
        let bounds = ascii_words(dated);
        let lower = bounds
            .iter()
            .map(|&(start, end)| dated[start..end].to_ascii_lowercase())
            .collect::<Vec<_>>();
        let mut undated = Self {
            text: String::with_capacity(dated.len()),
            words: Vec::with_capacity(bounds.len()),
            dates: Vec::new(),
        };
        let mut copied = 0usize;
        for (index, &(_, end)) in bounds.iter().enumerate() {
            undated.text.push_str(&dated[copied..end]);
            copied = end;
            undated
                .words
                .push((lower[index].clone(), undated.text.len()));
            let tag = dated.get(end..end + TAG_LEN).filter(|tag| is_date(tag));
            let previous = index.checked_sub(1).map(|i| lower[i].as_str());
            let next = lower.get(index + 1).map(String::as_str);
            // "last Friday [2025-01-01]" gets no date from `text_with_dates`, so that date is text.
            if let Some(tag) =
                tag.filter(|_| day_of_word(&lower[index], previous, next, 0).is_some())
            {
                let date = tag[2..TAG_LEN - 1].to_string();
                undated.dates.push((undated.text.len(), date));
                copied = end + TAG_LEN;
            }
        }
        undated.text.push_str(&dated[copied..]);
        undated
    }

    /// The day that `text_with_dates` gives the word that ends at byte `end`, if it dates it.
    fn day_of_word_ending(&self, end: usize, said_on: i64) -> Option<i64> {
        let index = self
            .words
            .iter()
            .position(|(_, word_end)| *word_end == end)?;
        let word = |i: usize| self.words.get(i).map(|(word, _)| word.as_str());
        let previous = index.checked_sub(1).and_then(word);
        day_of_word(&self.words[index].0, previous, word(index + 1), said_on)
    }
}

/// A count of days since 1970-01-01 as `YYYY-MM-DD`.
fn date_text(days: i64) -> String {
    let (year, month, day) = civil_date(days);
    format!("{year:04}-{month:02}-{day:02}")
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

    /// The stored text of a span said at noon UTC on the Tuesday.
    fn stored(text: &str) -> QuotedSpan<'_> {
        QuotedSpan {
            text,
            valid_from_ms: Some(TUESDAY * MS_PER_DAY + MS_PER_DAY / 2),
        }
    }

    /// The text of a span as a packed context shows it.
    fn packed(text: &str) -> QuotedSpan<'_> {
        QuotedSpan {
            text,
            valid_from_ms: None,
        }
    }

    #[test]
    fn quote_matches_with_or_without_the_added_dates() {
        let reply = "Reply by Friday.";
        let dated = "Reply by Friday [2025-07-18].";
        assert!(quote_matches(stored(reply), "by Friday [2025-07-18]."));
        assert!(quote_matches(packed(dated), "by Friday."));
        assert!(quote_matches(packed(dated), "by Friday [2025-07-18]."));
        assert!(!quote_matches(stored(reply), "by Monday."));
        assert!(!quote_matches(stored(reply), " [2025-07-18]"));
        // A date that the text itself holds is not a day word date, so the quote must repeat it.
        let release = "Released [2025-07-18].";
        assert!(quote_matches(stored(release), "Released [2025-07-18]."));
        assert!(!quote_matches(stored(release), "Released [2025-01-01]."));
        assert!(!quote_matches(stored("Due [2025-7-18]."), "Due."));
    }

    #[test]
    fn quote_keeps_a_date_after_a_day_word_that_gets_no_date() {
        let slipped = "It slipped last Friday [2025-01-01] again.";
        assert!(quote_matches(
            stored(slipped),
            "last Friday [2025-01-01] again."
        ));
        assert!(quote_matches(stored(slipped), "Friday [2025-01-01] again."));
        assert!(!quote_matches(stored(slipped), "last Friday again."));
        let compound = "Post it by EOD [2025-07-15] Friday.";
        assert!(!quote_matches(stored(compound), "by EOD Friday."));
        // The packer adds no date after "last Friday", so a quote cannot show one there.
        let undated = "It slipped last Friday again.";
        assert!(!quote_matches(
            stored(undated),
            "Friday [2025-07-18] again."
        ));
    }

    #[test]
    fn quote_with_a_date_that_the_day_word_does_not_name_does_not_match() {
        assert!(!quote_matches(
            stored("Due Friday."),
            "Due Friday [1999-01-01]."
        ));
        // The date of a packed text is the one the reader saw.
        let dated = packed("Due Friday [2025-07-18].");
        assert!(!quote_matches(dated, "Due Friday [1999-01-01]."));
        assert!(!quote_matches(
            packed("Due Friday."),
            "Due Friday [2025-07-18]."
        ));
        // A span without a time gets no dates, so a date in the quote is not from the packer.
        let timeless = QuotedSpan {
            text: "Due Friday.",
            valid_from_ms: None,
        };
        assert!(!quote_matches(timeless, "Due Friday [2025-07-18]."));
    }

    #[test]
    fn quote_date_can_be_the_date_in_any_time_zone_of_the_span_time() {
        // 01:00 UTC on Tuesday is Monday evening at UTC-7, so "today" is one of two dates.
        let early = QuotedSpan {
            text: "Ship today.",
            valid_from_ms: Some(TUESDAY * MS_PER_DAY + 3_600_000),
        };
        assert!(quote_matches(early, "Ship today [2025-07-15]."));
        assert!(quote_matches(early, "Ship today [2025-07-14]."));
        assert!(!quote_matches(early, "Ship today [2025-07-12]."));
    }
}
