//! Port of the model's `tts_norm.normalize`: expand numbers, dates, times, money, abbreviations,
//! units and symbols into their spoken form, keeping . , ! ? ; and dashes for prosody.
use regex::Regex;

use crate::num2words as n2w;

const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
];

fn units_map() -> Vec<(&'static str, &'static str)> {
    vec![
        ("km", "kilometres"), ("mi", "miles"), ("cm", "centimetres"), ("mm", "millimetres"),
        ("kg", "kilograms"), ("lb", "pounds"), ("lbs", "pounds"), ("oz", "ounces"),
        ("ml", "millilitres"), ("l", "litres"), ("ft", "feet"), ("in", "inches"),
        ("hr", "hours"), ("hrs", "hours"), ("min", "minutes"), ("mins", "minutes"),
        ("sec", "seconds"), ("secs", "seconds"), ("ms", "milliseconds"),
        ("gb", "gigabytes"), ("mb", "megabytes"), ("kb", "kilobytes"), ("tb", "terabytes"),
        ("hz", "hertz"), ("khz", "kilohertz"), ("mhz", "megahertz"), ("ghz", "gigahertz"),
        ("kw", "kilowatts"), ("mph", "miles per hour"), ("kph", "kilometres per hour"),
    ]
}

fn mag(k: &str) -> Option<&'static str> {
    match k {
        "k" => Some("thousand"),
        "m" => Some("million"),
        "b" => Some("billion"),
        "t" => Some("trillion"),
        _ => None,
    }
}

fn abbrev_map() -> Vec<(&'static str, &'static str)> {
    vec![
        ("mr", "Mister"), ("mrs", "Missus"), ("ms", "Miss"), ("dr", "Doctor"),
        ("prof", "Professor"), ("rev", "Reverend"), ("hon", "Honourable"), ("ext", "extension"),
        ("approx", "approximately"), ("vs", "versus"), ("est", "estimated"), ("dept", "department"),
        ("univ", "university"), ("fig", "figure"), ("eq", "equation"), ("ref", "reference"),
        ("vol", "volume"), ("incl", "including"), ("excl", "excluding"), ("min", "minimum"),
        ("max", "maximum"), ("avg", "average"), ("mt", "mount"), ("ft", "fort"),
        ("appt", "appointment"), ("pp", "pages"), ("jr", "Junior"), ("sr", "Senior"),
        ("inc", "Incorporated"), ("ltd", "Limited"), ("corp", "Corporation"), ("mins", "minutes"),
        ("hrs", "hours"), ("wks", "weeks"), ("yrs", "years"),
    ]
}

fn days_map() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Mon", "Monday"), ("Tue", "Tuesday"), ("Tues", "Tuesday"), ("Wed", "Wednesday"),
        ("Weds", "Wednesday"), ("Thu", "Thursday"), ("Thur", "Thursday"), ("Thurs", "Thursday"),
        ("Fri", "Friday"), ("Sat", "Saturday"), ("Sun", "Sunday"),
    ]
}

fn months_abbr() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Jan", "January"), ("Feb", "February"), ("Mar", "March"), ("Apr", "April"), ("Jun", "June"),
        ("Jul", "July"), ("Aug", "August"), ("Sep", "September"), ("Sept", "September"),
        ("Oct", "October"), ("Nov", "November"), ("Dec", "December"),
    ]
}

fn roman_map() -> Vec<(&'static str, i64)> {
    vec![
        ("i", 1), ("ii", 2), ("iii", 3), ("iv", 4), ("v", 5), ("vi", 6), ("vii", 7), ("viii", 8),
        ("ix", 9), ("x", 10), ("xi", 11), ("xii", 12),
    ]
}

const ROMAN_CTX: &str = "chapter|section|part|volume|book|act|phase|appendix|world war|type|class|figure|table|step|stage";

/// re.sub with a closure over the whole match.
fn sub<F: Fn(&str, &regex::Captures, usize) -> String>(re: &Regex, text: &str, f: F) -> String {
    let mut out = String::new();
    let mut last = 0usize;
    for c in re.captures_iter(text) {
        let m = match c.get(0) {
            Some(m) => m,
            None => continue,
        };
        out.push_str(&text[last..m.start()]);
        out.push_str(&f(m.as_str(), &c, m.end()));
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}
/// re.sub with a context veto: `pred(whole, start, end)` must be true for the match to be replaced
/// (the regex crate has no look-around in UTF-8 mode, so those assertions live here).
fn sub_veto<F, P>(re: &Regex, text: &str, f: F, pred: P) -> String
where
    F: Fn(&str, &regex::Captures, usize) -> String,
    P: Fn(&str, usize, usize) -> bool,
{
    let mut out = String::new();
    let mut last = 0usize;
    for c in re.captures_iter(text) {
        let m = match c.get(0) {
            Some(m) => m,
            None => continue,
        };
        out.push_str(&text[last..m.start()]);
        if pred(m.as_str(), m.start(), m.end()) {
            out.push_str(&f(m.as_str(), &c, m.end()));
        } else {
            out.push_str(m.as_str());
        }
        last = m.end();
    }
    out.push_str(&text[last..]);
    out
}

fn char_at(s: &str, i: usize) -> Option<char> {
    s[i..].chars().next()
}

fn first_after(s: &str, mut i: usize) -> Option<char> {
    while i < s.len() {
        let c = s[i..].chars().next()?;
        if !c.is_whitespace() {
            return Some(c);
        }
        i += c.len_utf8();
    }
    None
}


fn grp(c: &regex::Captures, i: usize) -> String {
    c.get(i).map(|m| m.as_str().to_string()).unwrap_or_default()
}

fn parse_i64(s: &str) -> Option<i64> {
    s.replace(',', "").parse::<i64>().ok()
}

fn card(n: i64) -> String {
    n2w::cardinal(n)
}

fn year(y: i64) -> String {
    if (1100..=1999).contains(&y) || (2000..=2099).contains(&y) {
        n2w::year(y)
    } else {
        card(y)
    }
}

fn decimal(tok: &str) -> String {
    let tok = tok.replace(',', "");
    if !tok.chars().any(|c| c.is_ascii_digit()) {
        return tok;
    }
    if let Some((a, b)) = tok.split_once('.') {
        let head = a.parse::<i64>().unwrap_or(0);
        let digits: String = b
            .chars()
            .filter(|c| c.is_ascii_digit())
            .map(|d| card(d as i64 - '0' as i64))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} point {}", card(head), digits)
    } else {
        card(tok.parse::<i64>().unwrap_or(0))
    }
}

/// "3:45pm." must not swallow the sentence-final period.
fn keep_final_dot(matched: &str, rest: &str, out: String) -> String {
    if !matched.ends_with('.') {
        return out;
    }
    let rest_trimmed = rest.trim();
    let starts_space = rest.starts_with(char::is_whitespace);
    let first_upper = rest_trimmed.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    if rest_trimmed.is_empty() || (starts_space && first_upper) {
        return format!("{out}.");
    }
    out
}

fn dates(s: &str) -> String {
    let s = sub(
        &Regex::new(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b").unwrap(),
        s,
        |whole, c, _end| {
            let (y, mo, d) = (
                parse_i64(&grp(c, 1)).unwrap_or(0),
                parse_i64(&grp(c, 2)).unwrap_or(0),
                parse_i64(&grp(c, 3)).unwrap_or(0),
            );
            if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
                return whole.to_string();
            }
            format!("{} {}, {}", MONTHS[(mo - 1) as usize], n2w::ordinal(d), year(y))
        },
    );
    let s = sub(
        &Regex::new(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b").unwrap(),
        &s,
        |whole, c, _end| {
            let (a, b, mut y) = (
                parse_i64(&grp(c, 1)).unwrap_or(0),
                parse_i64(&grp(c, 2)).unwrap_or(0),
                parse_i64(&grp(c, 3)).unwrap_or(0),
            );
            if !(1..=12).contains(&a) || !(1..=31).contains(&b) {
                return whole.to_string();
            }
            if y < 100 {
                y += if y < 50 { 2000 } else { 1900 };
            }
            format!("{} {}, {}", MONTHS[(a - 1) as usize], n2w::ordinal(b), year(y))
        },
    );
    sub(
        &Regex::new(r"(?i)\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b").unwrap(),
        &s,
        |_whole, c, _end| {
            let name = grp(c, 1);
            format!("{} {}, {}", format!("{}{}", &name[..1].to_uppercase(), &name[1..].to_lowercase()),
                n2w::ordinal(parse_i64(&grp(c, 2)).unwrap_or(0)),
                year(parse_i64(&grp(c, 3)).unwrap_or(0)))
        },
    )
}

fn times(s: &str) -> String {
    sub(
        &Regex::new(r"(?i)\b(\d{1,2}):(\d{2})(?:\s*([ap]\.?m\.?))?").unwrap(),
        s,
        |whole, c, end| {
            let (mut h, mi, ampm) = (
                parse_i64(&grp(c, 1)).unwrap_or(0),
                parse_i64(&grp(c, 2)).unwrap_or(0),
                grp(c, 3).to_lowercase(),
            );
            if h > 23 || mi > 59 {
                return whole.to_string();
            }
            let mut suffix = String::new();
            if !ampm.is_empty() {
                suffix = if ampm.starts_with('a') { " AM" } else { " PM" }.into();
            } else if h == 0 {
                h = 12;
                suffix = " AM".into();
            }
            let has = !suffix.is_empty();
            let spoken = if mi == 0 {
                if !has {
                    format!("{} o'clock", card(if h <= 12 { h } else { h }))
                } else {
                    card(if h <= 12 { h } else { h - 12 })
                }
            } else if mi < 10 {
                format!("{} oh {}", card(if h <= 12 || !has { h } else { h - 12 }), card(mi))
            } else {
                format!("{} {}", card(if h <= 12 || !has { h } else { h - 12 }), card(mi))
            };
            keep_final_dot(whole, &s[end..], format!("{spoken}{suffix}"))
        },
    )
}

fn bare_hour(s: &str) -> String {
    sub_veto(
        &Regex::new(r"(?i)\b(\d{1,2})\s*([ap])\.?m\.?").unwrap(),
        s,
        |whole, c, end| {
            let h = parse_i64(&grp(c, 1)).unwrap_or(0);
            if !(1..=12).contains(&h) {
                return whole.to_string();
            }
            let am = grp(c, 2).to_lowercase().starts_with('a');
            keep_final_dot(
                whole,
                &s[end..],
                format!("{} {}", card(h), if am { "AM" } else { "PM" }),
            )
        },
        |_w, _start, end| !matches!(char_at(s, end), Some(c) if c.is_ascii_alphabetic()),
    )
}

fn phones(s: &str) -> String {
    sub(
        &Regex::new(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]?\d{0,4}\b").unwrap(),
        s,
        |whole, _c, _end| {
            let digits: String = whole.chars().filter(|c| c.is_ascii_digit()).collect();
            if !(7..=11).contains(&digits.len()) {
                return whole.to_string();
            }
            let mut groups: Vec<String> = Vec::new();
            let mut i = 0usize;
            if digits.len() == 11 && digits.starts_with('1') {
                groups.push("one".to_string());
                i = 1;
            }
            let d: Vec<char> = digits.chars().collect();
            for (a, b) in [(i, i + 3), (i + 3, i + 6), (i + 6, d.len())] {
                if a < d.len() {
                    let g: Vec<String> = d[a..b.min(d.len())]
                        .iter()
                        .map(|c| card(*c as i64 - '0' as i64))
                        .collect();
                    groups.push(g.join(" "));
                }
            }
            groups.into_iter().filter(|g| !g.is_empty()).collect::<Vec<_>>().join(", ")
        },
    )
}

fn money(s: &str) -> String {
    sub(
        &Regex::new(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s?([KkMmBbTt])\b)?").unwrap(),
        s,
        |_whole, c, _end| {
            let amt = grp(c, 1).replace(',', "");
            let suf = grp(c, 2).to_lowercase();
            if !suf.is_empty() {
                let num: f64 = amt.parse().unwrap_or(0.0);
                let whole_word = if num.fract() == 0.0 {
                    card(num as i64)
                } else {
                    let i = num as i64;
                    let frac = amt.split('.').nth(1).unwrap_or("");
                    let digits: Vec<String> = frac
                        .chars()
                        .filter(|d| d.is_ascii_digit())
                        .map(|d| card(d as i64 - '0' as i64))
                        .collect();
                    format!("{} point {}", card(i), digits.join(" "))
                };
                return format!(
                    "{} {} dollars",
                    whole_word,
                    mag(&suf).unwrap_or("dollars")
                );
            }
            if let Some((d, c2)) = amt.split_once('.') {
                let cents = format!("{}00", c2);
                let cents: String = cents.chars().filter(|x| x.is_ascii_digit()).collect();
                let cents = &cents[..2.min(cents.len())];
                let cd = cents.parse::<i64>().unwrap_or(0);
                let dv = d.parse::<i64>().unwrap_or(0);
                let dollars = format!("{} dollar{}", card(dv), if dv == 1 { "" } else { "s" });
                if cd != 0 {
                    format!("{} and {} cent{}", dollars, card(cd), if cd == 1 { "" } else { "s" })
                } else {
                    dollars
                }
            } else {
                let dv = amt.parse::<i64>().unwrap_or(0);
                format!("{} dollar{}", card(dv), if dv == 1 { "" } else { "s" })
            }
        },
    )
}

fn magnitudes(s: &str) -> String {
    sub(
        &Regex::new(r"\b(\d[\d,]*(?:\.\d+)?)\s*([KMBT])\b").unwrap(),
        s,
        |whole, c, _end| {
            let v = grp(c, 1);
            let m = grp(c, 2).to_lowercase();
            match mag(&m) {
                Some(w) => format!("{} {}", decimal(&v), w),
                None => whole.to_string(),
            }
        },
    )
}

fn fractions(s: &str) -> String {
    sub(
        &Regex::new(r"\b(\d{1,3})/(\d{1,3})\b").unwrap(),
        s,
        |whole, c, _end| {
            let (a, b) = (
                parse_i64(&grp(c, 1)).unwrap_or(0),
                parse_i64(&grp(c, 2)).unwrap_or(0),
            );
            match (a, b) {
                (1, 2) => "a half".to_string(),
                (1, 3) => "a third".to_string(),
                (2, 3) => "two thirds".to_string(),
                (1, 4) => "a quarter".to_string(),
                (3, 4) => "three quarters".to_string(),
                (1, 8) => "an eighth".to_string(),
                _ if b == 0 => whole.to_string(),
                _ => {
                    let o = n2w::ordinal(b);
                    format!("{} {}{}", card(a), o, if a != 1 { "s" } else { "" })
                }
            }
        },
    )
}

fn versions(s: &str) -> String {
    sub(
        &Regex::new(r"\b(\d+)\.(\d+)\.(\d+)\b").unwrap(),
        s,
        |_w, c, _end| {
            let v: Vec<String> = [1, 2, 3].iter().map(|i| card(parse_i64(&grp(c, *i)).unwrap_or(0))).collect();
            v.join(" point ")
        },
    )
}

fn unit_words() -> Vec<(&'static str, &'static str)> {
    let mut u = units_map();
    u.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    u
}

fn units(s: &str) -> String {
    let pat = format!(
        r"(?i)\b(\d[\d,]*(?:\.\d+)?)\s?({})\b",
        unit_words().iter().map(|(k, _)| *k).collect::<Vec<_>>().join("|")
    );
    let s = sub(&Regex::new(&pat).unwrap(), s, |_w, c, _end| {
        let n = grp(c, 1);
        let key = grp(c, 2).to_lowercase();
        let val = decimal(&n);
        let word = units_map().into_iter().find(|(k, _)| *k == key).map(|(_, v)| v).unwrap_or("units");
        let word = if n.replace(',', "") == "1" || n.replace(',', "") == "1.0" {
            word.strip_suffix('s').unwrap_or(word)
        } else {
            word
        };
        format!("{val} {word}")
    });
    sub(
        &Regex::new(r"([-−]?[\d.]+)\s?°\s?([CF])\b").unwrap(),
        &s,
        |_w, c, _end| {
            let raw = grp(c, 1);
            let unit = grp(c, 2).to_uppercase();
            let neg = raw.starts_with('-') || raw.starts_with('−');
            let val = decimal(&raw[if neg { 1.. } else { 0.. }]);
            let scale = if unit == "C" { "Celsius" } else { "Fahrenheit" };
            format!("{}{} degrees {}", if neg { "minus " } else { "" }, val, scale)
        },
    )
}

fn abbrev(s: &str) -> String {
    let s = sub_veto(
        &Regex::new(r"\bSt\.\s+").unwrap(),
        s,
        |_w, _c, _end| "Saint ".to_string(),
        |_w, _start, end| matches!(char_at(s, end), Some(c) if c.is_uppercase()),
    );
    let s = sub_veto(
        &Regex::new(r"\bSt\.").unwrap(),
        &s,
        |_w, _c, _end| "Street.".to_string(),
        |_w, _start, end| s[end..].trim().is_empty(),
    );
    let s = sub_veto(
        &Regex::new(r"\bSt\.").unwrap(),
        &s,
        |_w, _c, _end| "Street.".to_string(),
        |_w, _start, end| matches!(first_after(&s, end), Some(c) if c == '"' || c == '\'' || c == ')'),
    );
    let s = sub_veto(
        &Regex::new(r"\bSt\.").unwrap(),
        &s,
        |_w, _c, _end| "Street".to_string(),
        |_w, _start, end| !(matches!(first_after(&s, end), Some(c) if c.is_uppercase())),
    );
    let s = sub(&Regex::new(r"\bAve\.").unwrap(), &s, |_w, _c, _end| "Avenue".to_string());
    let s = sub(&Regex::new(r"\bRd\.").unwrap(), &s, |_w, _c, _end| "Road".to_string());
    let s = sub(&Regex::new(r"\bBlvd\.").unwrap(), &s, |_w, _c, _end| "Boulevard".to_string());
    let s = sub_veto(
        &Regex::new(r"\bNo\.\s*").unwrap(),
        &s,
        |_w, _c, _end| "number ".to_string(),
        |_w, _start, end| matches!(char_at(&s, end), Some(c) if c.is_ascii_digit()),
    );
    let s = sub_veto(
        &Regex::new(r"\bNos\.\s*").unwrap(),
        &s,
        |_w, _c, _end| "numbers ".to_string(),
        |_w, _start, end| matches!(char_at(&s, end), Some(c) if c.is_ascii_digit()),
    );
    let mut s = s;
    for (k, v) in days_map().into_iter().chain(months_abbr()) {
        s = sub(&Regex::new(&format!(r"\b{k}\.")).unwrap(), &s, |_w, _c, _end| v.to_string());
    }
    for (k, v) in abbrev_map() {
        s = sub(&Regex::new(&format!(r"(?i)\b{k}\.")).unwrap(), &s, |_w, _c, _end| v.to_string());
    }
    s
}

fn romans(s: &str) -> String {
    let pat = format!(r"(?i)\b({ROMAN_CTX})\s+([IVX]{{1,4}})\b");
    sub(&Regex::new(&pat).unwrap(), s, |whole, c, _end| {
        let r = grp(c, 2).to_lowercase();
        match roman_map().into_iter().find(|(k, _)| *k == r) {
            Some((_, v)) => format!("{} {}", grp(c, 1), card(v)),
            None => whole.to_string(),
        }
    })
}

fn ordinals(s: &str) -> String {
    sub(
        &Regex::new(r"(?i)\b(\d+)(?:st|nd|rd|th)\b").unwrap(),
        s,
        |_w, c, _end| n2w::ordinal(parse_i64(&grp(c, 1)).unwrap_or(0)),
    )
}

fn ranges(s: &str) -> String {
    sub(
        &Regex::new(r"\b(\d+)\s?[-\u{2013}\u{2014}]\s?(\d+)\b").unwrap(),
        s,
        |_w, c, _end| {
            format!(
                "{} to {}",
                card(parse_i64(&grp(c, 1)).unwrap_or(0)),
                card(parse_i64(&grp(c, 2)).unwrap_or(0))
            )
        },
    )
}

fn numbers(s: &str) -> String {
    let s = sub_veto(
        &Regex::new(r"[-−](\d)").unwrap(),
        s,
        |_w, c, _end| format!("minus {}", grp(c, 1)),
        |_w, start, _end| !matches!(s[..start].chars().next_back(), Some(c) if c.is_alphanumeric() || c == '_' || c == '.'),
    );
    sub(
        &Regex::new(r"\b\d[\d,]*(?:\.\d+)?\b").unwrap(),
        &s,
        |whole, _c, _end| {
            let tok = whole;
            if !tok.contains(',') && tok.len() == 4 && tok.chars().all(|c| c.is_ascii_digit()) {
                let v: i64 = tok.parse().unwrap_or(0);
                if (1100..=2099).contains(&v) {
                    return year(v);
                }
            }
            decimal(tok)
        },
    )
}

/// Mini-NFKC: the compatibility decompositions that show up in prose.
fn nfkc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let rep: &str = match c {
            '\u{fb00}' => "ff",
            '\u{fb01}' => "fi",
            '\u{fb02}' => "fl",
            '\u{fb03}' => "ffi",
            '\u{fb04}' => "ffl",
            '\u{bd}' => " 1/2 ",
            '\u{bc}' => " 1/4 ",
            '\u{be}' => " 3/4 ",
            '\u{2122}' => "TM",
            '\u{00ae}' => "(R)",
            '\u{2160}'..='\u{217f}' => {
                // Roman numerals I..XII (upper) and i..xii (lower)
                let idx = c as u32 - 0x2160;
                let tab = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
                    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"];
                tab.get(idx as usize).copied().unwrap_or("?")
            }
            '\u{2460}'..='\u{2473}' => {
                let idx = c as u32 - 0x2460;
                let tab = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                    "15", "16", "17", "18", "19", "20"];
                tab.get(idx as usize).copied().unwrap_or("?")
            }
            _ => {
                // fullwidth latin + digits -> ascii
                let cp = c as u32;
                if (0xff10..=0xff19).contains(&cp) || (0xff21..=0xff3a).contains(&cp) || (0xff41..=0xff5a).contains(&cp) {
                    let a = (cp - 0xfee0) as u8;
                    out.push(a as char);
                    continue;
                }
                " "
            }
        };
        if rep == " " {
            out.push(c);
        } else {
            out.push_str(rep);
        }
    }
    out
}

pub fn normalize(text: &str) -> String {
    let s = nfkc(text);
    let mut s: String = s
        .chars()
        .map(|c| match c {
            '\u{2019}' | '\u{2018}' => '\'',
            '\u{201c}' | '\u{201d}' => '"',
            _ => c,
        })
        .collect();
    s = s.replace('\u{2014}', " - ").replace('\u{2013}', "-").replace('\u{2026}', "...");
    s = sub(&Regex::new(r"https?://\S+|www\.\S+").unwrap(), &s, |_w, _c, _end| " link ".to_string());
    s = sub(
        &Regex::new(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b").unwrap(),
        &s,
        |_w, _c, _end| " email address ".to_string(),
    );
    s = sub(&Regex::new(r"(?i)\be\.\s?g\.").unwrap(), &s, |_w, _c, _end| "for example,".to_string());
    s = sub(&Regex::new(r"(?i)\bi\.\s?e\.").unwrap(), &s, |_w, _c, _end| "that is,".to_string());
    s = sub(&Regex::new(r"(?i)\betc\.").unwrap(), &s, |_w, _c, _end| "etcetera".to_string());
    s = sub(&Regex::new(r"(?i)\bvs\.?\b").unwrap(), &s, |_w, _c, _end| "versus".to_string());
    let s = dates(&s);
    let s = times(&s);
    let s = bare_hour(&s);
    let s = phones(&s);
    let s = money(&s);
    let s = versions(&s);
    let s = fractions(&s);
    let s = magnitudes(&s);
    let s = units(&s);
    let s = abbrev(&s);
    let s = romans(&s);
    let s = ordinals(&s);
    let s = ranges(&s);
    let mut s = s;
    for (k, v) in [
        ("%", " percent "), ("&", " and "), ("@", " at "), ("+", " plus "), ("=", " equals "),
        ("\u{d7}", " times "), ("\u{f7}", " divided by "), ("<", " less than "),
        (">", " greater than "), ("~", " about "), ("\u{b0}", " degrees "), ("€", " euros "),
        ("£", " pounds "), ("¥", " yen "), ("©", " copyright "), ("™", " trademark "),
        ("§", " section "),
    ] {
        s = s.replace(k, v);
    }
    let s = numbers(&s);
    let s = s.replace(':', ",");
    let s = sub(&Regex::new(r"[_*`^|\\/\[\]{}<>]").unwrap(), &s, |_w, _c, _end| " ".to_string());
    let s = sub(&Regex::new(r"\s+([.,!?;])").unwrap(), &s, |_w, c, _end| {
        format!("{}", grp(c, 1))
    });
    let s = sub_veto(
        &Regex::new(r"([.,!?;])").unwrap(),
        &s,
        |_w, c, _end| format!("{} ", grp(c, 1)),
        |_w, _start, end| {
            !matches!(char_at(&s, end), Some(c) if c.is_whitespace() || c == '"' || c == '\'' || c == ',' || c == '.' || c == '!' || c == '?' || c == ';')
        },
    );
    let s = sub(&Regex::new(r"\s{2,}").unwrap(), &s, |_w, _c, _end| " ".to_string());
    s.trim().to_string()
}
