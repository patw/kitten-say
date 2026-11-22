//! English number -> words, matching num2words(EN) output: "one hundred and fifty",
//! "one thousand, two hundred and thirty-four", hyphen between tens and units.

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// n < 1000, the "and" style the library uses for a remainder under a hundred.
fn under1000(n: i64) -> String {
    if n < 100 {
        return under100(n);
    }
    let (h, r) = ((n / 100) as usize, n % 100);
    let head = format!("{} hundred", ONES[h]);
    if r == 0 {
        head
    } else {
        format!("{head} and {}", under100(r))
    }
}

fn under100(n: i64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    let (t, u) = ((n / 10) as usize, (n % 10) as usize);
    if u == 0 {
        TENS[t].to_string()
    } else {
        format!("{}-{}", TENS[t], ONES[u])
    }
}

pub fn cardinal(n: i64) -> String {
    if n < 0 {
        return format!("minus {}", cardinal(-n));
    }
    if n < 1000 {
        return under1000(n);
    }
    let scales: [(i64, &str); 4] = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    let mut parts: Vec<String> = Vec::new();
    let mut rest = n;
    for (div, name) in scales {
        if rest >= div {
            let head = rest / div;
            rest %= div;
            parts.push(format!("{} {}", under1000(head), name));
        }
    }
    if rest == 0 {
        parts.join(" ")
    } else if rest < 100 {
        format!("{} and {}", parts.join(", "), under100(rest))
    } else {
        format!("{}, {}", parts.join(", "), under1000(rest))
    }
}

/// num2words(n, to="ordinal"): the inflection lands on the last token (space- or hyphen-separated).
pub fn ordinal(n: i64) -> String {
    if n < 0 {
        return format!("minus {}", ordinal(-n));
    }
    let s = cardinal(n);
    let mut idx = 0usize;
    for (i, c) in s.char_indices() {
        if c == ' ' || c == '-' {
            idx = i + c.len_utf8();
        }
    }
    let (head, last) = (&s[..idx], &s[idx..]);
    format!("{}{}", head, inflect(last))
}

fn inflect(w: &str) -> String {
    match w {
        "one" => "first",
        "two" => "second",
        "three" => "third",
        "four" => "fourth",
        "five" => "fifth",
        "eight" => "eighth",
        "nine" => "ninth",
        "twelve" => "twelfth",
        _ => {
            if let Some(stem) = w.strip_suffix('y') {
                return format!("{stem}ieth");
            }
            return format!("{w}th");
        }
    }
    .to_string()
}

/// num2words(y, to="year").
pub fn year(y: i64) -> String {
    match y {
        0..=99 => cardinal(y),
        100..=999 => {
            let (h, t) = (y / 100, y % 100);
            if t == 0 {
                format!("{} hundred", ONES[h as usize])
            } else if t < 10 {
                format!("{} oh-{}", ONES[h as usize], ONES[t as usize])
            } else {
                format!("{} {}", cardinal(h), cardinal(t))
            }
        }
        1000 => "one thousand".to_string(),
        1001..=1009 => format!("one thousand and {}", ONES[(y % 10) as usize]),
        2000 => "two thousand".to_string(),
        2001..=2009 => format!("two thousand and {}", ONES[(y % 10) as usize]),
        1010..=1999 => {
            let (h, t) = (y / 100, y % 100);
            if t == 0 {
                format!("{} hundred", cardinal(h))
            } else if t < 10 {
                format!("{} oh-{}", cardinal(h), ONES[t as usize])
            } else {
                format!("{} {}", cardinal(h), cardinal(t))
            }
        }
        2010..=2099 => {
            let t = y % 100;
            format!("twenty {}", cardinal(t))
        }
        _ => {
            let (h, t) = (y / 100, y % 100);
            if t == 0 {
                format!("{} hundred", cardinal(h))
            } else if t < 10 {
                format!("{} oh-{}", cardinal(h), ONES[t as usize])
            } else {
                format!("{} {}", cardinal(h), cardinal(t))
            }
        }
    }
}
