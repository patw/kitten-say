//! English front end: normalize -> lexicon (misaki gold/silver, then CMU->misaki) -> stem rules ->
//! letter spelling -> BART for what is left; then the training-time FOLD and vocab filter.
use crate::assets::Lexicon;
use crate::assets::Vocab;
use crate::bart::Bart;
use crate::norm::normalize;

pub const MAX_PHON: usize = 350;

/// The lexicons spell the flap and the glottal stop as ɾ / ʔ; what the model was trained on writes
/// them as the letters T and t, which is the substitution misaki's lexicon path applies.
fn lex_icon(ps: &str) -> String {
    ps.replace('\u{27e}', "T").replace('\u{294}', "t")
}
const STRESSES: [char; 2] = ['\u{2cc}', '\u{2c8}']; // ˌ ˈ
const VOWELS: &str = "AIOQWYaiu\u{e6}\u{251}\u{252}\u{254}\u{259}\u{25b}\u{25c}\u{26a}\u{28a}\u{28c}\u{1d7b}";
const PUNCTS: &str = ";,.!?\u{2014}\u{2026}\"\u{201c}\u{201d}";
const LETTER_TRANSLIT: &[(char, &str); 12] = &[
    ('\u{f8}', "o"), ('\u{d8}', "O"), ('\u{df}', "ss"), ('\u{e6}', "ae"), ('\u{c6}', "AE"),
    ('\u{153}', "oe"), ('\u{152}', "OE"), ('\u{142}', "l"), ('\u{141}', "L"), ('\u{111}', "d"),
    ('\u{110}', "D"), ('\u{2019}', "'"),
];

fn is_vowel_start(ps: &str) -> bool {
    match ps.chars().next() {
        Some(c) => VOWELS.contains(c),
        None => false,
    }
}

/// misaki.en.apply_stress for the stress levels the G2P uses.
pub fn apply_stress(ps: &str, stress: Option<f32>) -> String {
    let stress = match stress {
        None => return ps.to_string(),
        Some(s) => s,
    };
    let has_primary = ps.contains(STRESSES[1]);
    let has_secondary = ps.contains(STRESSES[0]);
    if stress < -1.0 {
        return ps.replace(STRESSES[1], "").replace(STRESSES[0], "");
    }
    if (stress == -1.0) || ((stress == 0.0 || stress == -0.5) && has_primary) {
        return ps.replace(STRESSES[0], "").replace(STRESSES[1], &STRESSES[0].to_string());
    }
    if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_primary && !has_secondary {
        let has_vowel = ps.chars().any(|c| VOWELS.contains(c));
        if !has_vowel {
            return ps.to_string();
        }
        return restress(&format!("{}{}", STRESSES[0], ps));
    }
    if stress >= 1.0 && !has_primary && has_secondary {
        return ps.replace(STRESSES[0], &STRESSES[1].to_string());
    }
    if stress > 1.0 && !has_primary && !has_secondary {
        let has_vowel = ps.chars().any(|c| VOWELS.contains(c));
        if !has_vowel {
            return ps.to_string();
        }
        return restress(&format!("{}{}", STRESSES[1], ps));
    }
    ps.to_string()
}

fn restress(ps: &str) -> String {
    // Move each stress mark to just before the vowel it belongs to (misaki's restress).
    let chars: Vec<char> = ps.chars().collect();
    let n = chars.len();
    let mut items: Vec<(f32, char)> = chars.iter().enumerate().map(|(i, c)| (i as f32, *c)).collect();
    for i in 0..n {
        if STRESSES.contains(&chars[i]) {
            if let Some(j) = ((i + 1)..n).find(|&j| VOWELS.contains(chars[j])) {
                items[i].0 = j as f32 - 0.5;
            }
        }
    }
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    items.iter().map(|(_, c)| *c).collect()
}

pub struct Frontend<'a> {
    pub lex: &'a Lexicon,
    pub bart: &'a Bart,
    pub reduced: bool,
}

impl<'a> Frontend<'a> {
    /// The whole utterance as a phoneme string (normalized text included).
    pub fn phonemize(&self, text: &str) -> (String, String) {
        let norm = normalize(text);
        let ps = self.phonemize_raw(&norm);
        let ps = fold_extra(&fold(&ps));
        (ps, norm)
    }

    /// phonemes -> chunks of at most MAX_PHON phonemes, cut at sentence ends then at spaces.
    pub fn chunks(&self, text: &str) -> Vec<String> {
        let (ps, _) = self.phonemize(text);
        if ps.chars().count() <= MAX_PHON {
            return if ps.is_empty() { vec![] } else { vec![ps] };
        }
        let mut sentences: Vec<String> = Vec::new();
        let mut cur = String::new();
        for line in ps.split(' ') {
            cur.push_str(line);
            // sentence end markers attach to the last word of a sentence
            let ends = ["\u{2026}", ".", "!", "?"].iter().any(|t| line.ends_with(t));
            if ends {
                sentences.push(std::mem::take(&mut cur));
            }
            cur.push(' ');
        }
        if !cur.trim().is_empty() {
            sentences.push(cur.trim().to_string());
        }
        let mut out: Vec<String> = Vec::new();
        let mut acc = String::new();
        for s in sentences {
            for piece in wrap(&s) {
                let merged = if acc.is_empty() {
                piece.clone()
                } else {
                    format!("{acc} {piece}")
                };
                if merged.chars().count() > MAX_PHON && !acc.is_empty() {
                    out.push(std::mem::take(&mut acc));
                    acc = piece;
                } else {
                    acc = merged;
                }
            }
            if !acc.is_empty() {
                out.push(std::mem::take(&mut acc));
            }
        }
        if !acc.is_empty() {
            out.push(acc);
        }
        out
    }

    fn phonemize_raw(&self, text: &str) -> String {
        #[derive(Clone)]
        enum Piece {
            Space,
            Punct(char),
            Word(String, String), // lowercased spelling, phonemes
        }
        let mut pieces: Vec<Piece> = Vec::new();
        for tok in tokenize(text) {
            match tok {
                Token::Space => pieces.push(Piece::Space),
                Token::Punct(p) => {
                    if PUNCTS.contains(p) {
                        pieces.push(Piece::Punct(p));
                    }
                }
                Token::Word(w) => {
                    let ps = self.word(&w, &None, "");
                    if !ps.is_empty() {
                        pieces.push(Piece::Word(w.to_lowercase(), ps));
                    }
                }
                Token::Digits(d) => {
                    let n = d.parse().unwrap_or(0);
                    let words = if (1100..=2099).contains(&n) && d.len() == 4 && !d.starts_with('0') {
                        crate::num2words::year(n)
                    } else {
                        crate::num2words::cardinal(n)
                    };
                    for w in words.split(['-', ' ']) {
                        if !w.is_empty() {
                            let ps = self.word(w, &None, "");
                            if !ps.is_empty() {
                                pieces.push(Piece::Word(w.to_lowercase(), ps));
                            }
                        }
                    }
                }
            }
        }
        // Context-sensitive reductions, using the phonemes of the following word.
        for i in 0..pieces.len() {
            let next_word = pieces[i + 1..]
                .iter()
                .find_map(|p| if let Piece::Word(_, ps) = p { Some(ps.clone()) } else { None });
            if let Piece::Word(lower, ps) = pieces[i].clone() {
                let reduced = match lower.as_str() {
                    "the" => Some(if next_word.as_deref().map(is_vowel_start).unwrap_or(false) {
                        "\u{f0}\u{26a}".to_string()
                    } else {
                        "\u{f0}\u{259}".to_string()
                    }),
                    "to" => Some(match next_word {
                        Some(ref n) if is_vowel_start(n) => "t\u{28a}".to_string(),
                        Some(_) => "t\u{259}".to_string(),
                        None => "t\u{275}".to_string(),
                    }),
                    _ => None,
                };
                if let Some(r) = reduced {
                    if self.reduced {
                        pieces[i] = Piece::Word(lower, r);
                    }
                    let _ = ps;
                }
            }
        }
        // join: words are space separated, punctuation attaches to what precedes it
        let mut s = String::new();
        for piece in pieces {
            match piece {
                Piece::Space => {
                    if !s.is_empty() && !s.ends_with(' ') {
                        s.push(' ');
                    }
                }
                Piece::Punct(p) => s.push(p),
                Piece::Word(_, ps) => {
                    if !s.is_empty() && !s.ends_with(' ') {
                        s.push(' ');
                    }
                    s.push_str(&ps);
                }
            }
        }
        s.replace("  ", " ").trim().to_string()
    }

    /// One word token.
    pub fn word(&self, word: &str, _prev: &Option<String>, _next: &str) -> String {
        let w = fold_word(word);
        let w = w.trim_matches(|c| c == '\'' || c == '.' || c == '-' || c == ' ');
        if w.is_empty() {
            return String::new();
        }
        let lower = w.to_lowercase();
        let all_upper = w.chars().all(|c| c.is_uppercase()) && w.chars().count() >= 2;

        // Context reductions that the training G2P applied to function words.
        if self.reduced {
            if let Some(ps) = self.reduction(&lower) {
                return ps;
            }
        }
        // Lexicon, exact spelling first (gold > silver > CMU by insertion order). Every source is
        // spelled with ɾ / ʔ; the training phonemizer wrote them as the letters T / t.
        if let Some((_, ps)) = self.lex.map.get(w).map(|(p, ps)| (*p, ps.as_str())) {
            return apply_stress(&lex_icon(ps), if all_upper { Some(0.0) } else { None });
        }
        // ALL-CAPS with no entry of its own: spell the letters, as the training G2P does.
        if all_upper {
            if let Some(ps) = self.spell(&w) {
                return ps;
            }
        }
        if let Some((_, ps)) = self.lex.find(&lower) {
            return lex_icon(ps);
        }
        // possessive: the misaki -s rule, on the stem the dictionary has
        if let Some(stem) = w.strip_suffix("'s") {
            if let Some((_, ps)) = self
                .lex
                .map
                .get(stem)
                .map(|(p, ps)| (*p, ps.as_str()))
                .or_else(|| self.lex.find(stem))
            {
                return suffix_s(ps, false);
            }
        }
        if let Some(stem) = lower.strip_suffix("'s") {
            if let Some((_, ps)) = self.lex.find(stem) {
                return suffix_s(ps, false);
            }
        }
        // hyphenated: pronounce the parts and glue them (the training lists keep compounds tight,
        // and the trailing element loses its primary stress)
        if w.contains('-') {
            let mut parts: Vec<String> = w
                .split('-')
                .filter(|p| !p.is_empty())
                .map(|p| self.word(p, &None, ""))
                .collect();
            parts.retain(|p| !p.is_empty());
            if !parts.is_empty() {
                let last = parts.len() - 1;
                parts[last] = apply_stress(&parts[last].clone(), Some(-1.0));
                return parts.join("");
            }
        }
        // stems: -s, -ed, -ing
        if let Some(ps) = self.stem(&lower) {
            return ps;
        }
        // ALL-CAPS acronym: spell the letters
        if all_upper {
            if let Some(ps) = self.spell(&w) {
                return ps;
            }
        }
        // single letter: the letter's own name, the way the lexicon spells it
        if w.chars().count() == 1 {
            let up = w.to_uppercase();
            if let Some(ps) = self.lex.letters.get(&up.chars().next().unwrap()).cloned() {
                return ps;
            }
        }
        // last resort: the BART G2P
        self.bart.phonemes(w)
    }

    fn stem(&self, lower: &str) -> Option<String> {
        let n = lower.chars().count();
        let chars: Vec<char> = lower.chars().collect();
        // -s
        if n >= 3 && chars[n - 1] == 's' && !(chars[n - 2] == 's') {
            for cut in [1usize, 2] {
                let stem: String = chars[..n - cut].iter().collect();
                if (cut == 1) || lower.ends_with("'s") || (n > 4 && lower.ends_with("es") && !lower.ends_with("ies")) {
                    if let Some((_, ps)) = self.lex.find(&stem) {
                        return Some(suffix_s(ps, false));
                    }
                }
            }
            if n > 4 && lower.ends_with("ies") {
                let stem: String = chars[..n - 3].iter().collect::<String>() + "y";
                if let Some((_, ps)) = self.lex.find(&stem) {
                    return Some(suffix_s(ps, false));
                }
            }
        }
        // -ed
        if n >= 4 && chars[n - 1] == 'd' {
            if chars[n - 2] != 'd' {
                let stem: String = chars[..n - 1].iter().collect();
                if let Some((_, ps)) = self.lex.find(&stem) {
                    return Some(suffix_ed(ps, false));
                }
            }
            if n > 4 && lower.ends_with("ed") && !lower.ends_with("eed") {
                let stem: String = chars[..n - 2].iter().collect();
                if let Some((_, ps)) = self.lex.find(&stem) {
                    return Some(suffix_ed(ps, false));
                }
            }
        }
        // -ing
        if n >= 5 && lower.ends_with("ing") && n > 5 {
            let stem: String = chars[..n - 3].iter().collect();
            if let Some((_, ps)) = self.lex.find(&stem) {
                return Some(suffix_ing(ps));
            }
            // "waking" -> "wake" + ing: the silent-e form
            let stem_e: String = chars[..n - 3].iter().collect::<String>() + "e";
            if let Some((_, ps)) = self.lex.find(&stem_e) {
                return Some(format!("{}ɪŋ", ps));
            }
        }
        None
    }

    fn spell(&self, word: &str) -> Option<String> {
        let mut ps = String::new();
        for c in word.chars() {
            match self.lex.letters.get(&c) {
                Some(v) => ps.push_str(v),
                None => return None,
            }
        }
        let ps = apply_stress(&ps, Some(0.0));
        let parts: Vec<&str> = ps.rsplit_once(STRESSES[0]).map(|(a, b)| vec![a, b]).unwrap_or(vec![ps.as_str()]);
        Some(parts.join(&STRESSES[1].to_string()))
    }

    fn reduction(&self, lower: &str) -> Option<String> {
        // "the" before a vowel word, "a" as an article, "to" as a reduced infinitive: the three the
        // training G2P conditioned on context. Reduced forms for the rest come from the lexicon.
        match lower {
            "a" => Some("\u{250}".to_string()), // ɐ
            _ => None,
        }
    }
}

fn suffix_s(stem: &str, british: bool) -> String {
    let last = stem.chars().last().unwrap_or(' ');
    if "ptkf\u{3b8}".contains(last) {
        format!("{stem}s")
    } else if "sz\u{283}\u{292}\u{2a7}\u{2a4}".contains(last) {
        format!("{stem}{}z", if british { "\u{26a}" } else { "\u{1d7b}" })
    } else {
        format!("{stem}z")
    }
}

fn suffix_ed(stem: &str, british: bool) -> String {
    let chars: Vec<char> = stem.chars().collect();
    let last = chars.last().copied().unwrap_or(' ');
    if "pkf\u{3b8}\u{283}s\u{2a7}".contains(last) {
        format!("{stem}t")
    } else if last == 'd' {
        format!("{stem}{}d", if british { "\u{26a}" } else { "\u{1d7b}" })
    } else if last == 't' {
        if british || chars.len() < 2 {
            format!("{stem}\u{26a}d")
        } else {
            let taus = "AIOWYiu\u{e6}\u{251}\u{259}\u{25b}\u{279}\u{28a}\u{28c}\u{1d7b}";
            if taus.contains(chars[chars.len() - 2]) {
                format!("{}\u{27e}\u{1d7b}d", &stem[..stem.len() - 1])
            } else {
                format!("{stem}\u{1d7b}d")
            }
        }
    } else {
        format!("{stem}d")
    }
}

fn suffix_ing(stem: &str) -> String {
    let chars: Vec<char> = stem.chars().collect();
    if chars.len() > 1 && chars[chars.len() - 1] == 't'
        && "AIOWYiu\u{e6}\u{251}\u{259}\u{25b}\u{279}\u{28a}\u{28c}\u{1d7b}".contains(chars[chars.len() - 2])
    {
        format!("{}\u{27e}\u{26a}\u{14b}", &stem[..stem.len() - 1])
    } else {
        format!("{stem}\u{26a}\u{14b}")
    }
}

/// phonemize_v1.FOLD + phonemize_v2.FOLD_EXTRA
/// phonemize_v1.FOLD + phonemize_v2.FOLD_EXTRA, exposed for the espeak path.
pub fn fold_public(ps: &str) -> String {
    fold_extra(&fold(ps))
}

fn fold(ps: &str) -> String {
    let table: [(char, &str); 12] = [
        ('\u{32a}', ""), ('\u{329}', ""), ('\u{30d}', ""), ('\u{32f}', ""),
        ('\u{26d}', "l"), ('\u{26b}', "l"), ('\u{275}', "\u{f8}"), ('\u{3b5}', "\u{25b}"),
        ('\u{5}', ""), ('\u{bf}', ""), ('\u{a1}', ""), ('\u{202f}', " "),
    ];
    let mut s = ps.to_string();
    for (c, r) in table {
        s = s.replace(c, r);
    }
    for c in ['{', '}', '[', ']', '\u{6}', '\u{7}', '\u{8}'] {
        s = s.replace(c, "");
    }
    s
}

fn fold_extra(ps: &str) -> String {
    let mut s = ps
        .replace('\u{2009}', " ")
        .replace('\t', " ")
        .replace('\u{ab}', "")
        .replace('\u{bb}', "")
        .replace('\u{2018}', "")
        .replace('\u{2019}', "")
        .replace('\u{b7}', "")
        .replace('~', "")
        .replace('#', "")
        .replace('g', "\u{261}");
    s = s.replace('\'', "");
    s.trim().to_string()
}

/// Strip accents so "caf" stays sayable, the way the training G2P folds them.
fn fold_word(w: &str) -> String {
    let mut s = String::with_capacity(w.len());
    for c in w.chars() {
        if let Some((from, to)) = LETTER_TRANSLIT.iter().find(|(f, _)| *f == c) {
            let _ = from;
            s.push_str(to);
        } else {
            // drop combining marks (NFKD-decomposed accents)
            let d = c as u32;
            let combining = (0x300..=0x36f).contains(&d) || (0x1ab0..=0x1aff).contains(&d)
                || (0x1dc0..=0x1dff).contains(&d) || (0x20d0..=0x20ff).contains(&d);
            if !combining {
                s.push(c);
            }
        }
    }
    s
}

enum Token {
    Space,
    Punct(char),
    Word(String),
    Digits(String),
}

fn tokenize(text: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            out.push(Token::Space);
            i += 1;
        } else if c.is_alphabetic() || c == '\'' || c == '\u{2019}' || c == '-' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphabetic() || chars[i] == '\'' || chars[i] == '\u{2019}') {
                i += 1;
            }
            // a hyphen glues word parts together (the dictionaries carry hyphenated entries)
            while i < chars.len() && (chars[i] == '-' || chars[i] == '\u{2010}') {
                let mut j = i + 1;
                while j < chars.len() && (chars[j].is_alphabetic() || chars[j] == '\'' || chars[j] == '\u{2019}') {
                    j += 1;
                }
                if j == i + 1 {
                    break;
                }
                i = j;
            }
            let w: String = chars[start..i]
                .iter()
                .collect::<String>()
                .trim_matches('-')
                .to_string();
            if w.is_empty() {
                // a bare hyphen: consume the character and keep moving
                out.push(Token::Punct('-'));
                i = start + 1;
            } else {
                out.push(Token::Word(w));
            }
        } else if c.is_ascii_digit() || c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',') {
                i += 1;
            }
            let raw: String = chars[start..i].iter().collect();
            let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
            out.push(Token::Digits(digits));
        } else {
            out.push(Token::Punct(c));
            i += 1;
        }
    }
    out
}

/// Cut one over-long phoneme string at spaces (phonemize_v2._wrap).
fn wrap(ps: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = ps.to_string();
    loop {
        let len = rest.chars().count();
        if len <= MAX_PHON {
            if !rest.trim().is_empty() {
                out.push(rest.trim().to_string());
            }
            return out;
        }
        // last space within the limit
        let mut best = None;
        for (i, c) in rest.char_indices() {
            if i > MAX_PHON {
                break;
            }
            if c == ' ' {
                best = Some(i);
            }
        }
        let cut = best.unwrap_or(MAX_PHON);
        let head: String = rest.chars().take(cut).collect();
        out.push(head);
        rest = rest.chars().skip(cut).collect::<String>().trim().to_string();
    }
}

pub fn tokens_for(phonemes: &str, vocab: &Vocab) -> Vec<i64> {
    let mut ids = vec![0i64];
    for c in phonemes.chars() {
        match vocab.map.get(&c) {
            Some(id) => ids.push(*id),
            None => {
                if c == '(' || c == ')' {
                    ids.push(*vocab.map.get(&',').unwrap_or(&3));
                }
            }
        }
    }
    ids.push(10);
    ids.push(0);
    ids
}

