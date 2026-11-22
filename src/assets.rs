//! Embedded assets: the phoneme vocab, the English lexicons (misaki us_gold/us_silver + CMU->misaki),
//! the letter table, the voice style packs and the char-level BART G2P weights.
//! Everything is embedded in the binary (gzipped); the only external file the tool needs is the model .onnx.
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::io::Read;

pub static VOCAB_GZ: &[u8] = include_bytes!("../assets/vocab.tsv.gz");
pub static LEXICON_GZ: &[u8] = include_bytes!("../assets/lexicon.tsv.gz");
pub static LETTERS_GZ: &[u8] = include_bytes!("../assets/letters.tsv.gz");
pub static VOICES_GZ: &[u8] = include_bytes!("../assets/voices_en.npz.gz");
pub static BART_GZ: &[u8] = include_bytes!("../assets/bart.safetensors.gz");

pub fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    GzDecoder::new(bytes).read_to_end(&mut out).expect("gunzip failed");
    out
}

/// symbol -> token id, plus the id of the end token / pad used by the model.
pub struct Vocab {
    pub map: HashMap<char, i64>,
}

impl Vocab {
    pub fn load() -> Vocab {
        let text = String::from_utf8(gunzip(VOCAB_GZ)).expect("vocab not utf8");
        let mut map = HashMap::new();
        for line in text.lines() {
            let mut it = line.split('\t');
            let sym = it.next().unwrap_or("");
            let id: i64 = it.next().unwrap_or("0").parse().unwrap_or(0);
            for c in sym.chars() {
                map.insert(c, id);
            }
        }
        Vocab { map }
    }
}

/// word -> (priority, phonemes). 4 = misaki gold, 3 = misaki silver, 2 = CMU (converted).
/// Two indexes: the spelling as the dictionary writes it, and a lowercase one for the fallback.
pub struct Lexicon {
    pub map: HashMap<String, (u8, String)>,
    pub lower: HashMap<String, (u8, String)>,
    pub letters: HashMap<char, String>,
}

impl Lexicon {
    pub fn load() -> Lexicon {
        let text = String::from_utf8(gunzip(LEXICON_GZ)).expect("lexicon not utf8");
        let mut map = HashMap::with_capacity(300_000);
        let mut lower = HashMap::with_capacity(300_000);
        for line in text.lines() {
            let mut it = line.split('\t');
            let (Some(p), Some(w), Some(ps)) = (it.next(), it.next(), it.next()) else { continue };
            let prio: u8 = p.parse().unwrap_or(9);
            map.entry(w.to_string()).or_insert((prio, ps.to_string()));
            lower.entry(w.to_lowercase()).or_insert((prio, ps.to_string()));
        }
        let text = String::from_utf8(gunzip(LETTERS_GZ)).expect("letters not utf8");
        let mut letters = HashMap::new();
        for line in text.lines() {
            let mut it = line.split('\t');
            if let (Some(c), Some(v)) = (it.next(), it.next()) {
                if let Some(c) = c.chars().next() {
                    letters.insert(c, v.to_string());
                }
            }
        }
        Lexicon {
            map,
            lower,
            letters,
        }
    }

    /// The dictionary's own spelling first (how the training G2P resolves case), then lowercase.
    pub fn find(&self, word: &str) -> Option<(u8, &str)> {
        self.map
            .get(word)
            .or_else(|| self.lower.get(&word.to_lowercase()))
            .map(|(p, ps)| (*p, ps.as_str()))
    }
}
