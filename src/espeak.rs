//! Non-English languages: espeak-ng as the G2P, with the same symbol folding the python front end
//! applies (misaki.espeak.EspeakG2P + phonemize_v1/v2 FOLD). Needs the espeak-ng binary.
use std::process::Command;

use crate::assets::Vocab;

pub fn espeak_voice(lang: &str) -> Option<&'static str> {
    Some(match lang {
        "es" => "es",
        "fr" => "fr-fr",
        "it" => "it",
        "pt" => "pt-br",
        "hi" => "hi",
        "de" => "de",
        "nl" => "nl",
        "pl" => "pl",
        "ru" => "ru",
        "tr" => "tr",
        "ar" => "ar",
        "he" => "he",
        "sv" => "sv",
        "da" => "da",
        "no" | "nb" => "nb",
        "fi" => "fi",
        "el" => "el",
        "cs" => "cs",
        "hu" => "hu",
        "ro" => "ro",
        "uk" => "uk",
        "ms" => "ms",
        "sw" => "sw",
        "ko" => "ko",
        "en" => "en-us",
        "en-us" => "en-us",
        "en-gb" => "en-gb",
        _ => return None,
    })
}

/// misaki.espeak.EspeakG2P.e2m (the version-1 table; the 2.0 one adds the nasal-capital forms).
fn e2m() -> Vec<( &'static str, &'static str)> {
    vec![
        ("a^ɪ", "I"),
        ("a^ʊ", "W"),
        ("d^z", "ʣ"),
        ("d^ʒ", "ʤ"),
        ("e^ɪ", "A"),
        ("o^ʊ", "O"),
        ("ə^ʊ", "Q"),
        ("s^s", "S"),
        ("t^s", "ʦ"),
        ("t^ʃ", "ʧ"),
        ("ɔ^ɪ", "Y"),
    ]
}

pub fn phonemize(lang: &str, text: &str, vocab: &Vocab) -> Result<String, Box<dyn std::error::Error>> {
    let voice = espeak_voice(lang).ok_or(format!("no espeak voice mapped for {lang:?}"))?;
    let dir = std::env::temp_dir();
    let id = std::process::id();
    let inp = dir.join(format!("kitten-espeak-{id}.txt"));
    let out = dir.join(format!("kitten-espeak-{id}.phon"));
    std::fs::write(&inp, text.as_bytes())?;
    let status = Command::new("espeak-ng")
        .arg("-q")
        .arg("-v")
        .arg(voice)
        .arg("-f")
        .arg(&inp)
        .arg("--phonout")
        .arg(&out)
        .status()?;
    let _ = std::fs::remove_file(&inp);
    if !status.success() {
        return Err("espeak-ng failed (is it installed?)".into());
    }
    let raw = std::fs::read_to_string(&out)?;
    let _ = std::fs::remove_file(&out);
    let mut ps = raw.trim().to_string();
    // « » come back as parentheses, and the phonemizer's tie character is dropped
    ps = ps.replace('«', "\u{201c}").replace('»', "\u{201d}");
    for (old, new) in e2m() {
        ps = ps.replace(old, new);
    }
    ps = ps.replace('^', "").replace('-', "");
    ps = ps.replace('\u{201c}', "(").replace('\u{201d}', ")");
    ps = crate::g2p::fold_public(&ps);
    let kept: String = ps.chars().filter(|c| vocab.map.contains_key(c)).collect();
    Ok(kept)
}
