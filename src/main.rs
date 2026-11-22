//! kitten-say: text to speech with the KittenTTS 2 models, from a single binary.
//!
//!   kitten-say -m kitten_tts_2_mini.onnx -t "Hello world!"
//!
//! The model .onnx is the only file it needs: the phoneme vocab, the English lexicons
//! (misaki gold/silver + CMU), the letter table, the voice packs and the char-level BART G2P are
//! embedded in the binary.
use std::io::Read;
use std::process::{Command, Stdio};

mod assets;
mod bart;
mod g2p;
mod norm;
mod npz;
mod num2words;
mod synth;

use assets::{Lexicon, Vocab};
use bart::Bart;
use g2p::Frontend;
use synth::Synth;

const HELP: &str = "\
kitten-say - KittenTTS 2 speech synthesis (CPU, offline)

  kitten-say -m kitten_tts_2_mini.onnx -t \"Hello world!\"
  kitten-say -m model.onnx -t text -v Eleanor -s 1.1 -o out.wav

options
  -m, --model PATH     the model .onnx (required)
  -t, --text TEXT      text to speak (default: read stdin)
  -v, --voice NAME     voice name, e.g. Eleanor, Max, danny (default: the language's first)
  -l, --lang CODE      language (default: en). Non-English needs espeak-ng and voices_<lang>.npz
  -s, --speed FACT     >1 speaks faster (default 1.0)
  -o, --output PATH    write a .wav here (default: play it, keeping nothing)
  -p, --phonemes       print the phoneme chunks and exit (no audio)
      --bart           print the BART G2P output for each word (debug)
      --words          print the lexicon/BART phonemes for each word (debug)
      --voices         list the voices of the selected language (--json for machine-readable)
      --threads N      onnxruntime intra-op threads (default: all)
      --keep           keep the temporary .wav when playing
  -q, --quiet          no progress chatter on stderr
      --json           (with --voices) JSON output
  -h, --help           this text
";

struct Args {
    model: String,
    text: String,
    voice: Option<String>,
    lang: String,
    speed: f32,
    output: Option<String>,
    phonemes: bool,
    bart_only: bool,
    debug: bool,
    normalize_only: bool,
    words_only: bool,
    list_voices: bool,
    threads: usize,
    keep: bool,
    quiet: bool,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model = String::new();
    let mut text = String::new();
    let mut voice = None;
    let mut lang = "en".to_string();
    let mut speed = 1.0f32;
    let mut output = None;
    let mut phonemes = false;
    let mut bart_only = false;
    let mut debug = false;
    let mut normalize_only = false;
    let mut words_only = false;
    let mut list_voices = false;
    let mut threads = 0usize;
    let mut keep = false;
    let mut quiet = false;
    let mut json = false;
    let mut i = 0usize;
    let mut text_from_stdin = true;
    let next = |i: usize| argv.get(i).cloned().ok_or(format!("{} needs a value", argv[i]));
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut used = 1usize;
        match a {
            "-m" | "--model" => {
                model = next(i + 1)?;
                used = 2;
            }
            "-t" | "--text" => {
                text = next(i + 1)?;
                text_from_stdin = false;
                used = 2;
            }
            "-v" | "--voice" => {
                voice = Some(next(i + 1)?);
                used = 2;
            }
            "-l" | "--lang" => {
                lang = next(i + 1)?;
                used = 2;
            }
            "-s" | "--speed" => {
                speed = next(i + 1)?.parse().map_err(|_| "-s needs a number".to_string())?;
                used = 2;
            }
            "-o" | "--output" => {
                output = Some(next(i + 1)?);
                used = 2;
            }
            "-p" | "--phonemes" => phonemes = true,
            "--bart" => bart_only = true,
            "--normalize" => normalize_only = true,
            "--debug" => debug = true,
            "--words" => words_only = true,
            "--voices" => list_voices = true,
            "--threads" => {
                threads = next(i + 1)?.parse().map_err(|_| "--threads needs a number".to_string())?;
                used = 2;
            }
            "--keep" => keep = true,
            "-q" | "--quiet" => quiet = true,
            "--json" => json = true,
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            _ => {
                if a.starts_with('-') {
                    return Err(format!("unknown option {a}"));
                }
                // a bare argument is the text
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(a);
                text_from_stdin = false;
            }
        }
        i += used;
    }
    if text_from_stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).map_err(|e| e.to_string())?;
        text = buf;
    }
    if model.is_empty() {
        return Err("no -m/--model given".to_string());
    }
    Ok(Args {
        model,
        text,
        voice,
        lang,
        speed,
        output,
        phonemes,
        bart_only,
        debug,
        normalize_only,
        words_only,
        list_voices,
        threads,
        keep,
        quiet,
        json,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("kitten-say: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("{e} (try --help)"))?;
    let t0 = std::time::Instant::now();
    if args.normalize_only {
        println!("{}", norm::normalize(&args.text));
        return Ok(());
    }
    let vocab = Vocab::load();
    let lex = Lexicon::load();
    let bart = Bart::load();
    // voice packs: embedded for English, a file next to the model for the other languages
    let voices = load_voices(&args.model, &args.lang)?;
    if args.list_voices {
        if args.json {
            let items: Vec<String> = voices
                .iter()
                .map(|(k, _, _)| format!("{{\"key\":\"{k}\"}}"))
                .collect();
            println!("{{\"language\":\"{}\",\"voices\":[{}]}}", args.lang, items.join(","));
        } else {
            for (k, _, _) in &voices {
                println!("{k}");
            }
        }
        return Ok(());
    }
    let voice = pick_voice(&voices, args.voice.as_deref(), &args.lang)?;
    let frontend = Frontend {
        lex: &lex,
        bart: &bart,
        reduced: true,
    };
    if args.bart_only {
        for w in args.text.split_whitespace() {
            println!("{}", bart.phonemes_dbg(w, args.debug));
        }
        return Ok(());
    }
    if args.words_only {
        for w in args.text.split_whitespace() {
            println!("{}", frontend.word(w, &None, ""));
        }
        return Ok(());
    }
    let chunks = if args.lang == "en" {
        frontend.chunks(&args.text)
    } else {
        let ps = crate::espeak::phonemize(&args.lang, &args.text, &vocab)?;
        vec![ps]
    };
    if args.phonemes {
        for c in &chunks {
            println!("{c}");
        }
        return Ok(());
    }
    let token_chunks: Vec<Vec<i64>> = chunks.iter().map(|c| g2p::tokens_for(c, &vocab)).collect();
    if !args.quiet {
        eprintln!(
            "kitten-say: {} chunk(s), {} phonemes, voice {voice}, {:.1}s setup",
            token_chunks.len(),
            chunks.iter().map(|c| c.chars().count()).sum::<usize>(),
            t0.elapsed().as_secs_f32()
        );
    }
    let mut s = Synth::new(&args.model, voices, args.threads)?;
    let t1 = std::time::Instant::now();
    let audio = s.synthesize(&token_chunks, &voice, args.speed);
    if !args.quiet {
        eprintln!(
            "kitten-say: {:.2}s of audio in {:.2}s ({:.1}x realtime)",
            audio.len() as f32 / synth::SAMPLE_RATE as f32,
            t1.elapsed().as_secs_f32(),
            (audio.len() as f32 / synth::SAMPLE_RATE as f32) / t1.elapsed().as_secs_f32().max(1e-6)
        );
    }
    let out = match &args.output {
        Some(p) => p.clone(),
        None => {
            let p = std::env::temp_dir().join(format!("kitten-say-{}.wav", std::process::id()));
            p.to_string_lossy().to_string()
        }
    };
    synth::wav::write(&out, &audio, synth::SAMPLE_RATE)?;
    if args.output.is_none() {
        play(&out, args.keep)?;
    } else if !args.quiet {
        eprintln!("kitten-say: wrote {out}");
    }
    Ok(())
}

fn load_voices(model: &str, lang: &str) -> Result<Vec<(String, Vec<f32>, usize)>, Box<dyn std::error::Error>> {
    let blob = if lang == "en" {
        assets::gunzip(assets::VOICES_GZ)
    } else {
        let path = std::path::Path::new(model)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(format!("voices_{lang}.npz"));
        if !path.exists() {
            return Err(format!("no embedded voices for {lang:?}; expected {}", path.display()).into());
        }
        std::fs::read(&path)?
    };
    let voices = npz::voices_from_npz(&blob);
    if voices.is_empty() {
        return Err("voice packs could not be parsed".into());
    }
    let names: Vec<String> = voices.iter().map(|(k, _, _)| k.clone()).collect();
    let _ = names;
    Ok(voices)
}

fn pick_voice(voices: &[(String, Vec<f32>, usize)], want: Option<&str>, lang: &str) -> Result<String, Box<dyn std::error::Error>> {
    let available: Vec<&str> = voices.iter().map(|(k, _, _)| k.as_str()).collect();
    match want {
        None => Ok(available[0].to_string()),
        Some(w) => {
            let w = w.to_lowercase();
            available
                .iter()
                .find(|k| *k == &w)
                .map(|k| k.to_string())
                .ok_or_else(|| format!("no voice {w:?} for {lang}; voices: {}", available.join(", ")).into())
        }
    }
}

fn play(path: &str, keep: bool) -> Result<(), Box<dyn std::error::Error>> {
    for (cmd, args) in [
        ("pw-play", vec![path]),
        ("aplay", vec![path]),
        ("paplay", vec![path]),
        ("ffplay", vec!["-nodisp", "-autoexit", "-loglevel", "quiet", path]),
    ] {
        if which(cmd) {
            let status = Command::new(cmd)
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            match status {
                Ok(s) if s.success() => {
                    if !keep {
                        let _ = std::fs::remove_file(path);
                    }
                    return Ok(());
                }
                _ => continue,
            }
        }
    }
    eprintln!("kitten-say: no player found (pw-play/aplay/paplay/ffplay); the wav is at {path}");
    Ok(())
}

fn which(cmd: &str) -> bool {
    let path = match std::env::var("PATH") {
        Ok(p) => p,
        Err(_) => return false,
    };
    path.split(':').any(|dir| std::path::Path::new(dir).join(cmd).exists())
}

mod espeak;
