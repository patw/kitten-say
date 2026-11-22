# kitten-say

Text to speech with the [KittenTTS 2](https://huggingface.co/KittenML/kitten-tts-2-mini) models in
one self-contained Rust binary: CPU only, no GPU, no Python, no torch, no model download at
runtime, no shared libraries beyond the system libc.

It is a port of the model's official Python/ONNX front end. The only file it reads at run time is
the model `.onnx`; everything the Python package loads out of its repo — the phoneme vocabulary,
the English lexicons, the voice style packs, even the small neural grapheme→phoneme model — is
embedded in the executable.

```
kitten-say -m kitten_tts_2_mini.onnx -t "Hello world!"          # speak it
kitten-say -m model.onnx -t "Hello world!" -o hello.wav         # write a wav
kitten-say -m model.onnx -p -t "Hello world!"                   # phonemes only
echo "piped text" | kitten-say -m model.onnx -v max             # pick a voice
kitten-say -m model.onnx --voices --json                         # list voices
```

`-h/--help` lists the options; `-q` silences the stderr progress lines.

## What you need

1. **The model.** `kitten_tts_2_mini.onnx` (~82 MB) from the root of the
   [KittenML/kitten-tts-2-mini](https://huggingface.co/KittenML/kitten-tts-2-mini) repository:

   ```
   huggingface-cli download KittenML/kitten-tts-2-mini kitten_tts_2_mini.onnx --local-dir .
   ```

2. **A Rust toolchain.** `cargo build --release` fetches and statically links ONNX Runtime (the
   `ort` crate's `download-binaries` build), so the first build needs network access and a C++
   toolchain on the build host. Nothing of it is needed at run time.

```
cargo build --release
install -m 755 target/release/kitten-say ~/.local/bin/kitten-say
```

## What "no deps other than the onnx file" means here

| embedded                | source in the HF repo            | gzipped size |
|-------------------------|----------------------------------|-------------:|
| phoneme vocabulary      | `vocab.json`                     |        1 KB |
| English lexicons        | `g2p/cmu_misaki_us.json` + misaki's `us_gold.json` / `us_silver.json` (the dictionary the training phonemizer uses) | 3.6 MB |
| letter names            | single-letter entries of the gold lexicon | <1 KB |
| voice style packs       | `voices_en.npz` (11 voices, one style vector per input length) | 4.5 MB |
| grapheme→phoneme model  | `g2p/g2p_bart_en_us/model.safetensors` (0.75M-param char-level BART, re-implemented in Rust) | 2.8 MB |

ONNX Runtime is linked **statically**, so `ldd` shows only libc/libm/libstdc++/libgcc. The binary
is ~34 MB, peak RSS ~270 MB.

Non-English languages (`-l de`, `-l fr`, …) are the one place where the single-file promise breaks:
the HF repo ships those dictionaries inside `phonemizer`/`espeak-ng`, so the tool shells out to the
`espeak-ng` binary (its own process, not linked in) and reads `voices_<lang>.npz` next to the
model. English needs nothing. 25 language codes are mapped; see `src/espeak.rs`.

## Performance (20-thread x86-64 desktop, CPU only)

| metric                              | value                              |
|-------------------------------------|------------------------------------|
| startup (decompress + parse lexicon + load model) | ~0.2 s |
| synthesis                            | ~3× realtime                      |
| single-threaded synthesis            | ~1.2× realtime                    |
| output                               | 24 kHz mono, 16-bit PCM           |

The decoder samples noise, so the same sentence renders slightly differently on each run — the
same property the Python path has under a sampling decoder. Length, energy and envelope match the
reference; sample-for-sample equality is not achievable with a sampling decoder.

## How it works

1. **Normalise** — a port of the model's `tts_norm`: numbers, years, times, dates, money, phones,
   fractions, units, abbreviations, roman numerals, URLs, emails, symbols.
2. **Phonemise** — the same dictionary cascade the training data used: misaki gold → misaki silver
   → CMU (converted to the model's alphabet) → morphological stems (`-s`, `-ed`, `-ing`) →
   letter-spelling for ALL-CAPS → the char-level BART G2P for whatever is left. Context rules for
   `the`/`a`/`to`, the flap/glottal fold (`ɾ→T`, `ʔ→t`), the v1+v2 FOLD tables, and the vocab
   filter, then chunked at 350 phonemes at sentence ends.
3. **Synthesise** — `tokens → [0] + ids + [10, 0]`, the style row for that token length out of the
   voice pack, and the ONNX graph (`input_ids`, `style`, `speed` → `waveform`, `durations`). Each
   chunk is trimmed by its own durations (leading pause, end-token frames), 5 ms fades are applied,
   and chunks are joined with 0.15 s gaps.

## Regenerating the embedded assets

`assets/*.gz` are committed, so a normal `cargo build` needs nothing else. They are generated from
an unmodified snapshot of the HF repo plus an installed [misaki](https://github.com/hexgrad/misaki):

```
python3 prep_assets.py \
    --model-dir  ~/models/kitten-tts-2-mini \
    --misaki-dir .venv/lib/python3.12/site-packages/misaki
```

`--out` defaults to `./assets`; `--golden-dir` additionally copies reference fixtures into
`assets/golden/`. The assets are `include_bytes!`-embedded, so a rebuild must be reinstalled for a
change to take effect.

## Checks against the reference

A local parity harness compares this port stage by stage against the official Python pipeline:

| stage | what is compared | result |
|-------|------------------|--------|
| `tts_norm` | normalised text, 22 sentences | 22/22 exact |
| phonemes | phoneme strings, 22 sentences | 14 exact, mean similarity 0.991 |
| BART G2P | greedy decode vs transformers | 47/51 words exact |
| audio | wav length, energy, envelope vs reference audio | length 1.000, envelope corr 0.96 |
| voices | 11 voices × 0.8–1.6× speed | all pass, self-consistent |

Remaining phoneme differences are dictionary-choice and stress-placement nuances (misaki picks word
variants from spaCy POS tags, which this port does not model); none of them take a string out of
the model's vocabulary. The harness itself is not part of this repository yet — it needs torch,
onnxruntime, numpy and soundfile to build its fixtures.

## Repository layout

```
src/          norm.rs (text normalisation), g2p.rs (lexicon cascade + FOLD),
              bart.rs (char-level BART in plain Rust), espeak.rs (non-English),
              synth.rs (ONNX session + chunk joining), npz.rs, assets.rs, num2words.rs
assets/       the embedded blobs (gzipped) + assets/golden/ reference fixtures
prep_assets.py  regenerates assets/ from the HF snapshot
```

## Licence

The Rust code is MIT (see `LICENSE`). The embedded assets are derived from third-party sources
under their own licences — the KittenTTS 2 model weights and its BART G2P are Apache-2.0, misaki's
lexicons are MIT, and the CMU pronouncing dictionary is BSD-2-Clause. See `NOTICE`. `espeak-ng`,
used only as an external process for non-English languages, is GPLv3 and is neither linked nor
redistributed here.
