#!/usr/bin/env python3
"""Build the embedded assets for kitten-say.

Writes into --out (default: ./assets next to this script):

  lexicon.tsv.gz       priority<TAB>word<TAB>phonemes lines: misaki us_gold + us_silver + CMU
                       (converted to the model's alphabet), with the dictionary's own
                       capitalized/lowercase twins grown in and POS sub-dictionaries
                       flattened to DEFAULT. Priority byte: gold=4, silver=3, cmu=2.
  letters.tsv.gz       single-letter pronunciations (from the gold lexicon, as written)
  vocab.tsv.gz         symbol<TAB>id
  voices_en.npz(.gz)   the model's voice style packs, verbatim (a zip of .npy arrays)
  bart.safetensors.gz  the char-level BART G2P weights
  golden/              reference fixtures, only when --golden-dir is given

Inputs are the files the tool embeds, so point this at an unmodified snapshot of
https://huggingface.co/KittenML/kitten-tts-2-mini plus an installed `misaki`
(https://github.com/hexgrad/misaki). Example:

  python3 prep_assets.py \\
      --model-dir ~/models/kitten-tts-2-mini \\
      --misaki-dir .venv/lib/python3.12/site-packages/misaki

The committed assets are already up to date; re-run this only when the model repo
or the lexicons change, then rebuild — the assets are include_bytes!-embedded, so a
rebuild alone is not enough, the binary has to be reinstalled.
"""
import argparse
import gzip
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = ("phonemes.json", "norm.json", "bart.json", "say.json", "utt_case0.npy",
            "utt_case4.npy", "onnx_c0_0.npz", "onnx_c4_0.npz", "utt_case0.wav",
            "utt_case4.wav")


def write_gz(path, data):
    """gzip with mtime=0 so regenerating the assets is byte-reproducible."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as f:
            f.write(data)


def grow(d):
    """Add the capitalized/lowercase twin misaki's grow_dictionary() would add."""
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def flat(v):
    """misaki entries are either a string or {"DEFAULT": str, "<pos>": str, ...}."""
    if isinstance(v, dict):
        return v.get("DEFAULT")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True,
                    help="snapshot of KittenML/kitten-tts-2-mini (vocab.json, voices_en.npz, g2p/)")
    ap.add_argument("--misaki-dir", required=True,
                    help="installed misaki package directory (contains data/us_gold.json)")
    ap.add_argument("--out", default=os.path.join(HERE, "assets"),
                    help="output directory (default: ./assets next to this script)")
    ap.add_argument("--golden-dir", default=None,
                    help="optional directory of reference fixtures to copy into <out>/golden")
    a = ap.parse_args()

    model, misaki, out = a.model_dir, a.misaki_dir, a.out
    vocab_json = f"{model}/vocab.json"
    voices_npz = f"{model}/voices_en.npz"
    cmu_json = f"{model}/g2p/cmu_misaki_us.json"
    bart_weights = f"{model}/g2p/g2p_bart_en_us/model.safetensors"
    for path in (vocab_json, voices_npz, cmu_json, bart_weights,
                 f"{misaki}/data/us_gold.json", f"{misaki}/data/us_silver.json"):
        if not os.path.isfile(path):
            sys.exit(f"missing input: {path}")
    os.makedirs(out, exist_ok=True)

    gold = grow(json.load(open(f"{misaki}/data/us_gold.json")))
    silver = grow(json.load(open(f"{misaki}/data/us_silver.json")))
    cmu = grow(json.load(open(cmu_json)))

    rows, seen = [], set()
    for prio, d in ((4, gold), (3, silver), (2, cmu)):
        for w, v in d.items():
            ps = flat(v)
            if not ps:
                continue
            # keep the spelling as the dictionary writes it: the training G2P resolves case, so
            # both "CPU" (the acronym form) and "cpu" (a word) have to stay addressable
            if w in seen:
                continue
            seen.add(w)
            rows.append(f"{prio}\t{w}\t{ps}")
    write_gz(f"{out}/lexicon.tsv.gz", "\n".join(rows).encode())
    print("lexicon rows:", len(rows), "bytes gz:", os.path.getsize(f"{out}/lexicon.tsv.gz"))

    letters = {c: v for c, v in gold.items() if len(c) == 1 and c.isalpha()}
    write_gz(f"{out}/letters.tsv.gz",
             "".join(f"{c}\t{v}\n" for c, v in sorted(letters.items())).encode())
    print("letters:", len(letters))

    with open(vocab_json) as f:
        vocab = json.load(f)
    write_gz(f"{out}/vocab.tsv.gz",
             "".join(f"{k}\t{v}\n" for k, v in sorted(vocab.items(), key=lambda kv: kv[1])).encode())
    print("vocab:", len(vocab))

    with open(voices_npz, "rb") as f:
        voices = f.read()
    shutil.copyfile(voices_npz, f"{out}/voices_en.npz")
    write_gz(f"{out}/voices_en.npz.gz", voices)
    with open(bart_weights, "rb") as f:
        write_gz(f"{out}/bart.safetensors.gz", f.read())
    print("voices/bart gz:", os.path.getsize(f"{out}/voices_en.npz.gz"),
          os.path.getsize(f"{out}/bart.safetensors.gz"))

    # a truncated write here silently degrades every pronunciation, so check the sizes
    for name, minimum in (("lexicon.tsv.gz", 1_000_000), ("voices_en.npz.gz", 3_000_000),
                          ("bart.safetensors.gz", 1_000_000)):
        size = os.path.getsize(f"{out}/{name}")
        assert size >= minimum, f"{name} is {size} bytes, expected >= {minimum}"

    if a.golden_dir:
        os.makedirs(f"{out}/golden", exist_ok=True)
        for fn in FIXTURES:
            src = os.path.join(a.golden_dir, fn)
            if os.path.isfile(src):
                shutil.copyfile(src, f"{out}/golden/{fn}")
            else:
                print("golden fixture missing, skipped:", src)
    print("assets written to", out)


if __name__ == "__main__":
    main()
