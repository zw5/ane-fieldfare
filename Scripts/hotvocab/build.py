#!/usr/bin/env python3
"""Builds Sources/ANEFieldfareHotpath/HotVocabulary/vocab-<V>.bin: token IDs
ranked by frequency, special and byte-fallback tokens first.

The runtime only guesses from this list and checks every guess against the
full head, so it changes speed, never output. The shipped list came from the
first 80 MB of theblackcat102/sharegpt-english (assistant turns, 9 of every 10
conversations) plus the Python standard library's sources.

usage: build.py <tokenizer.json> <out.bin> <text files or dirs...> [--count N]
needs: pip install tokenizers
"""
import argparse, collections, json, os, struct
from tokenizers import Tokenizer


def texts(paths):
    for root in paths:
        files = [root] if os.path.isfile(root) else [
            os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
        for path in files:
            if path.endswith(".jsonl"):  # ShareGPT layout: assistant turns only
                for line in open(path, errors="ignore"):
                    try:
                        turns = json.loads(line)["conversations"]
                    except (ValueError, KeyError):
                        continue
                    yield from (t["text"] for t in turns if t.get("user") == "gpt")
            elif path.endswith((".py", ".txt", ".md")):
                yield open(path, errors="ignore").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("out")
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--count", type=int, default=65536)
    args = ap.parse_args()
    spec = json.load(open(args.tokenizer))
    vocab = len(spec["model"]["vocab"])
    tok = Tokenizer.from_file(args.tokenizer)
    counts = collections.Counter()
    batch = []
    for text in texts(args.corpus):
        batch.append(text)
        if len(batch) == 1024:
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                counts.update(enc.ids)
            batch = []
    for enc in tok.encode_batch(batch, add_special_tokens=False):
        counts.update(enc.ids)
    first = [t["id"] for t in spec["added_tokens"] if t["id"] < vocab]
    first += [i for k, i in spec["model"]["vocab"].items() if k.startswith("<0x") and len(k) == 6]
    ranked, seen = [], set()
    for i in first + [i for i, _ in counts.most_common()]:
        if i not in seen and 0 <= i < vocab:
            seen.add(i)
            ranked.append(i)
    ranked = ranked[:args.count]
    with open(args.out, "wb") as f:
        f.write(b"GFHOTV1\0" + struct.pack("<II", vocab, len(ranked)) + struct.pack(f"<{len(ranked)}I", *ranked))
    print(f"{args.out}: {len(ranked)} of {vocab} tokens")


if __name__ == "__main__":
    main()
