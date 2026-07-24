#!/usr/bin/env python3
"""
make_jsonl.py — Build train.jsonl from corrected transcripts.
Usage:
    python scripts/make_jsonl.py --clips_dir data/speaker --transcripts data/speaker/transcripts.txt --ref data/speaker/ref.wav
"""

import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", default="data/speaker")
    parser.add_argument("--transcripts", default="data/speaker/transcripts.txt")
    parser.add_argument("--ref", default="data/speaker/ref.wav")
    parser.add_argument("--output", default="data/speaker/train.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"❌ ref.wav not found: {args.ref}"); return

    lines = []
    with open(args.transcripts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"  ⚠️  Skipping: {line[:60]}"); continue
            fname, text = parts[0].strip(), parts[1].strip()
            if not os.path.exists(os.path.join(args.clips_dir, fname)):
                print(f"  ⚠️  Missing: {fname}"); continue
            if not text:
                continue
            entry = {
                "audio": f"data/speaker/{fname}",
                "text": text,
                "ref_audio": args.ref,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ {len(lines)} entries → {args.output}")

if __name__ == "__main__":
    main()
