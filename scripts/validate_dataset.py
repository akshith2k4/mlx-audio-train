#!/usr/bin/env python3
"""
validate_dataset.py — Final check before training.
Usage:
    python scripts/validate_dataset.py --jsonl data/speaker/train.jsonl
"""

import json, os, sys
import soundfile as sf

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/speaker/train.jsonl")
    args = parser.parse_args()

    errors, warnings = [], []
    total_dur = 0.0

    entries = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                entries.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: bad JSON — {e}")

    print(f"📋 {len(entries)} entries\n")
    ref_ok = False

    for num, e in entries:
        audio = e.get("audio", "")
        text = e.get("text", "")
        ref = e.get("ref_audio", "")

        if not os.path.exists(audio):
            errors.append(f"Line {num}: missing {audio}"); continue
        if len(text.strip()) < 3:
            errors.append(f"Line {num}: text too short")

        if not ref_ok:
            if not os.path.exists(ref):
                errors.append(f"ref missing: {ref}")
            else:
                d, sr = sf.read(ref)
                dur = len(d) / sr
                if sr != 24000: warnings.append(f"ref.wav is {sr}Hz")
                if not (3 <= dur <= 15): warnings.append(f"ref.wav {dur:.1f}s")
                else: print(f"✅ ref.wav: {dur:.1f}s @ {sr}Hz")
            ref_ok = True

        try:
            d, sr = sf.read(audio)
            dur = len(d) / sr
            total_dur += dur
            if sr != 24000: warnings.append(f"Line {num}: {sr}Hz")
            if dur < 2: warnings.append(f"Line {num}: {dur:.1f}s short")
            if dur > 20: warnings.append(f"Line {num}: {dur:.1f}s long")
        except Exception as ex:
            errors.append(f"Line {num}: {ex}")

    print(f"\n{'='*50}")
    print(f"   Clips: {len(entries)}  |  Duration: {total_dur/60:.1f} min")
    print(f"   Errors: {len(errors)}  |  Warnings: {len(warnings)}")

    if total_dur / 60 < 15:
        warnings.append("Under 15 min — record more for quality.")
    for e in errors[:15]: print(f"  ❌ {e}")
    for w in warnings[:15]: print(f"  ⚠️  {w}")

    if not errors:
        print("\n✅ READY FOR TRAINING.")
    else:
        print("\n❌ FIX ERRORS FIRST."); sys.exit(1)

if __name__ == "__main__":
    main()
