#!/usr/bin/env python3
"""
transcribe_clips.py — Draft Telugu transcription via Whisper.
Usage:
    python scripts/transcribe_clips.py --input_dir data/speaker --output data/speaker/transcripts.txt
"""

import argparse, os, glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/speaker")
    parser.add_argument("--output", default="data/speaker/transcripts.txt")
    parser.add_argument("--model_size", default="large-v3")
    args = parser.parse_args()

    try:
        import whisper
    except ImportError:
        os.system("pip install openai-whisper")
        import whisper

    print(f"Loading Whisper {args.model_size}...")
    model = whisper.load_model(args.model_size)

    wavs = sorted(
        glob.glob(os.path.join(args.input_dir, "clip*.wav")),
        key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))) or 0),
    )
    if not wavs:
        print(f"No clip*.wav in {args.input_dir}"); return

    print(f"Transcribing {len(wavs)} clips...\n")
    lines = []
    for i, w in enumerate(wavs):
        fname = os.path.basename(w)
        print(f"  [{i+1}/{len(wavs)}] {fname}...", end=" ", flush=True)
        result = model.transcribe(w, language="te", task="transcribe")
        text = result["text"].strip()
        lines.append(f"{fname}\t{text}")
        print(f"→ {text}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n✅ Draft → {args.output}")
    print("⚠️  OPEN IT AND CORRECT EVERY TELUGU WORD. Non-negotiable.")

if __name__ == "__main__":
    main()
