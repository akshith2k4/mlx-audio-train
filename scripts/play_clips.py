#!/usr/bin/env python3
"""
play_clips.py — Play all audio clips sequentially while displaying Telugu transcripts.

Usage:
    python scripts/play_clips.py
    python scripts/play_clips.py --delay 1.0     # 1 second pause between clips
    python scripts/play_clips.py --interactive  # Press Enter for next clip
"""

import os
import sys
import time
import argparse
import subprocess
import soundfile as sf

def load_transcripts(transcripts_file):
    transcripts = {}
    if os.path.exists(transcripts_file):
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    fname, text = line.split('\t', 1)
                    transcripts[fname] = text
    return transcripts

def main():
    parser = argparse.ArgumentParser(description="Play audio clips sequentially with transcripts.")
    parser.add_argument("--dir", default="data/speaker", help="Directory containing audio clips")
    parser.add_argument("--transcripts", default="data/speaker/transcripts.txt", help="Transcripts text file")
    parser.add_argument("--delay", type=float, default=0.5, help="Pause duration between clips in seconds")
    parser.add_argument("--interactive", action="store_true", help="Wait for Enter key press between clips")
    args = parser.parse_args()

    transcripts = load_transcripts(args.transcripts)

    # Collect and sort clips
    wav_files = []
    for i in range(1, 41):
        fname = f"clip{i}.wav"
        fpath = os.path.join(args.dir, fname)
        if os.path.exists(fpath):
            wav_files.append((i, fname, fpath))

    if not wav_files:
        print(f"❌ No clip files found in '{args.dir}'")
        sys.exit(1)

    print(f"\n==================================================")
    print(f" 🎙️  Playing {len(wav_files)} Clips Sequentially")
    print(f" Mode: {'Interactive (Press Enter)' if args.interactive else f'Auto-Play ({args.delay}s pause)'}")
    print(f"==================================================\n")

    for i, fname, fpath in wav_files:
        text = transcripts.get(fname, "No transcript available")
        
        # Get duration
        data, sr = sf.read(fpath)
        duration = len(data) / sr

        print(f"[{i:02d}/40] 🔊 {fname} ({duration:.2f}s)")
        print(f"       💬 {text}\n")

        # Play audio using macOS afplay
        try:
            subprocess.run(["afplay", fpath], check=True)
        except Exception as e:
            print(f"       ❌ Error playing audio: {e}")

        if args.interactive:
            input("       --> Press Enter for next clip (or Ctrl+C to stop)... ")
            print()
        else:
            time.sleep(args.delay)

    print("==================================================")
    print(" ✅ All clips played successfully!")
    print("==================================================\n")

if __name__ == "__main__":
    main()
