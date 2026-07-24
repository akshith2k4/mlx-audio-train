#!/usr/bin/env python3
"""
prepare_audio.py — Validate & convert WAV clips to correct format.
Ensures: Mono, 24000Hz, 16-bit PCM, 3-16 seconds, silence-trimmed.

Usage:
    python scripts/prepare_audio.py --input_dir raw_clips/ --output_dir data/speaker/
"""

import argparse
import os
import glob
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 24000
MIN_DURATION = 1.0
MAX_DURATION = 16.0


def convert_clip(input_path, output_path):
    data, sr = sf.read(input_path, dtype="float32")

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g).astype(np.float32)

    # Trim silence (threshold -50 dB)
    threshold = 10 ** (-50 / 20)
    non_silent = np.where(np.abs(data) > threshold)[0]
    if len(non_silent) > 0:
        pad = int(0.1 * TARGET_SR)
        start = max(0, non_silent[0] - pad)
        end = min(len(data), non_silent[-1] + pad)
        data = data[start:end]

    # Normalize to -1 dB peak
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * (10 ** (-1 / 20))

    duration = len(data) / TARGET_SR
    sf.write(output_path, data, TARGET_SR, subtype="PCM_16")
    return duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="raw_clips")
    parser.add_argument("--output_dir", default="data/speaker")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
        files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    files.sort(key=lambda x: int("".join(filter(str.isdigit, os.path.basename(x))) or 0))

    if not files:
        print(f"❌ No audio files in {args.input_dir}/")
        return

    print(f"Found {len(files)} files. Processing...\n")
    good, rejected = [], []

    total_dur = 0.0
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        out_name = f"clip{i+1}.wav"
        out_path = os.path.join(args.output_dir, out_name)

        try:
            dur = convert_clip(fpath, out_path)
            if dur < MIN_DURATION:
                rejected.append((fname, f"too short ({dur:.1f}s)"))
                os.remove(out_path)
            elif dur > MAX_DURATION:
                rejected.append((fname, f"too long ({dur:.1f}s)"))
                os.remove(out_path)
            else:
                good.append(out_name)
                total_dur += dur
                print(f"  ✅ {fname} → {out_name} ({dur:.1f}s)")
        except Exception as e:
            rejected.append((fname, str(e)))
            print(f"  ❌ {fname}: {e}")

    print(f"\n{'='*50}")
    print(f"✅ Good: {len(good)}  |  ❌ Rejected: {len(rejected)}")
    for name, reason in rejected:
        print(f"   {name}: {reason}")
    print(f"\n📊 Total Duration: {total_dur:.2f}s ({total_dur/60:.2f} min / {int(total_dur//60)}m {total_dur%60:.1f}s). {'✅ Enough clips!' if len(good)>=80 else '⚠️ Record more clips.'}")

    with open(os.path.join(args.output_dir, "clip_list.txt"), "w") as f:
        f.write("\n".join(good))


if __name__ == "__main__":
    main()
