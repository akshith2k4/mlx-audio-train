#!/usr/bin/env python3
"""
preprocess_dataset.py — Convert WAV+text into mel spectrograms + token IDs.
Saves RAM during training by doing all heavy I/O upfront.

Usage:
    python scripts/preprocess_dataset.py --input data/speaker/train.jsonl
"""

import argparse
import json
import os
import sys
import numpy as np
import soundfile as sf

try:
    import yaml
except ImportError:
    os.system("pip install pyyaml")
    import yaml


def load_config(path="configs/qwen3_tts_speaker.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_mel(audio_path, sr=24000, n_fft=1024, hop=256, n_mels=80):
    """Extract log-mel spectrogram from audio file."""
    data, file_sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(file_sr, sr)
        data = resample_poly(data, sr // g, file_sr // g).astype(np.float32)

    # Pad if too short
    min_len = n_fft
    if len(data) < min_len:
        data = np.pad(data, (0, min_len - len(data)))

    # STFT
    from scipy.signal import stft
    _, _, Zxx = stft(data, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    mag = np.abs(Zxx)

    # Mel filterbank
    try:
        import librosa
        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    except ImportError:
        # Fallback: simple linear mel approximation
        freq_bins = n_fft // 2 + 1
        mel_fb = np.zeros((n_mels, freq_bins), dtype=np.float32)
        mel_points = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        for i in range(n_mels):
            for j in range(bin_points[i], bin_points[i + 1]):
                if j < freq_bins:
                    mel_fb[i, j] = (j - bin_points[i]) / max(1, bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], bin_points[i + 2]):
                if j < freq_bins:
                    mel_fb[i, j] = (bin_points[i + 2] - j) / max(1, bin_points[i + 2] - bin_points[i + 1])

    mel = mel_fb @ mag
    log_mel = np.log(np.clip(mel, 1e-5, None))
    return log_mel.T.astype(np.float32)  # (T, n_mels)


def tokenize_text(text, tokenizer):
    """Tokenize text using the model's tokenizer."""
    tokens = tokenizer.encode(text)
    return np.array(tokens, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/speaker/train.jsonl")
    parser.add_argument("--config", default="configs/qwen3_tts_speaker.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sr = cfg.get("processor", {}).get("sample_rate", 24000)
    out_dir = cfg.get("data", {}).get("preprocessed_dir", "data/speaker/preprocessed")
    os.makedirs(out_dir, exist_ok=True)

    # Load entries
    entries = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"📋 {len(entries)} entries to preprocess")

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        from transformers import AutoTokenizer
        model_id = cfg["model"]["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️  Could not load tokenizer: {e}")
        print("   Falling back to character-level tokenization.")
        tokenizer = None

    # Process each entry
    manifest = []
    total_audio_dur = 0.0

    for i, entry in enumerate(entries):
        audio_path = entry["audio"]
        text = entry["text"]
        ref_path = entry.get("ref_audio", "")

        if not os.path.exists(audio_path):
            print(f"  ❌ [{i+1}] Missing: {audio_path}")
            continue

        try:
            # Extract mel for target audio
            mel = extract_mel(audio_path, sr=sr)
            audio_data, _ = sf.read(audio_path, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            dur = len(audio_data) / sr
            total_audio_dur += dur

            # Tokenize text
            if tokenizer:
                tokens = tokenize_text(text, tokenizer)
            else:
                tokens = np.array([ord(c) for c in text], dtype=np.int32)

            # Extract ref mel if needed
            ref_mel = None
            if ref_path and os.path.exists(ref_path):
                ref_mel = extract_mel(ref_path, sr=sr)

            # Save
            np.save(os.path.join(out_dir, f"mel_{i:04d}.npy"), mel)
            np.save(os.path.join(out_dir, f"tokens_{i:04d}.npy"), tokens)
            if ref_mel is not None:
                np.save(os.path.join(out_dir, f"ref_mel_{i:04d}.npy"), ref_mel)

            manifest.append({
                "index": i,
                "mel": f"mel_{i:04d}.npy",
                "tokens": f"tokens_{i:04d}.npy",
                "ref_mel": f"ref_mel_{i:04d}.npy" if ref_mel is not None else None,
                "text": text,
                "duration": round(dur, 2),
            })

            print(f"  ✅ [{i+1}/{len(entries)}] {os.path.basename(audio_path)} → mel({mel.shape[0]} frames) + {len(tokens)} tokens ({dur:.1f}s)")

        except Exception as e:
            print(f"  ❌ [{i+1}] {os.path.basename(audio_path)}: {e}")

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Preprocessed {len(manifest)}/{len(entries)} entries")
    print(f"📊 Total audio: {total_audio_dur/60:.1f} minutes")
    print(f"📁 Saved to: {out_dir}/")
    print(f"📋 Manifest: {manifest_path}")

    if total_audio_dur / 60 < 15:
        print(f"\n⚠️  Only {total_audio_dur/60:.1f} min. Recommend 20+ min for quality.")
    else:
        print(f"\n✅ Enough data for training. Proceed to train.py")


if __name__ == "__main__":
    main()
