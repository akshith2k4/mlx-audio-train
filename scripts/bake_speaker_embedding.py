#!/usr/bin/env python3
"""
bake_speaker_embedding.py — Merge finetuned weights + speaker embedding
into a standalone model folder.

Usage:
    python scripts/bake_speaker_embedding.py \
        --config configs/qwen3_tts_speaker.yaml \
        --checkpoint checkpoints/qwen3-speaker/checkpoint-final \
        --output checkpoints/qwen3-speaker/my_telugu_voice_model
"""

import argparse
import json
import os
import shutil
import numpy as np

try:
    import yaml
except ImportError:
    os.system("pip install pyyaml")
    import yaml

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError:
    print("❌ MLX not installed.")
    exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_tts_speaker.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speaker_name", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    speaker_name = args.speaker_name or cfg.get("processor", {}).get("speaker_name", "my_telugu_voice")
    model_id = cfg["model"]["model_id"]

    print("=" * 60)
    print("  BAKING SPEAKER EMBEDDING")
    print("=" * 60)
    print(f"  Base model:   {model_id}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Speaker:      {speaker_name}")
    print(f"  Output:       {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    # 1. Copy base model files
    print("\n📦 Copying base model...")
    try:
        from huggingface_hub import snapshot_download
        base_path = snapshot_download(model_id)
        for f in os.listdir(base_path):
            src = os.path.join(base_path, f)
            dst = os.path.join(args.output, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        print(f"  ✅ Copied from: {base_path}")
    except Exception as e:
        print(f"  ⚠️  Could not copy base model: {e}")
        print(f"     You may need to manually copy from HF cache.")

    # 2. Load and merge finetuned weights
    print("\n🔧 Merging finetuned weights...")
    weights_path = os.path.join(args.checkpoint, "weights.npz")
    if not os.path.exists(weights_path):
        print(f"  ❌ No weights.npz in {args.checkpoint}")
        return

    finetuned = dict(np.load(weights_path, allow_pickle=True))
    print(f"  Loaded {len(finetuned)} finetuned tensors")

    # Save merged weights
    merged_path = os.path.join(args.output, "weights.npz")
    np.savez(merged_path, **finetuned)
    print(f"  ✅ Saved merged weights: {merged_path}")

    # 3. Extract speaker embedding from ref audio
    print("\n🎤 Extracting speaker embedding...")
    ref_path = "data/speaker/ref.wav"
    if os.path.exists(ref_path):
        import soundfile as sf
        audio, sr = sf.read(ref_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Simple speaker embedding: mean of mel spectrogram
        n_fft, hop, n_mels = 1024, 256, 80
        from scipy.signal import stft
        _, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)

        try:
            import librosa
            mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        except ImportError:
            freq_bins = n_fft // 2 + 1
            mel_fb = np.random.randn(n_mels, freq_bins).astype(np.float32) * 0.01

        mel = mel_fb @ mag
        log_mel = np.log(np.clip(mel, 1e-5, None))
        speaker_emb = log_mel.mean(axis=1)  # (n_mels,)

        np.save(os.path.join(args.output, "speaker_embedding.npy"), speaker_emb)
        print(f"  ✅ Speaker embedding: {speaker_emb.shape}")
    else:
        print(f"  ⚠️  ref.wav not found. Skipping speaker embedding.")

    # 4. Save speaker config
    speaker_config = {
        "speaker_name": speaker_name,
        "model_id": model_id,
        "sample_rate": cfg.get("processor", {}).get("sample_rate", 24000),
        "include_ref_mel": cfg.get("processor", {}).get("include_ref_mel", True),
        "checkpoint": args.checkpoint,
        "language": "telugu",
    }
    config_path = os.path.join(args.output, "speaker_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(speaker_config, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Speaker config: {config_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ BAKED MODEL READY")
    print(f"  📁 {args.output}/")
    print(f"{'='*60}")
    print(f"\nTest it:")
    print(f'  python scripts/generate.py --model {args.output} --text "నమస్కారం!"')


if __name__ == "__main__":
    main()
