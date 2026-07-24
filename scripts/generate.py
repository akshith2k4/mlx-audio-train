#!/usr/bin/env python3
"""
generate.py — Generate Telugu speech in your cloned voice.

Usage:
    python scripts/generate.py \
        --model checkpoints/qwen3-speaker/my_telugu_voice_model \
        --text "నమస్కారం! ఇది నా వాయిస్." \
        --output output.wav
"""

import argparse
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/qwen3-speaker/my_telugu_voice_model")
    parser.add_argument("--text", default="నమస్కారం! ఇది నా వాయిస్. నా Mac లో రన్ అవుతోంది.")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--ref_audio", default="data/speaker/ref.wav")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    print(f"🔊 Generating speech...")
    print(f"   Model:  {args.model}")
    print(f"   Text:   {args.text}")
    print(f"   Output: {args.output}")

    # Method 1: mlx_audio API
    try:
        from mlx_audio.tts import load_model, generate
        import soundfile as sf

        model_path = args.model
        if not os.path.exists(os.path.join(model_path, "config.json")):
            model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

        model = load_model(model_path)

        # Apply trained weights if available
        weights_path = os.path.join(args.model, "weights.npz")
        if os.path.exists(weights_path):
            import mlx.core as mx
            from mlx.utils import tree_unflatten
            loaded = dict(np.load(weights_path, allow_pickle=True))
            model.update(tree_unflatten([(k, mx.array(v)) for k, v in loaded.items()]))
            print("   Applied fine-tuned speaker weights!")

        audio_result = generate(
            model,
            text=args.text,
            ref_audio=args.ref_audio if os.path.exists(args.ref_audio) else None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        audio_chunks = []
        if hasattr(audio_result, "__iter__") and not isinstance(audio_result, (np.ndarray, list)):
            for chunk in audio_result:
                if hasattr(chunk, "audio"):
                    audio_chunks.append(np.array(chunk.audio, dtype=np.float32))
                elif isinstance(chunk, (np.ndarray, list)):
                    audio_chunks.append(np.array(chunk, dtype=np.float32))
        elif isinstance(audio_result, (np.ndarray, list)):
            audio_chunks.append(np.array(audio_result, dtype=np.float32))
        elif hasattr(audio_result, "audio"):
            audio_chunks.append(np.array(audio_result.audio, dtype=np.float32))

        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            sf.write(args.output, final_audio, 24000)
            print(f"\n✅ Saved generated speech to: {args.output} ({len(final_audio)/24000:.2f}s)")
        else:
            print(f"\n⚠️  No audio samples generated.")
        return

    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  mlx_audio generate failed: {e}")

    # Method 2: mlx_lm fallback
    try:
        import mlx_lm
        import mlx.core as mx
        import soundfile as sf

        model, tokenizer = mlx_lm.load(args.model)

        prompt = f"Generate Telugu speech: {args.text}"
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temp=args.temperature,
        )

        print(f"\n📝 Model output: {response[:200]}")
        print(f"\n⚠️  mlx_lm generates text, not audio directly.")
        print(f"   For audio output, ensure mlx_audio is installed:")
        print(f"   pip install mlx-audio")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. pip install mlx-audio soundfile scipy")
        print(f"  2. Ensure model path exists: {args.model}")
        print(f"  3. Check ref_audio exists: {args.ref_audio}")


if __name__ == "__main__":
    main()
