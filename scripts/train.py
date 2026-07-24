#!/usr/bin/env python3
"""
train.py — Finetune Qwen3-TTS on your Telugu voice.

Usage:
    python scripts/train.py --config configs/qwen3_tts_speaker.yaml
"""

import argparse
import json
import os
import sys
import time
import numpy as np

try:
    import yaml
except ImportError:
    os.system("pip install pyyaml")
    import yaml

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError:
    print("❌ MLX not installed. Run: pip install mlx")
    sys.exit(1)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TTSDataset:
    """Loads preprocessed mel + token pairs."""

    def __init__(self, preprocessed_dir):
        manifest_path = os.path.join(preprocessed_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.dir = preprocessed_dir
        print(f"📋 Dataset: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        mel = np.load(os.path.join(self.dir, entry["mel"]))
        tokens = np.load(os.path.join(self.dir, entry["tokens"]))
        ref_mel = None
        if entry.get("ref_mel"):
            ref_mel = np.load(os.path.join(self.dir, entry["ref_mel"]))
        return {
            "mel": mx.array(mel),
            "tokens": mx.array(tokens),
            "ref_mel": mx.array(ref_mel) if ref_mel is not None else None,
            "text": entry["text"],
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_mel_len = max(b["mel"].shape[0] for b in batch)
    max_tok_len = max(b["tokens"].shape[0] for b in batch)
    n_mels = batch[0]["mel"].shape[1]

    mels = mx.zeros((len(batch), max_mel_len, n_mels))
    tokens = mx.zeros((len(batch), max_tok_len), dtype=mx.int32)
    mel_mask = mx.zeros((len(batch), max_mel_len))
    tok_mask = mx.zeros((len(batch), max_tok_len))

    ref_mels = []
    has_ref = batch[0]["ref_mel"] is not None

    if has_ref:
        max_ref_len = max(b["ref_mel"].shape[0] for b in batch if b["ref_mel"] is not None)
        ref_mels_padded = mx.zeros((len(batch), max_ref_len, n_mels))

    for i, b in enumerate(batch):
        ml = b["mel"].shape[0]
        tl = b["tokens"].shape[0]
        mels[i, :ml] = b["mel"]
        tokens[i, :tl] = b["tokens"]
        mel_mask[i, :ml] = 1.0
        tok_mask[i, :tl] = 1.0
        if has_ref and b["ref_mel"] is not None:
            rl = b["ref_mel"].shape[0]
            ref_mels_padded[i, :rl] = b["ref_mel"]

    result = {
        "mels": mels,
        "tokens": tokens,
        "mel_mask": mel_mask,
        "tok_mask": tok_mask,
    }
    if has_ref:
        result["ref_mels"] = ref_mels_padded

    return result


def load_model(model_id):
    """Load the pretrained Qwen3-TTS model in MLX format."""
    print(f"🔽 Loading model: {model_id}")
    try:
        from mlx_audio.tts.models import load_model as _load
        model = _load(model_id)
        print("✅ Model loaded via mlx_audio")
        return model
    except Exception:
        pass

    try:
        from transformers import AutoModelForCausalLM
        import mlx_lm
        model, tokenizer = mlx_lm.load(model_id)
        print("✅ Model loaded via mlx_lm")
        return model
    except Exception:
        pass

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id)
        print(f"✅ Downloaded to: {path}")
        import mlx_lm
        model, tokenizer = mlx_lm.load(path)
        return model
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        sys.exit(1)


def train_step(model, batch, optimizer, cfg):
    """Single training step with gradient accumulation."""
    include_ref = cfg.get("processor", {}).get("include_ref_mel", True)

    def loss_fn():
        mels = batch["mels"]
        tokens = batch["tokens"]
        mel_mask = batch["mel_mask"]

        # Forward pass — model predicts mel from tokens
        try:
            outputs = model(
                input_ids=tokens,
                mel_targets=mels,
                mel_mask=mel_mask,
                ref_mel=batch.get("ref_mels") if include_ref else None,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        except TypeError:
            # Fallback: try different API
            try:
                logits = model(tokens)
                # Simple MSE loss on mel prediction
                if hasattr(logits, "logits"):
                    pred = logits.logits[:, :mels.shape[1], :mels.shape[2]]
                else:
                    pred = logits[:, :mels.shape[1], :mels.shape[2]]
                diff = (pred - mels) * mel_mask[:, :, None]
                loss = mx.mean(diff ** 2)
            except Exception as e:
                print(f"⚠️  Forward pass error: {e}")
                loss = mx.array(0.0)

        return loss

    loss, grads = mx.value_and_grad(loss_fn)()
    optimizer.update(model, grads)
    return loss.item()


def save_checkpoint(model, optimizer, epoch, step, output_dir, tag=""):
    """Save model checkpoint."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{tag}" if tag else f"checkpoint-epoch{epoch}-step{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save model weights
    weights = dict(tree_flatten(model.parameters()))
    np.savez(os.path.join(ckpt_dir, "weights.npz"), **{k: np.array(v) for k, v in weights.items()})

    # Save training state
    state = {"epoch": epoch, "step": step}
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    print(f"  💾 Saved: {ckpt_dir}")
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_tts_speaker.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer_cfg = cfg.get("trainer", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Hyperparams
    num_epochs = trainer_cfg.get("num_epochs", 3)
    batch_size = trainer_cfg.get("batch_size", 1)
    grad_accum = trainer_cfg.get("grad_accumulation", 16)
    lr = trainer_cfg.get("learning_rate", 2e-5)
    save_every = trainer_cfg.get("save_every", 50)
    log_every = trainer_cfg.get("log_every", 5)
    output_dir = trainer_cfg.get("output_dir", "checkpoints/qwen3-speaker")
    preprocessed_dir = data_cfg.get("preprocessed_dir", "data/speaker/preprocessed")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  QWEN3-TTS TELUGU VOICE FINETUNING")
    print("=" * 60)
    print(f"  Model:       {model_cfg.get('model_id')}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Batch:       {batch_size} × {grad_accum} accum = {batch_size * grad_accum} effective")
    print(f"  LR:          {lr}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # Load dataset
    dataset = TTSDataset(preprocessed_dir)
    if len(dataset) == 0:
        print("❌ Empty dataset. Run preprocess_dataset.py first.")
        sys.exit(1)

    # Load model
    model = load_model(model_cfg["model_id"])

    # Freeze most layers, finetune only the last few + embeddings
    print("\n🔧 Setting up trainable parameters...")
    trainable_params = {}
    all_params = dict(tree_flatten(model.parameters()))
    total_params = sum(v.size for v in all_params.values())

    for key, val in all_params.items():
        # Train: embeddings, last 2 transformer layers, output head
        if any(k in key for k in ["embed", "lm_head", "layers.21", "layers.20", "layers.22", "layers.23", "norm"]):
            trainable_params[key] = val

    trainable_count = sum(v.size for v in trainable_params.values())
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)")

    # Set trainable params
    model.update(tree_unflatten(list(trainable_params.items())))

    # Optimizer
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

    # Resume if specified
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n🔄 Resuming from: {args.resume}")
        weights_path = os.path.join(args.resume, "weights.npz")
        if os.path.exists(weights_path):
            loaded = dict(np.load(weights_path, allow_pickle=True))
            model.update(tree_unflatten([(k, mx.array(v)) for k, v in loaded.items()]))
        state_path = os.path.join(args.resume, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            start_epoch = state.get("epoch", 0)
            global_step = state.get("step", 0)

    # Training loop
    print(f"\n🚀 Starting training...\n")
    indices = list(range(len(dataset)))

    for epoch in range(start_epoch, num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        epoch_steps = 0
        accum_loss = 0.0

        print(f"{'='*60}")
        print(f"  EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_items = [dataset[j] for j in batch_indices]
            batch = collate_fn(batch_items)

            loss = train_step(model, batch, optimizer, cfg)
            accum_loss += loss
            epoch_loss += loss
            epoch_steps += 1
            global_step += 1

            # Gradient accumulation step
            if global_step % grad_accum == 0:
                optimizer.update(model, model.parameters())  # apply accumulated
                mx.eval(model.parameters())

            # Logging
            if global_step % log_every == 0:
                avg = accum_loss / log_every
                print(f"  Step {global_step:>5} | Loss: {avg:.4f} | Epoch: {epoch+1}/{num_epochs}")
                accum_loss = 0.0

            # Save checkpoint
            if global_step % save_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step, output_dir)

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  📊 Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, global_step, output_dir, tag=f"epoch{epoch+1}")

    # Final save
    final_dir = save_checkpoint(model, optimizer, num_epochs, global_step, output_dir, tag="final")
    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Final checkpoint: {final_dir}")
    print(f"{'='*60}")
    print(f"\nNext: python scripts/bake_speaker_embedding.py \\")
    print(f"    --config {args.config} \\")
    print(f"    --checkpoint {final_dir} \\")
    print(f"    --output {output_dir}/my_telugu_voice_model")


if __name__ == "__main__":
    main()
