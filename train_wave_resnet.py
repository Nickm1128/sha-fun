# pylint: disable=missing-function-docstring
import argparse
import json
import os
import time
import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from psann import WaveResNetRegressor
from data_generator import GenerateEvolutionMapDataset


def relative_mae_loss(pred, target, eps: float = 1e-6):
    """
    Custom loss: mean over samples of MAE(pred_delta, true_delta) / MAE(true_delta, 0).
    """
    if pred.shape != target.shape:
        target = target.view(target.shape[0], -1)
        pred = pred.view(pred.shape[0], -1)

    reduce_axes = tuple(range(1, pred.ndim))
    mae_out = (pred - target).abs().mean(dim=reduce_axes)
    baseline = target.abs().mean(dim=reduce_axes)
    return torch.mean(mae_out / (baseline + eps))


def build_dataset(args):
    alphabet = args.alphabet if args.alphabet else None
    save_flag = bool(args.save_dataset)
    if save_flag:
        out_path = args.save_dataset
    else:
        out_path = os.path.join(args.out_dir, "generated_evolution_maps.npz")
    inputs, residuals, targets = GenerateEvolutionMapDataset(
        num_samples=int(args.num_samples),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        alphabet=alphabet,
        noise_std=float(args.noise_std),
        seed=int(args.seed),
        out_path=out_path,
        save=save_flag,
    )
    if inputs is None or residuals is None or targets is None:
        raise RuntimeError("Failed to generate any evolution map samples.")
    return inputs.astype(np.float32), residuals.astype(np.float32), targets.astype(np.float32)


def evaluate_refinement(estimator, inputs, clean_states):
    preds = estimator.predict(inputs)
    preds = preds.astype(np.float32, copy=False)
    if preds.ndim == 2:
        preds = preds.reshape((-1, 64, 8))
    noisy_states = inputs[..., :8, 0]
    refined = noisy_states + preds
    baseline = np.mean(np.abs(clean_states - noisy_states), axis=(1, 2))
    refined_mae = np.mean(np.abs(clean_states - refined), axis=(1, 2))
    ratios = refined_mae / np.maximum(baseline, 1e-8)
    return float(np.mean(ratios)), float(np.mean(baseline)), float(np.mean(refined_mae))


def refine_dataset_with_model(estimator, inputs, targets):
    """
    Apply a trained estimator once to build a new dataset:
      - new baseline = previous baseline + predicted_delta
      - new residual = clean - new baseline
    """
    preds = estimator.predict(inputs).astype(np.float32, copy=False)
    if preds.ndim == 2:
        preds = preds.reshape((-1, 64, 8))
    # channel 0 holds the baseline state grid; later channels (if any) are metadata (e.g., stage id)
    baseline = inputs[..., :8, 0]
    refined = baseline + preds
    refined = np.clip(refined, -1.0, 1.0)
    new_inputs = inputs.copy()
    new_inputs[..., :8, 0] = refined
    new_residuals = targets - refined
    return new_inputs, new_residuals, targets


def _encode_stage_embed(stage: int, stages_total: int, embed_dim: int = 4) -> np.ndarray:
    """
    Build a small embedding vector for the given stage index.
    Uses simple scalar + sinusoidal features; pads/truncates to embed_dim.
    """
    s = float(stage) / float(max(stages_total - 1, 1))
    feats = [s, math.sin(math.pi * s), math.cos(math.pi * s), s * s]
    if embed_dim <= len(feats):
        return np.asarray(feats[:embed_dim], dtype=np.float32)
    # pad with zeros if more dims requested
    feats.extend([0.0] * (embed_dim - len(feats)))
    return np.asarray(feats, dtype=np.float32)


def _set_stage_channel(inputs: np.ndarray, stage: int, stages_total: int, embed_dim: int = 1) -> np.ndarray:
    """
    Ensure inputs carry a stage indicator/embedding as extra channels. baseline stays in channel 0.
    Stage value is normalized to [0,1] and optionally expanded with sinusoidal features.
    """
    stage_vec = _encode_stage_embed(stage, stages_total, embed_dim=embed_dim)
    # Broadcast per-sample stage embedding across spatial dims
    stage_planes = np.tile(stage_vec, (inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))
    return np.concatenate([inputs, stage_planes.astype(inputs.dtype)], axis=-1)


def build_estimator(args):
    # psann 0.10.x does not accept an attention kwarg; newer versions might.
    # Probe the signature at runtime to stay compatible.
    import inspect

    supports_attention = "attention" in inspect.signature(WaveResNetRegressor.with_conv_stem).parameters
    attention = None
    if supports_attention and args.attention_heads > 0:
        attention = {"kind": "mha", "num_heads": int(args.attention_heads)}

    return WaveResNetRegressor.with_conv_stem(
        conv_channels=int(args.conv_channels),
        conv_kernel_size=int(args.kernel_size),
        data_format="channels_last",
        hidden_layers=int(args.hidden_layers),
        hidden_units=int(args.hidden_units),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        loss=relative_mae_loss,
        random_state=int(args.seed),
        device=args.device,
        output_shape=(64, 8),
        early_stopping=bool(args.early_stopping),
        patience=int(args.patience),
        **({"attention": attention} if supports_attention else {}),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Train a WaveResNet refiner on SHA-256 evolution maps.")
    ap.add_argument("--out_dir", type=str, default=os.path.join("runs", "wave_resnet"))
    ap.add_argument("--stages", type=int, default=1, help="Number of iterative refinement stages to train.")
    ap.add_argument("--stage_embed_dim", type=int, default=4, help="Embedding dims for stage conditioning channels.")
    ap.add_argument("--cache_dataset", type=str, default="", help="Optional .npz to load/save generated stage-0 dataset.")
    ap.add_argument("--pooled_after", action="store_true", help="Also train a pooled final model on all stage datasets.")
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--min_len", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--noise_std", type=float, default=0.2)
    ap.add_argument("--alphabet", type=str, default="")
    ap.add_argument("--save_dataset", type=str, default="", help="Optional path to persist the generated dataset.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--conv_channels", type=int, default=32)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--hidden_layers", type=int, default=4)
    ap.add_argument("--hidden_units", type=int, default=96)
    ap.add_argument("--attention_heads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--verbose", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"wave_resnet_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Optionally load cached stage-0 dataset to speed up iterations.
    if args.cache_dataset and os.path.isfile(args.cache_dataset):
        with np.load(args.cache_dataset) as data:
            inputs, residuals, targets = data["inputs"], data["residuals"], data["targets"]
        if args.verbose:
            print(f"Loaded cached dataset from {args.cache_dataset} (inputs {inputs.shape})")
    else:
        inputs, residuals, targets = build_dataset(args)
        if args.cache_dataset:
            np.savez_compressed(args.cache_dataset, inputs=inputs, residuals=residuals, targets=targets)
            if args.verbose:
                print(f"Cached stage-0 dataset to {args.cache_dataset}")

    estimator = None
    stage_metrics = []
    pooled_inputs = []
    pooled_residuals = []
    pooled_targets = []

    for stage in range(int(args.stages)):
        if stage > 0:
            if args.verbose:
                print(f"\n--- Building dataset for stage {stage} from previous model ---")
            inputs, residuals, targets = refine_dataset_with_model(estimator, inputs, targets)
        # Tag inputs with the current stage embedding so the model can condition on refinement depth.
        inputs = _set_stage_channel(
            inputs, stage=stage, stages_total=int(args.stages), embed_dim=int(args.stage_embed_dim)
        )

        if args.pooled_after:
            pooled_inputs.append(inputs)
            pooled_residuals.append(residuals)
            pooled_targets.append(targets)

        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
            inputs,
            residuals,
            targets,
            test_size=args.val_frac,
            random_state=args.seed,
            shuffle=True,
        )

        estimator = build_estimator(args)

        if args.verbose:
            print(
                f"\n=== Stage {stage} ===\n"
                f"Training WaveResNet on {X_train.shape[0]} samples "
                f"(val {X_val.shape[0]}) with lr={args.lr}, epochs={args.epochs}."
            )

        estimator.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            verbose=int(args.verbose),
        )

        val_ratio, baseline_mae, refined_mae = evaluate_refinement(estimator, X_val, t_val)
        if args.verbose:
            print(
                f"Stage {stage} validation relative MAE: {val_ratio:.4f} "
                f"(baseline {baseline_mae:.4f} -> refined {refined_mae:.4f})"
            )

        ckpt_path = os.path.join(run_dir, f"estimator_stage{stage}.pt")
        estimator.save(ckpt_path)
        stage_metrics.append(
            {
                "stage": int(stage),
                "val_ratio": val_ratio,
                "baseline_mae": baseline_mae,
                "refined_mae": refined_mae,
                "checkpoint": os.path.basename(ckpt_path),
            }
        )

    # Alias final stage checkpoint for convenience
    final_ckpt = os.path.join(run_dir, "estimator.pt")
    estimator.save(final_ckpt)

    metrics = {
        "stages": stage_metrics,
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    if args.verbose:
        print(f"\nSaved multi-stage checkpoints and metrics in {run_dir}")

    # Optional pooled final model across all stages
    if args.pooled_after and pooled_inputs:
        if args.verbose:
            print("\n=== Training pooled model across all stages ===")
        pooled_inputs_arr = np.concatenate(pooled_inputs, axis=0)
        pooled_residuals_arr = np.concatenate(pooled_residuals, axis=0)
        pooled_targets_arr = np.concatenate(pooled_targets, axis=0)

        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
            pooled_inputs_arr,
            pooled_residuals_arr,
            pooled_targets_arr,
            test_size=args.val_frac,
            random_state=args.seed,
            shuffle=True,
        )

        pooled_estimator = build_estimator(args)
        pooled_estimator.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            verbose=int(args.verbose),
        )
        val_ratio, baseline_mae, refined_mae = evaluate_refinement(pooled_estimator, X_val, t_val)
        pooled_ckpt = os.path.join(run_dir, "estimator_pooled.pt")
        pooled_estimator.save(pooled_ckpt)
        pooled_metrics = {
            "stage": "pooled",
            "val_ratio": val_ratio,
            "baseline_mae": baseline_mae,
            "refined_mae": refined_mae,
            "checkpoint": os.path.basename(pooled_ckpt),
        }
        stage_metrics.append(pooled_metrics)
        metrics["stages"] = stage_metrics
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        if args.verbose:
            print(
                f"Pooled model validation relative MAE: {val_ratio:.4f} "
                f"(baseline {baseline_mae:.4f} -> refined {refined_mae:.4f})"
            )


if __name__ == "__main__":
    main()
