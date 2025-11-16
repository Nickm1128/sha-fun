import os
import argparse
import time
import numpy as np
import tensorflow as tf

from models import build_sine_state_predictor


def make_datasets(X, Y, batch_size=256, val_frac=0.1, seed=42):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_frac))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]
    ds_tr = tf.data.Dataset.from_tensor_slices((Xtr, Ytr)).shuffle(len(tr_idx), seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_va = tf.data.Dataset.from_tensor_slices((Xva, Yva)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join("data", "state_dataset.npz"))
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,256,256")
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Generate it with GenerateStateDataset().")

    npz = np.load(data_path)
    X = npz["X"]
    Y = npz["Y"]
    if X.ndim != 2:
        raise ValueError(f"X should be 2D (N,D). Got shape {X.shape}")
    if Y.ndim != 3 or Y.shape[1:] != (64, 8):
        raise ValueError(f"Y should be 3D (N,64,8). Got shape {Y.shape}")

    input_dim = X.shape[1]
    hidden = tuple(int(h) for h in args.hidden.split(",") if h.strip())

    ds_tr, ds_va = make_datasets(X, Y, batch_size=args.batch_size, val_frac=args.val_frac, seed=args.seed)

    model = build_sine_state_predictor(input_dim=input_dim, hidden=hidden, w0=args.w0, dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mae", metrics=["mae"])

    tstamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir, f"predictor-{tstamp}")
    os.makedirs(out_dir, exist_ok=True)

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "best.keras"), monitor="val_mae", save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "log.csv")),
    ]

    hist = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=cbs)
    val_mae = min(hist.history.get("val_mae", [float("inf")]))
    print(f"Best val MAE: {val_mae:.6f}")

    # Save final model
    model.save(os.path.join(out_dir, "final.keras"))
    print(f"Saved artifacts in {out_dir}")


if __name__ == "__main__":
    main()

