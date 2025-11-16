import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from utils import (
    generate_random_string,
    prepare_ml_data_v3,
    message_to_first_block_states,
    hash_to_features,
    message_to_evolution_maps,
)

def GenerateData(num_samples=5000, save=True):
    min_len = 1
    max_len = 128 # Generate strings of random length up to 2 blocks typical max
    num_hash_avg_rounds_for_X = 16 # How many averaging rounds for the HASH features

    print(f"Generating {num_samples} ML data points with random messages...")
    dataset_X = []
    dataset_Y = []

    # Generate Data
    for i in range(num_samples):
        msg_len = random.randint(min_len, max_len)
        input_message = generate_random_string(msg_len)
        try:
            # Optional: Reduce print frequency for large sample sizes
            if (i + 1) % 100 == 0:
                 print(f"Processing message {i+1}/{num_samples} (length {msg_len})")

            X_features, Y_target = prepare_ml_data_v3(input_message, hash_avg_rounds=num_hash_avg_rounds_for_X)

            if X_features is not None and Y_target is not None:
                dataset_X.append(X_features)
                dataset_Y.append(Y_target)
            # else: # Reduce verbosity
            #     print(f"  Skipped generating data point for message: '{input_message[:30].replace(chr(10),' ').replace(chr(13),'')}...'")
        except Exception as e:
             print(f"  Unexpected error processing message '{input_message[:30]}...': {e}")


    print(f"\n--- Generation Summary ---")
    print(f"Successfully generated {len(dataset_X)} data points out of {num_samples} attempts.")

    # --- Train/Test Split and Model Evaluation ---
    if len(dataset_X) > 1 and len(dataset_X) == len(dataset_Y):
        print("\nSplitting data into Training and Testing sets...")
        X_matrix = np.array(dataset_X)
        Y_matrix = np.array(dataset_Y)

        if save:
            x = pd.DataFrame(X_matrix)
            y = pd.DataFrame(Y_matrix)

            x.to_csv("x.csv", index=False)
            y.to_csv("y.csv", index=False)

        return X_matrix, Y_matrix

    #     # Split data (e.g., 80% train, 20% test)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         X_matrix, Y_matrix, test_size=0.2, random_state=42
    #     )
    #     print(f"Training set size: {X_train.shape[0]} samples")
    #     print(f"Testing set size: {X_test.shape[0]} samples")

    #     print("\nTraining RandomForestRegressor model...")
    #     # Initialize the model (n_jobs=-1 uses all available CPU cores)
    #     # RandomForestRegressor inherently handles multi-output regression
    #     model = RandomForestRegressor(n_estimators=100, # Number of trees
    #                                   random_state=42,
    #                                   n_jobs=-1,
    #                                   max_depth=10,      # Limit tree depth to prevent overfitting
    #                                   min_samples_leaf=5 # Require more samples per leaf
    #                                   )

    #     # Train the model
    #     model.fit(X_train, y_train)
    #     print("Model training complete.")

    #     print("\nEvaluating model on the test set...")
    #     # Make predictions
    #     y_pred = model.predict(X_test)

    #     # Calculate metrics
    #     mae = mean_absolute_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred) # R-squared score

    #     print(f"\n--- Model Evaluation Results ---")
    #     print(f"Mean Absolute Error (MAE): {mae:.6f}")
    #     print(f"R-squared (R2 Score):    {r2:.6f}")

    #     print("\nInterpretation:")
    #     print(f"MAE indicates the average absolute difference between the predicted averaged word values and the actual averaged word values.")
    #     if mae < 0.5:
    #          print("MAE is less than 0.5, which was your indicator threshold.")
    #          print("However, consider if the model is simply predicting values close to the overall average (~0.5).")
    #     else:
    #          print("MAE is >= 0.5.")
    #     print(f"R2 Score indicates the proportion of variance in the target explained by the model.")
    #     print(f"(An R2 score close to 0 or negative suggests the model does not explain the variance well).")


    #     print("\nReminder: Even if MAE is low, this does NOT necessarily imply the model")
    #     print("has learned to reverse SHA-256. It might simply predict average values well.")
    #     print("The lack of information in the input features (derived only from the hash)")
    #     print("remains the fundamental challenge.")

    # elif num_samples > 0:
    #      print("\nNot enough data points generated successfully to train/test a model.")


def GenerateStateDataset(
    num_samples=10000,
    out_dir="data",
    min_len=8,
    max_len=20,
    alphabet=None,
    feature_mode="basic",
    save=True,
):
    """
    Generate dataset for Model 1 (H -> per-round states).

    X: hash features (shape: [N, D])
    Y: normalized states per round (shape: [N, 64, 8])
    """
    if alphabet is None:
        import string as _string
        alphabet = _string.ascii_letters + _string.digits + " "

    os.makedirs(out_dir, exist_ok=True)

    X_list = []
    Y_list = []

    print(f"Generating {num_samples} state samples (len {min_len}-{max_len})...")
    for i in range(num_samples):
        L = random.randint(min_len, max_len)
        msg = generate_random_string(L, char_set=alphabet)
        try:
            states_norm, final_h = message_to_first_block_states(msg)
            if states_norm is None or final_h is None:
                continue
            x = hash_to_features(final_h, mode=feature_mode)
            if x is None:
                continue
            X_list.append(x)
            Y_list.append(states_norm)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{num_samples} samples")
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")

    if not X_list:
        print("No samples generated.")
        return None, None

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    if save:
        np.savez_compressed(os.path.join(out_dir, "state_dataset.npz"), X=X, Y=Y)
        print(f"Saved to {os.path.join(out_dir, 'state_dataset.npz')} (X: {X.shape}, Y: {Y.shape})")

    return X, Y

def GenerateEvolutionMapDataset(
    num_samples=10000,
    out_path=os.path.join("data", "evolution_maps.npz"),
    min_len=8,
    max_len=20,
    alphabet=None,
    noise_std=0.2,
    seed=42,
    save=True,
):
    """
    Generate datasets for the WaveResNet refinement model.

    Stage-0 setup: the per-round state map is initialised to a flat 0.5
    everywhere, so the model learns the full residual from this neutral
    baseline towards the true states, without seeing the per-round W[t].

    Inputs (X): baseline per-round states only (shape: [N, 64, 8, 1])
                All 8 channels are fixed at 0.5 for stage 0.
    Targets (Y): residual map (ground truth - baseline) with shape [N, 64, 8].
    Extras:
        - clean_states saved alongside for evaluation.
    """
    if alphabet is None:
        import string as _string
        alphabet = _string.ascii_letters + _string.digits + " "

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    inputs = []
    residuals = []
    clean_states = []

    print(f"Generating {num_samples} evolution samples (len {min_len}-{max_len}) with flat baseline=0.5 ...")
    for i in range(num_samples):
        L = random.randint(min_len, max_len)
        msg = generate_random_string(L, char_set=alphabet)
        try:
            states_norm, _ = message_to_evolution_maps(msg)
            if states_norm is None:
                continue
            baseline_states = np.full_like(states_norm, 0.5, dtype=np.float32)
            delta = states_norm - baseline_states
            grid = baseline_states[..., np.newaxis]  # (64, 8, 1)

            inputs.append(grid)
            residuals.append(delta)
            clean_states.append(states_norm)

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{num_samples} samples processed")
        except Exception as e:
            print(f"  Error on sample {i+1}: {e}")

    if not inputs:
        print("No evolution samples generated.")
        return None, None, None

    X = np.asarray(inputs, dtype=np.float32)
    Y = np.asarray(residuals, dtype=np.float32)
    T = np.asarray(clean_states, dtype=np.float32)

    if save:
        np.savez_compressed(out_path, inputs=X, residuals=Y, targets=T)
        print(f"Saved evolution dataset to {out_path} (inputs: {X.shape}, residuals: {Y.shape})")

    return X, Y, T
