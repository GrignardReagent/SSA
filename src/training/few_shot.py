"""Few-shot evaluation helpers for generated unseen classes."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.processing.normalisation import instance_norm_np


def prepare_group_input(pool, num_traj, crop_len, device):
    """Sample, crop, normalise, and concatenate trajectories for one model input."""
    indices = np.random.choice(len(pool), num_traj, replace=True)
    selected = [pool[i] for i in indices]

    processed_parts = []
    for traj in selected:
        if traj.shape[0] > crop_len:
            start = np.random.randint(0, traj.shape[0] - crop_len)
            crop = traj[start : start + crop_len]
        else:
            crop = np.pad(traj, (0, crop_len - len(traj)), mode="edge")

        norm = instance_norm_np(crop)
        if norm.ndim == 1:
            norm = norm.reshape(-1, 1)

        processed_parts.append(norm)

    concat_group = np.concatenate(processed_parts, axis=0)
    return torch.tensor(concat_group, dtype=torch.float32).unsqueeze(0).to(device)


def test_baseline_performance(
    model,
    unseen_data,
    scaler=None,
    num_traj: int = 10,
    crop_len: int = 1811,
    samples_per_class: int = 50,
) -> float:
    """Evaluate few-shot nearest-prototype performance on generated classes."""
    model.eval()
    device = next(model.parameters()).device

    correct = 0
    total_queries = 0

    print("\n=== Testing Baseline Model (Feature Distance) ===")
    print(f"Model expects groups of {num_traj} trajectories (Concatenated Time).")

    class_prototypes = {}
    for data in unseen_data:
        pool = data["trajectories"]
        ref_embeddings = []

        for _ in range(5):
            tensor_in = prepare_group_input(pool, num_traj, crop_len, device)
            with torch.no_grad():
                ref_embeddings.append(model.encode(tensor_in).cpu().numpy())

        class_prototypes[data["class_id"]] = np.mean(ref_embeddings, axis=0).flatten()

    print(f"Built prototypes for {len(class_prototypes)} classes.")

    for data in unseen_data:
        true_class = data["class_id"]
        pool = data["trajectories"]

        for _ in range(samples_per_class):
            tensor_in = prepare_group_input(pool, num_traj, crop_len, device)
            with torch.no_grad():
                query_vec = model.encode(tensor_in).cpu().numpy().flatten()

            best_dist = float("inf")
            predicted = None
            for cls_id, proto in class_prototypes.items():
                dist = np.linalg.norm(query_vec - proto)
                if dist < best_dist:
                    best_dist = dist
                    predicted = cls_id

            if predicted == true_class:
                correct += 1
            total_queries += 1

    acc = correct / total_queries
    print(f"Baseline Few-Shot Accuracy: {acc:.2%} ({correct}/{total_queries})")
    return acc


def test_svm_few_shot(unseen_data, crop_len: int = 1811, n_support: int = 5) -> float:
    """Train a small SVM on generated classes and evaluate held-out trajectories."""
    print("\n=== Testing SVM Few-Shot Baseline (Raw Data) ===")

    X_support = []
    y_support = []

    for label_idx, data in enumerate(unseen_data):
        refs = data["trajectories"][:n_support]

        for traj in refs:
            if len(traj) > crop_len:
                crop = traj[:crop_len]
            else:
                crop = np.pad(traj, (0, crop_len - len(traj)), mode="edge")

            X_support.append(crop)
            y_support.append(label_idx)

    X_support = np.array(X_support)
    y_support = np.array(y_support)

    scaler = StandardScaler()
    X_support_scaled = scaler.fit_transform(X_support)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    clf.fit(X_support_scaled, y_support)
    print(f"Trained SVM on {len(X_support)} support samples.")

    correct = 0
    total = 0

    for label_idx, data in enumerate(unseen_data):
        queries = data["trajectories"][n_support:]

        for traj in queries:
            if len(traj) > crop_len:
                start = np.random.randint(0, len(traj) - crop_len)
                crop = traj[start : start + crop_len]
            else:
                crop = np.pad(traj, (0, crop_len - len(traj)), mode="edge")

            feat_vec = crop.reshape(1, -1)
            pred = clf.predict(scaler.transform(feat_vec))[0]

            if pred == label_idx:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"SVM Few-Shot Accuracy: {acc:.2%}")
    return acc
