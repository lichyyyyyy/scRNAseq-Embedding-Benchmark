#!/usr/bin/env python
"""
Few-shot cell-type annotation evaluation on pre-computed embeddings.

Inputs
------
embeddings.npy : (n_cells, d) array-like
meta.csv       : CSV with a 'cell_type' column (n_cells rows)

Usage
-----
python fewshot_eval.py --shots 5 --repeats 10 --seed 0


参数说明
参数	默认	作用
--shots	5	每个 cell type 取多少个样本做「支撑集」
--repeats	10	随机抽样重复次数（不同 seed）
--seed	0	抽样基准随机种子

注意：脚本只计算有 ≥ shots+1 个细胞的类别，少于该数量的 cell type 会被自动跳过。
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
rng = np.random.default_rng()

warnings.simplefilter("ignore", category=FutureWarning)


def load():
    X = np.load("embeddings.npy").astype(np.float32)
    meta = pd.read_csv("meta.csv")
    if len(meta) != len(X):
        sys.exit(f"[ERR] rows mismatch: {len(meta)} vs {len(X)}")
    y = meta["cell_type"].values
    return X, y


def sample_few_shot_indices(labels, shots, random_state):
    """Return train_idx, test_idx given desired shots per class."""
    rs = np.random.RandomState(random_state)
    train, test = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        if len(idx) < shots + 1:  # 至少留 1 个测试
            continue
        rs.shuffle(idx)
        train.extend(idx[:shots])
        test.extend(idx[shots:])
    return np.array(train), np.array(test)


def proto_predict(X_train, y_train, X_test):
    """Prototype-nearest-neighbor (cosine)"""
    prototypes = {}
    for cls in np.unique(y_train):
        prototypes[cls] = X_train[y_train == cls].mean(0, keepdims=True)
    proto_mat = np.vstack(list(prototypes.values()))
    cls_order = list(prototypes.keys())
    dist = cdist(X_test, proto_mat, metric="cosine")
    return np.array(cls_order)[dist.argmin(1)]


def evaluate(X, y, shots=5, repeats=10, seed=0):
    le = LabelEncoder().fit(y)
    y_num = le.transform(y)

    accs, f1s, recalls = [], [], []
    accs_knn, f1s_knn = [], []

    for r in range(repeats):
        train_idx, test_idx = sample_few_shot_indices(
            y_num, shots, random_state=seed + r)

        if len(test_idx) == 0:
            continue  # skip if no test

        X_tr, y_tr = X[train_idx], y_num[train_idx]
        X_te, y_te = X[test_idx], y_num[test_idx]

        # --- Prototype NN ---
        y_pred = proto_predict(X_tr, y_tr, X_te)
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, average="macro"))
        recalls.append(recall_score(y_te, y_pred, average=None).mean())

        # --- k-NN baseline ---
        clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
        clf.fit(X_tr, y_tr)
        y_pred_knn = clf.predict(X_te)
        accs_knn.append(accuracy_score(y_te, y_pred_knn))
        f1s_knn.append(f1_score(y_te, y_pred_knn, average="macro"))

    def mean_std(arr):
        return np.mean(arr), np.std(arr)

    metrics = {
        "proto_acc": mean_std(accs),
        "proto_macroF1": mean_std(f1s),
        "proto_avgRecall": mean_std(recalls),
        "knn_acc": mean_std(accs_knn),
        "knn_macroF1": mean_std(f1s_knn),
        "shots": shots,
        "repeats": repeats,
    }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    X, y = load()
    metrics = evaluate(X, y, shots=args.shots, repeats=args.repeats, seed=args.seed)

    print("\n===== Few-Shot Annotation Evaluation =====")
    for k, (m, s) in metrics.items():
        if isinstance(m, tuple):  # metric with mean/std
            print(f"{k:>18}: {m:.4f} ± {s:.4f}")
    # save
    Path("fewshot_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved → fewshot_metrics.json")


if __name__ == "__main__":
    main()
