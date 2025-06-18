"""
Few-shot cell-type annotation evaluation on pre-computed embeddings.

**Usage:**
```
eval = FewShotCellEmbeddingsEval(model_name="Geneformer",
                                 embedding_file_path="example\embedding\Geneformer\cell_embeddings.h5ad",
                                 cell_type_key="CellType",
                                 output_path="example\eval/few_shot/geneformer.json")
print(eval.cell_type_annotation_eval(shots=5, repeats=5, seed=0))
```
"""
import warnings

warnings.filterwarnings("ignore")

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


class FewShotCellEmbeddingsEval:
    def __init__(self, model_name: str,
                 embedding_file_path: str,
                 cell_type_key: str = "cell_type",
                 cell_embedding_key: Optional[str] = None,
                 output_path: Optional[str] = None):
        assert model_name in {"Geneformer", "scGPT", "genePT-w", "genePT-s"}, "Invalid model name"
        self.model_name = model_name
        adata = sc.read(embedding_file_path)
        assert cell_type_key in adata.obs.columns, "Cell type key not in adata.obs.columns"
        if cell_embedding_key is None:
            cell_embedding_key = f"X_{model_name}"
        assert cell_embedding_key in adata.obsm.keys(), "Cell embedding key not found in adata.obsm"

        self.embeddings = adata.obsm[cell_embedding_key]
        self.cell_types = adata.obs[cell_type_key]
        self.output_path = None if output_path is None else Path(output_path)

    def cell_type_annotation_eval(self, shots: int = 5, repeats: int = 10, seed: int = 0):
        """
        Evaluate cell type annotation using pre-computed embeddings with few-shot configurations (knn & proto NN based).
        :param embeddings: a dataframe containing pre-computed cell embeddings.
        :param cell_type: a series containing cell types matches to the embeddings.
        :param shots: # of examples per cell type (supporting dataset, usually <= 5).
            Note:  this approach only works when # cell in the embeddings >= shots + 1.
        :param repeats: # of random resampling iterations.
        :param seed: random seed.
        :return: mean and standard deviation of the cell type annotation.
        """

        def get_sample_few_shot_indices(labels, shots, random_state):
            """Return train_idx, test_idx given desired shots per class."""
            rs = np.random.RandomState(random_state)
            train, test = [], []
            for cls in np.unique(labels):
                idx = np.where(labels == cls)[0]
                if len(idx) < shots + 1:
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

        def mean_std(arr):
            return np.mean(arr), np.std(arr)

        accs, f1s, recalls = [], [], []
        accs_knn, f1s_knn = [], []

        for r in range(repeats):
            train_idx, test_idx = get_sample_few_shot_indices(self.cell_types, shots, random_state=seed + r)

            if len(test_idx) == 0:
                continue  # skip if no test

            X_tr, y_tr = self.embeddings[train_idx], self.cell_types[train_idx]
            X_te, y_te = self.embeddings[test_idx], self.cell_types[test_idx]

            # Prototype NN
            y_pred = proto_predict(X_tr, y_tr, X_te)
            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="macro"))
            recalls.append(recall_score(y_te, y_pred, average=None).mean())

            # k-NN
            clf = KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")
            clf.fit(X_tr, y_tr)
            y_pred_knn = clf.predict(X_te)
            accs_knn.append(accuracy_score(y_te, y_pred_knn))
            f1s_knn.append(f1_score(y_te, y_pred_knn, average="macro"))

        metrics = {
            "proto_acc": mean_std(accs),
            "proto_macroF1": mean_std(f1s),
            "proto_avgRecall": mean_std(recalls),
            "knn_acc": mean_std(accs_knn),
            "knn_macroF1": mean_std(f1s_knn),
            "shots": shots,
            "repeats": repeats,
        }
        print("\n===== Few-Shot Annotation Evaluation =====")
        for k, m in metrics.items():
            print(f"{k:>18}: {m[0]:.4f}" + f" ± {m[1]:.4f}" if isinstance(m, tuple) else "")

        if self.output_path is not None:
            os.makedirs(self.output_path.parent, exist_ok=True)
            self.output_path.write_text(json.dumps(metrics, indent=2))
            print(f"Metrics saved to {self.output_path}")
        return metrics

