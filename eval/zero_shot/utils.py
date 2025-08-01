## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

from typing import Dict, Optional

import numpy as np
import scanpy as sc
import scib
from anndata import AnnData

# MODIFIED wrapper for all scib metrics from
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
        adata: AnnData,
        batch_key: Optional[str] = "str_batch",
        label_key: str = "cell_type",
        embedding_key: str = "X_scGPT"
) -> Dict:
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:
        print(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
              "\nto make sure the optimal clustering is calculated for the "
              "correct embedding, removing neighbors from adata.uns."
              "\nOverwriting calculation of neighbors with "
              f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        print("neighbors in adata.uns removed, new neighbors calculated: "
              f"{adata.uns['neighbors']}")

    # in case just one batch scib.metrics.metrics doesn't work
    # call them separately
    results_dict = dict()

    res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
        adata,
        label_key=label_key,
        cluster_key="cluster",
        use_rep=embedding_key,
        function=scib.metrics.nmi,
        plot=False,
        verbose=False,
        inplace=True,
        force=True,
    )

    results_dict["NMI_cluster/label"] = scib.metrics.nmi(
        adata,
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )

    results_dict["ARI_cluster/label"] = scib.metrics.ari(
        adata,
        "cluster",
        label_key
    )

    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata,
        label_key,
        embedding_key,
        "euclidean"
    )

    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )

    # Calculate this only if there are multiple batches
    if batch_key != None and len(adata.obs[batch_key].unique()) > 1:
        results_dict["ASW_batch"] = scib.metrics.silhouette(
            adata,
            batch_key,
            embedding_key,
            "euclidean"
        )

        results_dict["ASW_label/batch"] = scib.metrics.silhouette_batch(
            adata,
            batch_key,
            label_key,
            embed=embedding_key,
            metric="euclidean",
            return_all=False,
            verbose=False
        )

        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )

    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster/label"],
            results_dict["ARI_cluster/label"],
            results_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    return results_dict
