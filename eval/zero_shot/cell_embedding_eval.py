## This file is adapted from https://github.com/microsoft/zero-shot-scfoundation/blob/main/sc_foundation_evals/cell_embeddings.py

## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

"""
Generate benchmark metrics and visualize UMAP figures for given cell embeddings.

Example usage:
```
    eval = CellEmbeddingsEval(model_name='Geneformer',
                              embedding_file_path='example/embedding/Geneformer/embedding.h5ad',
                              output_dir='example/eval/Geneformer/',
                              label_key=['cell_type'],
                              batch_key=None)
    eval.evaluate()
    eval.visualize()
```
"""

import os
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('fivethirtyeight')

import seaborn as sns
import scanpy as sc

import utils
import umap_util as umap


class CellEmbeddingsEval():
    def __init__(self,
                 model_name: str,
                 embedding_file_path: str,
                 output_dir: str,
                 label_key: List[str] = ["cell_type"],
                 batch_key: Optional[str] = None) -> None:
        assert model_name in {"Geneformer", "scGPT", "genePT-w", "genePT-s"}, "Invalid model name"
        self.model_name = model_name
        self.adata = sc.read(embedding_file_path)

        self.batch_key = batch_key
        if batch_key is not None and batch_key not in self.adata.obs.columns:
            print(f"batch_key {batch_key} not found in adata.obs")
            return

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.label_key = label_key

        # make sure that each label exists and is categorical in adata.obs
        for label in self.label_key:
            if label not in self.adata.obs.columns:
                print(f"Label {label} not found in adata.obs")
                return
            self.adata.obs[label] = self.adata.obs[label].astype("category")

    def evaluate(self,
                 n_cells: int = 7500,
                 embedding_key: Optional[str] = None) -> pd.DataFrame:
        if embedding_key is None:
            embedding_key = f'X_{self.model_name}'

        adata_ = self.adata.copy()

        # if adata_ too big, take a subset
        if adata_.n_obs > n_cells:
            print(f"adata_ has {adata_.n_obs} cells. "
                  f"Taking a subset of {n_cells} cells.")
            sc.pp.subsample(adata_, n_obs=n_cells, copy=False)

        met_df = pd.DataFrame(columns=["metric", "label", "value"])

        # get unique values in self.label_key preserving the order
        label_cols = [x for i, x in enumerate(self.label_key)
                      if x not in self.label_key[:i]]
        # remove label columns that are not in adata_.obs
        label_cols = [x for x in label_cols if x in adata_.obs.columns]

        if len(label_cols) == 0:
            msg = f"No label columns {self.label_key} found in adata.obs"
            print(msg)
            raise ValueError(msg)

        # check if the embeddings are in adata
        if embedding_key not in adata_.obsm.keys():
            msg = f"Embeddings {embedding_key} not found in adata.obsm"
            print(msg)
            raise ValueError(msg)

        for label in label_cols:
            metrics = utils.eval_scib_metrics(adata_,
                                              batch_key=self.batch_key,
                                              label_key=label,
                                              embedding_key=embedding_key)
            for metric in metrics.keys():
                # add row to the dataframe
                met_df.loc[len(met_df)] = [metric, label, metrics[metric]]

        met_df.to_csv(os.path.join(self.output_dir,
                                   f"{embedding_key}__metrics.csv"),
                      index=False)

        return met_df

    def create_original_umap(self,
                             out_emb: str = "X_umap_input") -> None:

        sc.pp.neighbors(self.adata)
        temp = sc.tl.umap(self.adata, min_dist=0.3, copy=True)
        self.adata.obsm[out_emb] = temp.obsm["X_umap"].copy()

    def visualize(self,
                  return_fig: bool = False,
                  plot_size: Tuple[float, float] = (9, 7),
                  plot_title: Optional[str] = None,
                  plot_type: [List, str] = "simple",
                  n_cells: int = 7500,
                  embedding_key: Optional[str] = None
                  ) -> Optional[Dict[str, plt.figure]]:

        raw_emb = "X_umap_input"
        if embedding_key is None:
            embedding_key = f'X_{self.model_name}'

        if embedding_key == raw_emb:
            # if the umap_raw embedding is used, create it first
            self.create_original_umap(out_emb=embedding_key)

        # if adata already has a umap embedding warn that it will be overwritten
        if "X_umap" in self.adata.obsm.keys():
            old_umap_name = "X_umap_old"
            self.adata.obsm[old_umap_name] = self.adata.obsm["X_umap"].copy()

        # check if the embeddings are in adata
        if embedding_key not in self.adata.obsm.keys():
            msg = f"Embeddings {embedding_key} not found in adata."
            print(msg)
            raise ValueError(msg)

        # if embedding_key contains the string umap, do not compute umap again
        if embedding_key != raw_emb:
            # compute umap embeddings
            sc.pp.neighbors(self.adata, use_rep=embedding_key)
            sc.tl.umap(self.adata, min_dist=0.3)

        adata_ = self.adata.copy()
        # if adata_ too big, take a subset
        if adata_.n_obs > n_cells:
            print(f"adata_ has {adata_.n_obs} cells. "
                  f"Taking a subset of {n_cells} cells.")
            sc.pp.subsample(adata_, n_obs=n_cells, copy=False)
            # save the subsetted adata.obs
            adata_.obs.to_csv(os.path.join(self.output_dir,
                                           "adata_obs_subset.csv"))

        # make sure plot size is a tuple of numbers
        try:
            w, h = plot_size
            if not isinstance(h, (int, float)) or not isinstance(w, (int, float)):
                msg = f"Height (h = {h}) or width (w = {w}) not valid."
                print(msg)
                raise TypeError(msg)
        except TypeError:
            msg = f"Plot size {plot_size} is not a tuple of numbers."
            print(msg)
            raise TypeError(msg)

        # get unique values in self.label_key preserving the order
        label_cols = self.label_key + [self.batch_key]
        label_cols = [x for i, x in enumerate(label_cols)
                      if x not in label_cols[:i]]
        # remove label columns that are not in adata_.obs
        label_cols = [x for x in label_cols
                      if x in self.adata.obs.columns]

        if len(label_cols) == 0:
            msg = f"No label columns {self.label_key} found in adata.obs"
            print(msg)
            raise ValueError(msg)

        # set the colors for the labels
        labels = dict()
        labels_colors = dict()
        palettes = ['viridis', 'inferno',
                    'mako', 'rocket',
                    'tab20', 'colorblind',
                    'tab20b', 'tab20c']

        if len(label_cols) > len(palettes):
            print("More labels than palettes. Adding random colors.")
            palettes = palettes + ["random"] * (len(label_cols) - len(palettes))

        # creating palettes for the labels
        for i, label in enumerate(label_cols):
            labels[label] = self.adata.obs[label].unique()
            if len(labels[label]) > 10:
                print(f"More than 10 labels for {label}."
                      f"The plots might be hard to read.")
            labels_colors[label] = dict(zip(labels[label],
                                            umap.generate_pallette(n=len(labels[label]),
                                                                   cmap=palettes[i])))

        figs = {}

        # if plot_type a string, convert to list
        if isinstance(plot_type, str):
            plot_type = [plot_type]

        plot_type = [x.lower() for x in plot_type]
        # get unique values in plot_type
        plot_type = [x for i, x in enumerate(plot_type)
                     if x not in plot_type[:i]]
        old_plot_type = plot_type
        # check if plot_type is valid
        valid_plot_types = ["simple", "wide", "scanpy"]

        # create a subset of plot_type that is valid
        plot_type = [x for x in plot_type if x in valid_plot_types]
        if len(plot_type) == 0:
            msg = f"Plot type {plot_type} is not valid. Valid plot types are {valid_plot_types}"
            print(msg)
            raise ValueError(msg)

        # print a warning if plot_type is not valid
        if len(plot_type) < len(old_plot_type):
            print(f"Some plot type(s) {old_plot_type} is not valid. "
                  f"Valid plot types are {valid_plot_types}. "
                  f"Plotting only {plot_type}")

        plt_emb = "X_umap" if embedding_key != raw_emb else embedding_key

        plot_title = (plot_title
                      if plot_title is not None
                      else "UMAP of the cell embeddings")

        if "simple" in plot_type:
            fig, axs = plt.subplots(ncols=len(label_cols),
                                    figsize=(len(label_cols) * w, h),
                                    squeeze=False)

            axs = axs.flatten()

            # basic plotting, problematic: size of the points
            embedding = self.adata.obsm[plt_emb]
            for i, label in enumerate(label_cols):
                # remove axis and grid from the plot
                axs[i].axis('off')
                # plot umap embeddings, add color by cell type
                axs[i].scatter(embedding[:, 0], embedding[:, 1],
                               # make points smaller
                               s=0.5,
                               c=[labels_colors[label][x] for x
                                  in self.adata.obs[label]])
                legend_handles = [axs[i].plot([], [],
                                              marker="o", ls="",
                                              color=c, label=l)[0]
                                  for l, c in labels_colors[label].items()]
                axs[i].legend(handles=legend_handles,
                              bbox_to_anchor=(1.05, 1),
                              loc='upper left')

                # Add a title to the plot
                axs[i].title.set_text(f"{label}")

            fig.suptitle(plot_title, fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.85)

            fig_savefig = os.path.join(self.output_dir,
                                       f"umap__{embedding_key}.png")
            fig.savefig(fig_savefig)

            if return_fig:
                figs["umap"] = fig

        # wide plotting
        if "wide" in plot_type:
            df = pd.DataFrame(self.adata.obsm[plt_emb],
                              columns=["umap_1", "umap_2"])
            for i, label in enumerate(label_cols):
                if self.adata.obs[label].unique().shape[0] <= 10:
                    df[label] = self.adata.obs[label].tolist()
                    wide_plot = sns.relplot(data=df,
                                            col=label,
                                            x="umap_1",
                                            y="umap_2",
                                            hue=label,
                                            style=label,
                                            legend="full",
                                            palette=palettes[i])
                    # switch off axes
                    for axes in wide_plot.axes.flat:
                        axes.set_axis_off()
                    sns.move_legend(wide_plot, "upper left", bbox_to_anchor=(1, 1))
                    wide_plot.fig.suptitle(plot_title, fontsize=16)
                    wide_plot.fig.tight_layout()
                    wide_plot.fig.subplots_adjust(top=0.85)

                    wide_plot_savefig = os.path.join(self.output_dir,
                                                     f"umap_wide__{embedding_key}_{label}.png")
                    wide_plot.savefig(wide_plot_savefig)

                    if return_fig:
                        figs[label] = wide_plot
                else:
                    print(f"More than 10 labels for {label}. Skipping wide plot.")

        if "scanpy" in plot_type:
            # scanpy plotting
            labels_colors_flat = {k: v for d in labels_colors
                                  for k, v in labels_colors[d].items()}
            if embedding_key == raw_emb:
                # TODO: this needs rewriting
                adata_temp__ = self.adata.copy()
                adata_temp__.obsm["X_umap"] = self.adata.obsm[raw_emb].copy()
                fig2 = sc.pl.umap(adata_temp__,
                                  color=label_cols,
                                  add_outline=True,
                                  layer=plt_emb,
                                  legend_loc='on data',
                                  palette=labels_colors_flat,
                                  return_fig=True)
                # remove the temporary adata
                del adata_temp__
            else:
                fig2 = sc.pl.umap(self.adata,
                                  color=label_cols,
                                  add_outline=True,
                                  layer=plt_emb,
                                  legend_loc='on data',
                                  palette=labels_colors_flat,
                                  return_fig=True)
            fig2.suptitle(plot_title, fontsize=16)
            fig2.tight_layout()
            fig2.subplots_adjust(top=0.85)

            fig2_savefig = os.path.join(self.output_dir,
                                        f"umap_scanpy__{embedding_key}.png")
            fig2.savefig(fig2_savefig)

            if return_fig:
                figs["umap_scanpy"] = fig2

        return figs if return_fig else None

