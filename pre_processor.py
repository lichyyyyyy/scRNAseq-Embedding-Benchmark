"""
Data pre-processor.

**Input data:**
Sc RNA seq in Anndata format (.h5ad).
| *Required row (gene) attribute:* gene ID indexed in `vars`, supporting naming system: {"gene_symbol", "ensembl_id",
 "entrez_id", "refseq_id"}.
Note： there is no need to pre-process, filter, or log-transform the input in advance.

**Output data:**
Sc RNA seq in Anndata format (.h5ad).
| *"ensemble_id": Ensembl ID
| *"gene_name": gene symbol
| *"n_counts"; total read counts in that cell.

**Usage:**
```
PreProcessor.pre_process()
```


"""
import os
from pathlib import Path

import pandas as pd
import scanpy as sc

import config


# Example scRNAseq data sample from https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html#scanpy.datasets.pbmc3k

class PreProcessor:
    @staticmethod
    def pre_process(gene_key_type=config.preprocessor_configs['gene_key_type'],
                    file_format=config.preprocessor_configs['file_format'],
                    raw_data_directory: str = config.raw_data_directory,
                    preprocessed_data_directory: str = config.preprocessed_data_directory):
        """
        Pre-process the Anndata file.
        :param gene_key_type: {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}
            The type of the gene index.
        :param file_format: {"h5ad"}
        """
        assert file_format in {"h5ad"}, "Unsupported file type"
        assert gene_key_type in {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}, "Unsupported gene key system"

        os.makedirs(preprocessed_data_directory, exist_ok=True)

        for file_path in Path(raw_data_directory.glob(f"*.{file_format}")):
            print(f"Pre-processing {file_path}")
            adata = sc.read_h5ad(file_path)
            gene_info_table = pd.read_csv(config.gene_info_table)
            gene_info_table.drop_duplicates(subset=gene_key_type, inplace=True)

            adata.var = pd.merge(adata.var, gene_info_table, how='left', left_on=adata.var.index,
                                 right_on=gene_info_table[gene_key_type])
            adata.var = adata.var[['gene_symbol', 'ensembl_id', 'gene_type']]

            # Add `n_counts` for each cell.
            adata.obs['n_counts'] = adata.X.sum(axis=1)
            print(f"Pre-process completed: {file_path}")
            adata.write_h5ad(file_path, compression="gzip")

        return None


PreProcessor.pre_process()
