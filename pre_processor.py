"""
Data pre-processor.

**Input data:**
Sc RNA seq in Anndata format (.h5ad).
| *Required row (gene) attribute:* "gene_ids", type: {"ensemble_id", "gene_symbol"}

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

import pandas as pd
import scanpy as sc

import config


# Example scRNAseq data sample from https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html#scanpy.datasets.pbmc3k

class PreProcessor:
    @staticmethod
    def pre_process(gene_key_col=config.preprocessor_configs['gene_key_col'],
                    gene_key_type=config.preprocessor_configs['gene_key_type'],
                    file_type=config.preprocessor_configs['data_file_type'],
                    raw_data_directory: str = config.raw_data_directory,
                    raw_data_filename: str = config.raw_data_filename,
                    preprocessed_data_directory: str = config.preprocessed_data_directory,
                    preprocessed_data_filename: str = config.preprocessed_data_filename):
        """
        Pre-process the Anndata file.
        :param gene_key_col: str
            The column name of the gene key/ ID column.
        :param gene_key_type: {"gene_symbol", "ensembl_id"}
            The type of the gene key / ID.
        :param file_type: {"Anndata"}
        """
        assert file_type in {"Anndata"}, "Unsupported file type"
        assert gene_key_type in {"gene_symbol", "ensembl_id"}, "Unsupported gene key system"

        input_filepath = os.path.join(raw_data_directory, raw_data_filename)
        output_filepath = os.path.join(preprocessed_data_directory, preprocessed_data_filename)
        os.makedirs(preprocessed_data_directory, exist_ok=True)

        if file_type == "Anndata":
            adata = sc.read_h5ad(input_filepath)
            assert gene_key_col in adata.var, "Gene key column not found in Anndata file"

            gene_info_table = pd.read_csv(config.gene_info_table)

            # Add `ensembl_id`, `gene_symbol` for each gene.
            adata.var = pd.merge(adata.var, gene_info_table, how='left',
                                 left_on=config.preprocessor_configs['gene_key_col'], right_on=gene_key_type)
            adata.var = adata.var[['gene_symbol', 'ensembl_id', 'gene_type']]

            # Add `n_counts` for each cell.
            adata.obs['n_counts'] = adata.X.sum(axis=1)
            return adata.write_h5ad(output_filepath, compression="gzip")

        return print("Invalid model name")


PreProcessor.pre_process()
