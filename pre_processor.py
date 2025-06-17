"""
Data pre-processor.

**Input data:**
Sc RNA seq in Anndata format (.h5ad).
| *Required row (gene) attribute:* gene ID in `vars`, supporting naming system: {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}.
| Note： there is no need to pre-process, filter, or log-transform the input in advance.

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


class PreProcessor:
    @staticmethod
    def pre_process(gene_id_col_name=config.preprocessor_configs['gene_id_col_name'],
                    gene_id_type=config.preprocessor_configs['gene_id_type'],
                    file_format=config.preprocessor_configs['file_format'],
                    raw_data_directory: str = config.raw_data_directory,
                    preprocessed_data_directory: str = config.preprocessed_data_directory,
                    keep_batch_key: bool = config.preprocessor_configs['keep_batch_key']):
        """
        Pre-process the Anndata file.
        :param gene_id_col_name: The column name of gene ID in `adata.var`. If the gene ID is the index, input `index`.
        :param gene_id_type: {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}
            The type of gene naming system in the gene ID.
        :param file_format: {"h5ad"}
        :param keep_batch_key: Whether to keep the batch key or not.
        """
        assert file_format in {"h5ad"}, "Unsupported file type"
        assert gene_id_type in {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}, "Unsupported gene key system"

        os.makedirs(preprocessed_data_directory, exist_ok=True)

        count, successful_cnt = 0, 0
        gene_info_table = pd.read_csv(config.gene_info_table)
        for file_path in Path(raw_data_directory).glob(f"*.{file_format}"):
            count += 1
            print(f"Pre-processing {file_path}")
            adata = sc.read_h5ad(file_path)
            if gene_id_col_name != 'index' and gene_id_col_name not in adata.var.columns:
                print(f"❗Failed to preprocess. The column name {gene_id_col_name} does not exist in {file_path}.")
                continue
            gene_info_table.drop_duplicates(subset=gene_id_type, inplace=True)
            for var_to_be_added in {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id", 'batch_key'}:
                if var_to_be_added in adata.var.columns:
                    adata.var[f'original_{var_to_be_added}'] = adata.var[var_to_be_added]
                    adata.var.drop(columns=[var_to_be_added], axis=1, inplace=True)
                    if gene_id_col_name == var_to_be_added:
                        gene_id_col_name = f'original_{var_to_be_added}'
            for col in adata.obs.columns:
                if col in config.preprocessor_configs['custom_cell_attr_names'].keys():
                    adata.obs[config.preprocessor_configs['custom_cell_attr_names'][col]] = adata.obs[col].copy()
                    adata.obs.drop(columns=[col], axis=1, inplace=True)
            left_key_column = adata.var.index if gene_id_col_name == 'index' else adata.var[gene_id_col_name]
            adata.var = pd.merge(adata.var, gene_info_table, how='left', left_on=left_key_column,
                                 right_on=gene_info_table[gene_id_type])
            if keep_batch_key:
                adata.obs['batch_key'] = os.path.basename(os.path.dirname(file_path))

            # Add `n_counts` for each cell.
            if 'n_counts' in adata.var.columns:
                adata.obs['original_n_counts'] = adata.obs['n_counts']
            adata.obs['n_counts'] = adata.X.sum(axis=1)
            output_filepath = os.path.join(preprocessed_data_directory, file_path.name)
            adata.write_h5ad(output_filepath, compression="gzip")
            successful_cnt += 1
            print(f"Pre-process completed: {output_filepath}. Shape:")
            print(adata)

        return print(f"Successfully pre-processed {successful_cnt} out of {count} files.")


PreProcessor.pre_process()