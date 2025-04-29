"""
Data pre-processor.

**Input data:**
Sc RNA seq in Anndata format (.h5ad).
| *Required row (gene) attribute:* "gene_ids"; ID (name) for each gene.

**Output data:**
Sc RNA seq in Anndata format (.h5ad).
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "n_counts"; total read counts in that cell.


**Usage:**
```
    p = PreProcessor()
    p.pre_process()
```


"""

import pandas as pd
import scanpy as sc

# Example scRNAseq data sample from https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html#scanpy.datasets.pbmc3k
EXAMPLE_INPUT_FILE_PATH = "../example/data/pbmc3k_raw.h5ad"
EXAMPLE_OUTPUT_FILE_PATH = "../example/data/processed/pbmc3k_processed.h5ad"
DEFAULT_GENE_INFO_MAPPING_FILE = "../data/gene_info_table.csv"


class PreProcessor:
    def __init__(self, input_filepath=EXAMPLE_INPUT_FILE_PATH,
                 output_filepath=EXAMPLE_OUTPUT_FILE_PATH,
                 gene_info_mapping=DEFAULT_GENE_INFO_MAPPING_FILE):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.gene_info_mapping = gene_info_mapping

    '''
    Add required attributions and generate the output file.
    '''

    def pre_process(self):
        adata = sc.read_h5ad(self.input_filepath)

        # Add `ensembl_id` for each gene.
        gene_info_table = pd.read_csv(self.gene_info_mapping)
        gene_name_to_ensembl = dict(
            zip(gene_info_table['gene_name'], gene_info_table['ensembl_id']))
        adata.var['ensembl_id'] = adata.var.index.map(gene_name_to_ensembl)

        # Add `n_counts` for each cell.
        adata.obs['n_counts'] = adata.X.sum(axis=1)

        adata.write_h5ad(self.output_filepath, compression="gzip")


p = PreProcessor()
p.pre_process()
