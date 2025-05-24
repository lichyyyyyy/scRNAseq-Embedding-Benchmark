import glob
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse
from transformers import AutoTokenizer

import config
from models.scGPT import tokenize_and_pad_batch
from models.scGPT.gene_tokenizer import GeneVocab

"""
Tokenize pre-processed scRNAseq data.


********************** Geneformer **********************
**Input data:**
| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "n_counts"; total read counts in that cell.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

**Output data:**
Tokenized scRNAseq data in Anndata format.


************************ scGPT *************************
Skipped. Tokenized scRNAseq data in the embedding extractor.


**Usage:**
```
    t = Tokenizer("Geneformer")
    t.tokenize()
```

"""


class Tokenizer:
    def __init__(self, model_name):
        """
        Initialize embedding extractor.

        :param model_name: {"Geneformer", "scGPT"}
            Name of the pretrained model.
        """
        assert model_name in {"Geneformer", "scGPT"}, "Invalid model name"
        self.model_name = model_name

    def tokenize(self):
        """
        Tokenize the pre-processed scRNAseq data for given model.
        - Input: a directory of pre-processed scRNAseq data in Anndata format.
        - Output: a directory of tokenized scRNAseq data in Anndata format.
        """
        if self.model_name == "Geneformer":
            # tk = TranscriptomeTokenizer(nproc=1,
            #                             # For the 95M model series, model_input_size should be 4096.
            #                             model_input_size=4096)
            tk = AutoTokenizer.from_pretrained("ctheodoris/Geneformer", force_download=True)
            return tk.tokenize_data(
                data_directory=config.geneformer_configs['preprocess_data_directory'],
                output_directory=config.geneformer_configs['tokenized_file_directory'],
                output_prefix=config.geneformer_configs['tokenized_file_prefix'], file_format="h5ad")

        elif self.model_name == "scGPT":
            return None

        return print("Invalid model name")


t = Tokenizer("Geneformer")
t.tokenize()
