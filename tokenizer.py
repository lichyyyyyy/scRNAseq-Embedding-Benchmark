import glob
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from model_source_code.scGPT.scgpt.tokenizer import tokenize_and_pad_batch, GeneVocab
from scipy.sparse import issparse
from transformers import AutoTokenizer

import config

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
**Input data:**
| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required row (gene) attribute:* "gene_id"; Gene symbol for each gene.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

**Output data:**
List of tuple (genes, tokenized values) of non-zero gene expressions in .pt format.


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
            for file_path in glob.glob(config.scgpt_configs['raw_data_directory'] + f"/*.h5ad"):
                print(f"Tokenizing {file_path}")
                adata = sc.read_h5ad(file_path)
                input_data = (adata.X.toarray() if issparse(adata.X) else adata.X)
                vocab = GeneVocab.from_file(
                    config.scgpt_configs['load_model_dir'] + "/vocab.json")
                vocab.set_default_index(vocab[config.scgpt_configs['pad_token']])
                gene_ids = np.array(vocab(adata.var.index.tolist()), dtype=int)
                # Return a list of tuple (genes, values) of non-zero gene expressions.
                # Return type: List[Tuple[torch.Tensor, torch.Tensor]].
                tokenized_data = tokenize_and_pad_batch(data=input_data,
                                                        gene_ids=gene_ids,
                                                        max_len=config.scgpt_configs['max_seq_len'],
                                                        vocab=vocab,
                                                        pad_token=config.scgpt_configs['pad_token'],
                                                        pad_value=config.scgpt_configs['pad_value'],
                                                        append_cls=True,
                                                        include_zero_gene=config.scgpt_configs['include_zero_gene'],
                                                        cls_token=config.scgpt_configs['cls_token'])
                os.makedirs(config.scgpt_configs['tokenized_file_dir'], exist_ok=True)
                output_file_path = config.scgpt_configs['tokenized_file_dir'] + '/' + Path(file_path).stem + '.pt'
                torch.save(tokenized_data, output_file_path)
                print("Tokenized data saved in " + output_file_path)
            return None

        return print("Invalid model name")


t = Tokenizer("Geneformer")
t.tokenize()
