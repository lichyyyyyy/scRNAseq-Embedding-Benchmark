import scanpy as sc
import torch
from scipy.sparse import issparse

import config
from models.Geneformer.geneformer import TranscriptomeTokenizer
from models.scGPT.scgpt.tokenizer import tokenize_and_pad_batch, GeneVocab

"""
Tokenize pre-processed scRNAseq data.


** --------- Geneformer ------------ **
**Input data:**
| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "n_counts"; total read counts in that cell.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

**Output data:**
Tokenized scRNAseq data in Anndata format.


** --------- scGPT ------------ **
**Input data:**
**Input data:**
| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required row (gene) attribute:* "gene_id"; Gene symbol for each gene.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

**Output data:**
List of tuple (genes, tokenzied values) of non-zero gene expressions in .pt format.


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
        Tokenize the pre-processed scRNAseq data.
        """
        if self.model_name == "Geneformer":
            tk = TranscriptomeTokenizer(nproc=4)
            tk.tokenize_data(
                config.geneformer_configs.preprocess_data_directory,
                config.geneformer_configs.tokenized_file_directory,
                config.geneformer_configs.tokenized_file_prefix, "h5ad")

        elif self.model_name == "scGPT":
            for file_path in config.scgpt_configs.raw_data_directory.glob(
                    f"*.h5ad"):
                print(f"Tokenizing {file_path}")
                adata = sc.read_h5ad(file_path)
                input_data = (
                    adata.layers[config.scgpt_configs.input_layer_key].toarray()
                    if issparse(
                        adata.layers[config.scgpt_configs.input_layer_key])
                    else adata.layers[config.scgpt_configs.input_layer_key])
                vocab = GeneVocab.from_file(
                    config.scgpt_configs.load_model_dir + "/vocab.json")
                # Return a list of tuple (genes, values) of non-zero gene expressions.
                # Return type: List[Tuple[torch.Tensor, torch.Tensor]].
                tokenized_data = tokenize_and_pad_batch(data=input_data,
                                                        gene_ids=adata.var[
                                                            config.scgpt_configs.gene_id_key].toarray(),
                                                        max_len=config.scgpt_configs.max_seq_len,
                                                        vocab=vocab,
                                                        pad_token=config.scgpt_configs.pad_token,
                                                        pad_value=config.scgpt_configs.pad_value,
                                                        append_cls=True,
                                                        include_zero_gene=config.scgpt_configs.include_zero_gene,
                                                        cls_token=config.scgpt_configs.cls_token)
                torch.save(tokenized_data,
                           config.scgpt_configs.configs.tokenized_file_path)
                print(
                    f"train set number of samples: {tokenized_data['genes'].shape[0]}, "
                    f"\n\t feature length: {tokenized_data['genes'].shape[1]}")

        return print("Invalid model name")


t = Tokenizer("Geneformer")
t.tokenize()
