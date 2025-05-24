import glob
import os
from pathlib import Path

from config import scgpt_configs, geneformer_configs
from models.geneformer import EmbExtractor
from models.scGPT import embed_data

"""
Generate embeddings for given scRNAseq data.

**Input data:**
- Geneformer: Tokenized scRNAseq data in Anndata format.
- scGPT: Raw Anndata file.

**Output data:**
Single cell transcriptomics embeddings in cvs.



**Usage:**
```
    emb_extractor = EmbeddingExtractor("Geneformer")
    emb_extractor.extract_embeddings()
```

"""


class EmbeddingExtractor:
    def __init__(self, model_name):
        """
        Initialize embedding extractor.

        :param model_name: {"Geneformer", "scGPT"}
            Name of the pretrained model.
        """
        assert model_name in {"Geneformer", "scGPT"}, "Invalid model name"
        self.model_name = model_name

    def extract_embeddings(self):
        """
        Extract transcriptomics embeddings for input scRNAseq data.
        """
        if self.model_name == "Geneformer":
            extractor = EmbExtractor(model_type="Pretrained",
                                     num_classes=0,  # 0 for the pre-trained model
                                     emb_mode=geneformer_configs['embedding_mode'],  # {"cls", "cell", "gene"}
                                     max_ncells=None,  # If None, will extract embeddings from all cells.
                                     emb_layer=-1,
                                     forward_batch_size=10,
                                     nproc=4)
            return extractor.extract_embs(
                model_directory=os.path.join(geneformer_configs['load_model_dir'],
                                             geneformer_configs['model_file_name']),
                input_data_file=os.path.join(geneformer_configs['tokenized_file_directory'],
                                             geneformer_configs['tokenized_file_prefix'] + '.dataset'),
                output_directory=geneformer_configs['embedding_output_directory'],
                output_prefix=geneformer_configs['embedding_output_prefix'],
                output_torch_embs=False)

        elif self.model_name == "scGPT":
            for file_path in glob.glob(scgpt_configs['raw_data_directory'] + f"/*.h5ad"):
                print(f"Embedding {file_path}")
                embed_adata = embed_data(
                    adata_or_file=file_path,
                    model_dir=scgpt_configs['load_model_dir'],
                    gene_col="gene_name",
                    max_length=1200,
                    batch_size=64,
                    obs_to_save=None,
                    use_fast_transformer=True,
                    return_new_adata=False)
                embed_adata.obsm['X_scGPT'].to_csv(
                    scgpt_configs['embedding_output_directory'] + scgpt_configs['embedding_output_prefix'] + Path(
                        file_path).stem, index=False, header=False)

        return print("Invalid model name")


emb_extractor = EmbeddingExtractor("Geneformer")
emb_extractor.extract_embeddings()
