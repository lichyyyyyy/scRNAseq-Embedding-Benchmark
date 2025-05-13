import os

import config
from models.Geneformer.geneformer import EmbExtractor
from models.scGPT.scgpt.tasks import GeneEmbedding

"""
Generate embeddings for given scRNAseq data.

********************** Geneformer **********************
**Input data:**
- Geneformer: Tokenized scRNAseq data in Anndata format.
- scGPT: Tokenized scRNAseq data in .pt format.

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
                                 emb_mode=config.geneformer_configs['embedding_mode'],  # {"cls", "cell", "gene"}
                                 max_ncells=None,  # If None, will extract embeddings from all cells.
                                 emb_layer=-1,
                                 forward_batch_size=10,
                                 nproc=4)
            return extractor.extract_embs(
                model_directory=config.geneformer_configs['pre_trained_model_path'],
                input_data_file=os.path.join(config.geneformer_configs['tokenized_file_directory'],
                                             config.geneformer_configs['tokenized_file_prefix'] + '.dataset'),
                output_directory=config.geneformer_configs['embedding_output_directory'],
                output_prefix=config.geneformer_configs['embedding_output_prefix'],
                output_torch_embs=False)
        elif self.model_name == "scGPT":
            extractor = GeneEmbedding()

        return print("Invalid model name")


emb_extractor = EmbeddingExtractor("Geneformer")
emb_extractor.extract_embeddings()
