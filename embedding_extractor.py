from models.Geneformer.geneformer import EmbExtractor, TranscriptomeTokenizer

"""
Generate embeddings for given scRNAseq data.

**Input data:**

| *Required format:* raw counts scRNAseq data without feature selection as anndata file.
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "n_counts"; total read counts in that cell.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.


**Output data:**
Single cell transcriptomics embeddings.


**Usage:**
```
    emb_extractor = EmbeddingExtractor("Geneformer")
    emb_extractor.tokenize()
    emb_extractor.extract_embeddings()
```


"""

GENEFORMER_PRE_TRAINED_MODEL_PATH = "models/Geneformer/gf-20L-95M-i4096"
EXAMPLE_INPUT_DIRECTORY = "example/data/processed"
EXAMPLE_TOKENIZED_FILE_DIRECTORY = "example/data/tokenized/tokenized.dataset"
EXAMPLE_EMBEDDING_OUTPUT_DIRECTORY = "example/embedding/geneformer"


class EmbeddingExtractor:
    def __init__(self, model_name):
        """
        Initialize embedding extractor.

        :param model_name: {"Geneformer"}
            Name of the pretrained model.
        """
        assert model_name in {"Geneformer"}, "Invalid model name"
        self.model_name = model_name

    """
    Tokenize raw scRNAseq data and dump into .dateset directory.
    """

    def tokenize(self, data_directory=EXAMPLE_INPUT_DIRECTORY,
                 output_directory=EXAMPLE_TOKENIZED_FILE_DIRECTORY):
        if self.model_name == "Geneformer":
            tk = TranscriptomeTokenizer(nproc=4)
            tk.tokenize_data(data_directory,
                             output_directory,
                             "tokenized", "h5ad")

        return print("Invalid model name")

    """
    Extract transcriptomics embeddings for input scRNAseq data.
    """

    def extract_embeddings(self,
                           model_directory=GENEFORMER_PRE_TRAINED_MODEL_PATH,
                           input_data_file=EXAMPLE_TOKENIZED_FILE_DIRECTORY,
                           output_directory=EXAMPLE_EMBEDDING_OUTPUT_DIRECTORY,
                           output_prefix="embedding"):
        if self.model_name == "Geneformer":
            embex = EmbExtractor(model_type="Pretrained",
                                 num_classes=0,  # 0 for the pre-trained model
                                 emb_mode="cell",
                                 emb_layer=-1,
                                 forward_batch_size=10)
            return embex.extract_embs(
                model_directory,
                input_data_file,
                output_directory,
                output_prefix)

        return print("Invalid model name")


emb_extractor = EmbeddingExtractor("Geneformer")
# emb_extractor.tokenize()
emb_extractor.extract_embeddings()
