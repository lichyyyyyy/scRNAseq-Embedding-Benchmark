from models.Geneformer.geneformer import EmbExtractor, TranscriptomeTokenizer

"""
Generate embeddings for given scRNAseq data.

**Input data:**

| *Required format:* raw counts scRNAseq data without feature selection as .loom or anndata file.
| *Required row (gene) attribute:* "ensembl_id"; Ensembl ID for each gene.
| *Required col (cell) attribute:* "n_counts"; total read counts in that cell.

| *Optional col (cell) attribute:* "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria.
| *Optional col (cell) attributes:* any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below.

"""

# class EmbeddingExtractor:
# specify which model to use

class Geneformer:
   tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=4)
   tk.tokenize_data("data_directory", "output_directory", "output_prefix")
    embex = EmbExtractor(model_type="Pretrained",
                     num_classes=0,  # 0 for the pre trained model
                     emb_mode="cell",
                     filter_data={"cell_type": ["cardiomyocyte"]},
                     max_ncells=1000,
                     emb_layer=-1,
                     emb_label=["disease", "cell_type"],
                     labels_to_plot=None)