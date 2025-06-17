# The gene info table.
gene_info_table = 'data/expanded_gene_info_table.csv'
# The directory to store raw data.
# Note: Currently only accept Anndata files with gene names in `vars`. The supported gene naming system includes
# "gene_symbol", "ensembl_id", "entrez_id", "refseq_id".
raw_data_directory = 'example/data/raw'
# The directory to store preprocessed data.
preprocessed_data_directory = 'example/data/pre_processed'

"""
Pre processor configs
"""
preprocessor_configs = dict(
    # The column name of gene ID in `adata.var`. If the gene ID is the index, input `index`.
    gene_id_col_name='ensembl_id',
    # The type of gene naming system in the gene ID: {"gene_symbol", "ensembl_id", "entrez_id", "refseq_id"}.
    gene_id_type='ensembl_id',
    # The input file format. Currently only Anndata is supported.
    file_format='h5ad',
    # Whether to keep batch key. If true, the input file directory name will be used as the batch key and stored under
    # `adata.obs.batch_key`.
    keep_batch_key=True,
    # Map of cell attribute labels in `obs` to keep. Key is the name in original file, value is the name in
    # pre-processed file.
    custom_cell_attr_names={'cell_type': 'cell_type'},
)

"""
Geneformer configs
"""
geneformer_configs = dict(
    # The output tokenized file directory.
    tokenized_file_directory="./example/data/tokenized/Geneformer",
    # The output tokenized filename prefix.
    tokenized_file_prefix='tokenized',
    # List of cell attribute labels to keep.
    custom_cell_attr_names=['cell_type', 'batch_key'],
    # Directory of the Geneformer pre-trained model.
    load_model_dir="models/geneformer/model/",
    # Name of the Geneformer pre-trained model.
    model_file_name="gf-20L-95M-i4096",
    # The cell embedding mode, using `cls` for `gf-20L-95M-i4096` model.
    embedding_mode="cls",
    # The output embedding file directory.
    embedding_output_directory="example/embedding/Geneformer/",
    # The output embedding file name.
    embedding_output_filename="cell_embeddings"
)

"""
scGPT configs
"""
scgpt_configs = dict(
    # Directory of the scGPT pre-trained model.
    load_model_dir='models/scGPT/model/',
    # File name of the model weights.
    model_file_name="model.pt",
    # The output embedding file directory.
    embedding_output_directory="example/embedding/scGPT/",
    # The output embedding file name.
    embedding_output_filename="cell_embeddings",
    # List of cell attribute labels to keep.
    custom_cell_attr_names=['cell_type', 'batch_key'],
)
"""
genePT configs
"""
genept_configs = dict(
    # Directory of the genePT pre-trained gene embedding file.
    load_model_dir='models/genePT/model/',
    # The file name of genePT pre-trained gene embedding file.
    embedding_file_name='GenePT_gene_embedding_ada_text.pickle',
    # The output embedding file directory for genePT-w.
    genept_w_embedding_output_directory="example/embedding/genePT_w/",
    # The output embedding file directory for genePT-s.
    genept_s_embedding_output_directory="example/embedding/genePT_s/",
    # The used openai model name.
    genept_s_openai_model_name='text-embedding-ada-002',
    # The output embedding file name.
    embedding_output_filename="cell_embeddings",
    # OpenAI api key.
    openai_api_key='',  # remember to set your open AI API key!
    # List of cell attribute labels to keep.
    custom_cell_attr_names=['cell_type', 'batch_key'],
)

"""
set up configs
"""
setup_configs = dict(
    scgpt_model_gdrive_file_id="14AebJfGOUF047Eg40hk57HCtrb0fyDTm",
    scgpt_vocab_gdrive_file_id="1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC",
    scgpt_args_gdrive_file_id="1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1",
    genept_embedding_gdrive_file_id='1gAc1XNIb4fnAwopc7tTbZl2YiYolaDMA'
)
