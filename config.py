gene_info_table = 'data/expanded_gene_info_table.csv'
raw_data_directory = 'example/data/raw'
preprocessed_data_directory = 'example/data/pre_processed'

preprocessor_configs = dict(
    gene_key_type='gene_symbol',  # The type of the gene index.
    file_format='h5ad',  # The type of input files.
)

"""
Geneformer configs
"""
geneformer_configs = dict(
    # tokenizer
    tokenized_file_directory="./example/data/tokenized/Geneformer",
    tokenized_file_prefix='tokenized',
    custom_cell_attr_names=['cell_type'],
    # embedding extractor
    load_model_dir="models/geneformer/model/",
    model_file_name="gf-20L-95M-i4096",
    embedding_output_directory="example/embedding/Geneformer",
    embedding_output_prefix="embedding",
    embedding_mode="cls"
)

"""
scGPT configs
"""
scgpt_configs = dict(
    load_model_dir='models/scGPT/model/',
    model_file_name="model.pt",
    tokenized_file_dir='example/data/tokenized/scGPT',
    embedding_output_directory="example/embedding/scGPT/",
    embedding_output_prefix="embedding_",
    custom_cell_attr_names=['cell_type'],
)
"""
genePT configs
"""
genept_configs = dict(
    load_model_dir='models/genePT/model/',
    embedding_file_name='GenePT_gene_embedding_ada_text.pickle',
    genept_w_embedding_output_directory="example/embedding/genePT_w/",
    genept_s_embedding_output_directory="example/embedding/genePT_s/",
    genept_s_openai_model_name='text-embedding-ada-002',
    embedding_output_prefix="embedding_",
    openai_api_key='',  # remember to set your open AI API key!
    custom_cell_attr_names=['cell_type'],
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
