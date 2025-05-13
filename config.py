"""
Geneformer configs
"""
geneformer_configs = dict(
    # pre-processor
    preprocess_data_directory="./example/data/processed",
    # tokenizer
    tokenized_file_directory="./example/data/tokenized/Geneformer",
    tokenized_file_prefix='tokenized',
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
    max_seq_len=3001,
    cls_token="<cls>",
    pad_token="<pad>",
    include_zero_gene=False,
    input_layer_key='X_binned',
    gene_id_key='index', #'gene_ids',
    pad_value=-2,
    load_model_dir='models/scGPT/model/',
    model_file_name="model.pt",
    raw_data_directory='example/data/raw',
    tokenized_file_dir='example/data/tokenized/scGPT'
)

"""
set up configs
"""
setup_configs = dict(
    scgpt_model_gdrive_file_id="14AebJfGOUF047Eg40hk57HCtrb0fyDTm",
    scgpt_vocab_gdrive_file_id="1H3E_MJ-Dl36AQV6jLbna2EdvgPaqvqcC",
    scgpt_args_gdrive_file_id="1hh2zGKyWAx3DyovD30GStZ3QlzmSqdk1",
)
