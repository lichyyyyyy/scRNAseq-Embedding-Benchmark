"""
Geneformer configs
"""
geneformer_configs = dict(
    preprocess_data_directory="./example/data/processed",
    tokenized_file_directory="./example/data/tokenized/Geneformer/tokenized.dataset",
    tokenized_file_prefix='tokenized'
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
    gene_id_key='gene_ids',
    pad_value=-2,
    load_model_dir='./data/scGPT_model',
    model_file_name="model.pt",
    raw_data_directory='./example/data/raw',
    tokenized_file_path='./example/data/tokenized/scGPT/tokenized_data.pt'
)

"""
set up configs
"""
setup_configs = dict(
    scgpt_model_gdrive_file_id="14AebJfGOUF047Eg40hk57HCtrb0fyDTm",
)
