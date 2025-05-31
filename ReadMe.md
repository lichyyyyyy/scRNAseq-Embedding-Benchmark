# ReadMe

## User Manual

### Step 0. Clone the repo

```angular2html
# Initialize Git LFS (once per machine):
git lfs install

git clone https://github.com/lichyyyyyy/scRNAseq-Embedding-Benchmark.git
```

### Step 1. Activate python environment

Python version == 3.10.

Activate environment from `environment.txt` by below commands. And please install the correct
version of Torch and TorchText in this
site: https://pytorch.org/get-started/locally (for example,
`pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121`).

```
# Activate the virtual environment
source .venv/bin/activate
pip install -r requirements.txt

# Deactivate the virtual environment
source .venv/bin/deactivate
```

### Step 2. Download pre-computed models

Run the below command to download pre-computed models into configured paths.

```angular2html
python setup.py
```

### Step 3. Pre process input data

Note: Currently only support Anndata files (.h5ad) as the input.

Update the `raw_data_directory`, `preprocessed_data_directory` and `preprocessor_configs` in `config.py` file. Run the
below command to pre-process raw data:

```angular2html
python pre_processor.py
```

### Step 4. Extract cell embeddings

Specify the desired foundation model in `EmbeddingExtractor` in the `embedding_extractor.py` file and update
corresponding configs in `config.py` file. For genePT-s model, don't forget to apply OpenAI api key from
this [link](https://openai.com/index/openai-api/).

```angular2html
python embedding_extractor.py
```

## Repo Structure

```angular2html
.
├── ReadMe.md
├── example
│         ├── data
│         │         ├── pre_processed
│         │         │         └── demo_cells_2k.h5ad
│         │         ├── raw
│         │         │         └── demo_cells_2k.h5ad
│         │         └── tokenized
│         │             └── Geneformer
│         │                 └── tokenized.dataset
│         │                     ├── data-00000-of-00001.arrow
│         │                     ├── dataset_info.json
│         │                     └── state.json
│         └── embedding
│             ├── genePT_s
│             │         └── embedding_demo_cells_2k.csv
│             ├── genePT_w
│             │         └── embedding_demo_cells_2k.csv
│             ├── geneformer
│             │         └── embedding.csv
│             └── scGPT
│                 └── embedding_demo_cells_2k.csv
├── models     <----------- Files downloaded from original repos.
│         ├── genePT
│         │         └── model
│         │             └── GenePT_gene_embedding_ada_text.pickle
│         ├── geneformer
│         │         ├── __init__.py
│         │         ├── emb_extractor.py
│         │         ├── ensembl_mapping_dict_gc95M.pkl
│         │         ├── gene_median_dictionary_gc95M.pkl
│         │         ├── gene_name_id_dict_gc95M.pkl
│         │         ├── model
│         │         │         └── gf-20L-95M-i4096
│         │         │             ├── config.json
│         │         │             ├── generation_config.json
│         │         │             ├── model.safetensors
│         │         │             └── training_args.bin
│         │         ├── perturber_utils.py
│         │         ├── token_dictionary_gc95M.pkl
│         │         └── tokenizer.py
│         └── scGPT
│             ├── __init__.py
│             ├── cell_emb.py
│             ├── data_collator.py
│             ├── dsbn.py
│             ├── gene_tokenizer.py
│             ├── grad_reverse.py
│             ├── grn.py
│             ├── model
│             │         ├── args.json
│             │         ├── model.pt
│             │         └── vocab.json
│             ├── model.py
│             ├── preprocess.py
│             └── util.py
├── config.py
├── embedding_extractor.py
├── data
│         └── expanded_gene_info_table.csv
├── pre_processor.py
├── requirements.txt
└── setup.py

```

## Included Models

| Model Name | Embedding Dimension | Repo URL                                                                                  |
|------------|---------------------|-------------------------------------------------------------------------------------------|
| Geneformer | 895                 | https://huggingface.co/ctheodoris/Geneformer                                              |
| scGPT      | 512                 | https://github.com/bowang-lab/scGPT.git                                                   |
| genePT     | 1536                | https://github.com/yiqunchen/GenePT.git  (genePT-w pre-computed gene embedding num 93800) |