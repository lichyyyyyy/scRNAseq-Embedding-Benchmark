# ReadMe

## User Manual

### Step 0. Clone the repo

```angular2html
# Initialize Git LFS (once per machine):
git lfs install

# Clone the repo using either one of below commands:
git clone git@github.com:lichyyyyyy/scRNAseq-Foundation-Model-Benchmark.git
git clone https://github.com/lichyyyyyy/scRNAseq-Embedding-Benchmark.git
```

### Step 1. Activate python environment

Python version == 3.10. GCU is required.

Activate environment from `environment.txt` by below commands.

```
# Activate the virtual environment
source .venv/bin/activate
pip install -r requirements.txt

# Deactivate the virtual environment
source .venv/bin/deactivate

# Install correct version of Torch from https://pytorch.org/get-started/locally. For example:
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
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

## Example Output Structure

```angular2html
.
├───pbmc
│   ├───eval
│   ├───pre_processed
│   ├───raw_data
│   └───tokenized
│       └───geneformer.dataset

```

## Evaluation Tasks

| Category  | Tasks                | Metrics                                                                                                                            |
|-----------|----------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Zero-shot | cell type clustering | UMAP, [benchmark metrics](https://github.com/theislab/scib?tab=readme-ov-file#metrics)(NMI, ARI, ASW, graph connectivity, avg bio) |

## References

| Name       | Directory         | Embedding Dimension | Repo URL                                                                                  |
|------------|-------------------|---------------------|-------------------------------------------------------------------------------------------|
| Geneformer | models/geneformer | 895                 | https://huggingface.co/ctheodoris/Geneformer                                              |
| scGPT      | models/scGPT      | 512                 | https://github.com/bowang-lab/scGPT.git                                                   |
| genePT     | models/genePT     | 1536                | https://github.com/yiqunchen/GenePT.git  (genePT-w pre-computed gene embedding num 93800) |
| zero shot  | eval/zero_shot    | -                   | https://github.com/microsoft/zero-shot-scfoundation.git                                   |
