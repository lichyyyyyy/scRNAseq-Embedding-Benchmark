# ReadMe

## Environment SetUp

Python version 3.10.

To activate environment from `environment.txt` and please install the correct
version of Torch and TorchText in this
site: https://pytorch.org/get-started/locally.

```
# Activate the virtual environment
source .venv/bin/activate
pip install -r requirements.txt

# Deactivate the virtual environment
source .venv/bin/deactivate
```

## Included Models

| Model Name | Embedding Dimension | Repo URL                                                                                  |
|------------|---------------------|-------------------------------------------------------------------------------------------|
| Geneformer | 895                 | https://huggingface.co/ctheodoris/Geneformer                                              |
| scGPT      | 512                 | https://github.com/bowang-lab/scGPT.git                                                   |
| genePT     | 1536                | https://github.com/yiqunchen/GenePT.git  (genePT-w pre-computed gene embedding num 93800) |


