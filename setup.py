import os
import shutil
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download

from config import scgpt_configs, setup_configs, geneformer_configs, genept_configs


def download_scgpt_model():
    """
    Download the model file for scGPT.
    Google drive: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
    """
    os.makedirs(scgpt_configs['load_model_dir'], exist_ok=True)

    model_url = f"https://drive.google.com/uc?id={setup_configs['scgpt_model_gdrive_file_id']}"
    args_url = f"https://drive.google.com/uc?id={setup_configs['scgpt_args_gdrive_file_id']}"
    vocab_url = f"https://drive.google.com/uc?id={setup_configs['scgpt_vocab_gdrive_file_id']}"
    gdown.download(model_url, os.path.join(scgpt_configs['load_model_dir'],
                                           scgpt_configs['model_file_name']),
                   quiet=False)
    gdown.download(args_url,
                   os.path.join(scgpt_configs['load_model_dir'], 'args.json'),
                   quiet=False)
    gdown.download(vocab_url,
                   os.path.join(scgpt_configs['load_model_dir'], 'vocab.json'),
                   quiet=False)

    print(f"✅ File downloaded to: {scgpt_configs['load_model_dir']}")


def download_geneformer_model():
    local_cache_path = snapshot_download(repo_id="ctheodoris/Geneformer",
                                         allow_patterns=[f"{geneformer_configs['model_file_name']}/*"])

    shutil.copytree(Path(local_cache_path) / geneformer_configs['model_file_name'],
                    Path(geneformer_configs['load_model_dir']) / geneformer_configs['model_file_name'],
                    dirs_exist_ok=True)

    print(f"✅ File downloaded to: {scgpt_configs['load_model_dir']}")


def download_genept_data():
    """
    Download the genePT pre-computed gene embeddings.
    Please download the pre-computed gene embeddings here: https://zenodo.org/records/10833191
    """
    os.makedirs(genept_configs['load_model_dir'], exist_ok=True)

    embedding_url = f"https://drive.google.com/uc?id={setup_configs['genept_embedding_gdrive_file_id']}"
    gdown.download(embedding_url, os.path.join(genept_configs['load_model_dir'],
                                               genept_configs['embedding_file_name']),
                   quiet=False)
    print(f"✅ File downloaded to: {genept_configs['load_model_dir']}")


download_genept_data()
