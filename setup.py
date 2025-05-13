import os

import gdown

from config import scgpt_configs, setup_configs

# Download the model file for scGPT.
# Google drive: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
os.makedirs(scgpt_configs['load_model_dir'], exist_ok=True)

print(f"Downloading the scGPT model")
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
