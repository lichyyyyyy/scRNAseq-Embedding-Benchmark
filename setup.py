import os

import gdown

from config import scgpt_configs, setup_configs

# Download the model file for scGPT.
os.makedirs(scgpt_configs['load_model_dir'], exist_ok=True)
output_path = os.path.join(scgpt_configs['load_model_dir'],
                          scgpt_configs['model_file_name'])

url = f"https://drive.google.com/uc?id={setup_configs['scgpt_model_gdrive_file_id']}"
print(f"Downloading scGPT model from official Google Drive: {url}")
gdown.download(url, output_path, quiet=False)

print(f"✅ File downloaded to: {output_path}")
