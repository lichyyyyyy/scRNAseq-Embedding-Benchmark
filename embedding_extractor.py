import glob
import os
import pickle
import platform
from pathlib import Path

import anndata as ad
import numpy as np
import openai
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

from config import geneformer_configs, preprocessed_data_directory, raw_data_directory, genept_configs, scgpt_configs
from models.geneformer import TranscriptomeTokenizer
from models.scGPT import embed_data

"""
Generate embeddings for given scRNAseq data.

**Input data:**
A directory of pre processed scRNAseq data in Anndata format.

**Output data:**
A directory of single cell transcriptomics embeddings in cvs or Anndata format.
If in Anndata, the embedding is stored in `adata.obsm[X_{model_name}]`.



**Usage:**
```
emb_extractor = EmbeddingExtractor("scGPT", output_file_type='h5ad')
emb_extractor.tokenize()
emb_extractor.extract_embeddings()
```

"""


class EmbeddingExtractor:
    def __init__(self, model_name, output_file_type):
        """
        Initialize embedding extractor.

        :param model_name: {"Geneformer", "scGPT", "genePT-w", "genePT-s"}
            Name of the pretrained model.
        :param output_file_type: {"csv", "h5ad"}
            Output file type.
        """
        assert model_name in {"Geneformer", "scGPT", "genePT-w", "genePT-s"}, "Invalid model name"
        assert output_file_type in {"csv", "h5ad"}, "Invalid output file type"
        self.model_name = model_name
        self.output_file_type = output_file_type

    def tokenize(self):
        """
        Tokenize the pre-processed scRNAseq data for given model.
        """
        if self.model_name == "Geneformer":
            custom_attr_name_dict = {}
            for attr_name in geneformer_configs['custom_cell_attr_names']:
                custom_attr_name_dict[attr_name] = attr_name
            tk = TranscriptomeTokenizer(
                custom_attr_name_dict=custom_attr_name_dict,
                special_token=True)
            return tk.tokenize_data(
                data_directory=preprocessed_data_directory,
                output_directory=geneformer_configs['tokenized_file_directory'],
                output_prefix=geneformer_configs['tokenized_file_prefix'], file_format="h5ad")

        elif self.model_name in {"scGPT", "genePT-w", "genePT-s"}:
            return print("Tokenizer skipped")

        return print("Invalid model name")

    def extract_embeddings(self):
        """
        Extract transcriptomics embeddings for input scRNAseq data.
        """

        def write_to_csv(dest_path: str, custom_cell_attr_names: list, embeddings: np.ndarray,
                         embedding_attrs: pd.DataFrame):
            output_df = pd.DataFrame()
            for attr_name in custom_cell_attr_names:
                output_df = pd.concat([output_df, embedding_attrs[attr_name].to_frame()], axis=1)
            output_df = pd.concat([output_df, pd.DataFrame(embeddings, index=output_df.index)], axis=1)
            output_df.to_csv(dest_path, index=False)
            print(f"Output embedding in {dest_path}")

        if self.model_name == "Geneformer":
            print("Extracting Geneformer embeddings")
            extractor = EmbExtractor(model_type="Pretrained",
                                     num_classes=0,  # 0 for the pre-trained model
                                     emb_mode=geneformer_configs['embedding_mode'],  # {"cls", "cell", "gene"}
                                     max_ncells=None,  # If None, will extract embeddings from all cells.
                                     emb_layer=-1,
                                     forward_batch_size=10,
                                     emb_label=geneformer_configs['custom_cell_attr_names'],
                                     nproc=4)
            embedding = extractor.extract_embs(
                model_directory=os.path.join(geneformer_configs['load_model_dir'],
                                             geneformer_configs['model_file_name']),
                input_data_file=os.path.join(geneformer_configs['tokenized_file_directory'],
                                             geneformer_configs['tokenized_file_prefix'] + '.dataset'),
                output_directory=geneformer_configs['embedding_output_directory'],
                output_prefix=geneformer_configs['embedding_output_filename'],
                output_torch_embs=False)
            output_path = geneformer_configs['embedding_output_directory'] + geneformer_configs[
                'embedding_output_filename'] + '.' + self.output_file_type
            if self.output_file_type == 'csv':
                embedding.to_csv(output_path)
            elif self.output_file_type == 'h5ad':
                obs = pd.DataFrame(
                    embedding.loc[:, embedding.columns.isin(geneformer_configs['custom_cell_attr_names'])],
                    index=embedding.index)
                obsm = pd.DataFrame(
                    embedding.loc[:, ~embedding.columns.isin(geneformer_configs['custom_cell_attr_names'])],
                    index=embedding.index)
                adata = ad.AnnData(obs=obs)
                adata.obsm["X_" + self.model_name] = obsm.to_numpy()
                adata.write(output_path)

            return print(f"Output embedding in {geneformer_configs['embedding_output_directory']}")

        elif self.model_name == "scGPT":
            print("Extracting scGPT embeddings")
            os.makedirs(scgpt_configs['embedding_output_directory'], exist_ok=True)
            for file_path in glob.glob(preprocessed_data_directory + f"/*.h5ad"):
                print(f"Embedding {file_path}")
                embed_adata = embed_data(
                    adata_or_file=file_path,
                    model_dir=scgpt_configs['load_model_dir'],
                    gene_col="gene_symbol",
                    max_length=1200,
                    batch_size=64,
                    obs_to_save=None,
                    use_fast_transformer=(platform.system() == "Linux"),
                    return_new_adata=False)
                output_path = scgpt_configs['embedding_output_directory'] + scgpt_configs[
                    'embedding_output_prefix'] + Path(
                    file_path).stem + '.' + self.output_file_type
                if self.output_file_type == 'csv':
                    write_to_csv(output_path, scgpt_configs['custom_cell_attr_names'], embed_adata.obsm['X_scGPT'],
                                 embed_adata.obs)
                elif self.output_file_type == 'h5ad':
                    embed_adata.write(output_path)
            return None

        elif self.model_name == "genePT-w":
            print("Extracting gene PT-W embeddings")
            with open(os.path.join(genept_configs['load_model_dir'], genept_configs['embedding_file_name']),
                      'rb') as fp:
                genept_embedding_data = pickle.load(fp)
            for file_path in glob.glob(preprocessed_data_directory + f"/*.h5ad"):
                print(f"Embedding {file_path}")
                adata = sc.read_h5ad(file_path)
                gene_names = list(adata.var['gene_symbol'])
                count_missing = 0
                EMBED_DIM = 1536  # embedding dim from GPT-3.5
                lookup_embed = np.zeros(shape=(len(gene_names), EMBED_DIM))
                for i, gene in enumerate(gene_names):
                    if gene in genept_embedding_data:
                        lookup_embed[i] = genept_embedding_data[gene]
                    else:
                        count_missing += 1
                adata_torch = torch.tensor(adata.X.toarray() if issparse(adata.X) else adata.X, dtype=torch.float32)
                lookup_embed_torch = torch.tensor(lookup_embed, dtype=torch.float32)
                genePT_w_emebed = torch.divide(torch.matmul(adata_torch, lookup_embed_torch), len(gene_names)).numpy()
                output_path = genept_configs['genept_w_embedding_output_directory'] + genept_configs[
                    'embedding_output_prefix'] + Path(
                    file_path).stem + '.' + self.output_file_type
                if self.output_file_type == 'csv':
                    write_to_csv(output_path, scgpt_configs['custom_cell_attr_names'], genePT_w_emebed,
                                 adata.obs)
                elif self.output_file_type == 'h5ad':
                    adata.obsm['X_' + self.model_name] = genePT_w_emebed
                    adata.write(output_path)
                print(f"Unable to match {count_missing} out of {len(gene_names)} genes in the GenePT-w embedding")
                print(f"Output embedding in {output_path}")
            return None

        elif self.model_name == "genePT-s":
            print("Extracting gene PT-s embeddings")

            def get_seq_embed_gpt(X, gene_names, prompt_prefix="", trunc_index=None):
                n_genes = X.shape[1]
                if trunc_index is not None and not isinstance(trunc_index, int):
                    raise Exception('trunc_index must be None or an integer!')
                elif isinstance(trunc_index, int) and trunc_index >= n_genes:
                    raise Exception('trunc_index must be smaller than the number of genes in the dataset')
                get_test_array = []
                for cell in X:
                    zero_indices = np.where(cell.toarray() == 0)[0]
                    gene_indices = np.argsort(cell)[::-1]
                    filtered_genes = gene_indices[~np.isin(gene_indices, list(zero_indices))]
                    if trunc_index is not None:
                        get_test_array.append(np.array(gene_names[filtered_genes])[0:trunc_index])
                    else:
                        get_test_array.append(np.array(gene_names[filtered_genes]))
                get_test_array_seq = [prompt_prefix + ' '.join(x) for x in get_test_array]
                return (get_test_array_seq)

            def get_gpt_embedding(text, model=genept_configs['genept_s_openai_model_name']):
                text = text.replace("\n", " ")
                return np.array(openai.Embedding.create(input=[text], model=model)['data'][0]['embedding'])

            openai.api_key = genept_configs['openai_api_key']
            os.makedirs(genept_configs['genept_s_embedding_output_directory'], exist_ok=True)
            for file_path in glob.glob(raw_data_directory + f"/*.h5ad"):
                print(f"Embedding {file_path}")
                adata = sc.read_h5ad(file_path)
                ranked_cells_data = get_seq_embed_gpt(adata.X,
                                                      np.array(adata.var.index),
                                                      prompt_prefix='A cell with genes ranked by expression: ',
                                                      trunc_index=None)
                embeddings = []
                for i, x in enumerate(ranked_cells_data):
                    embeddings.append(get_gpt_embedding(x))
                    if i % 100 == 0:
                        print(f"Processing {i} out of {adata.obs.shape[0]} cells...")
                output_path = genept_configs['genept_s_embedding_output_directory'] + genept_configs[
                    'embedding_output_prefix'] + Path(
                    file_path).stem + '.' + self.output_file_type
                if self.output_file_type == 'csv':
                    write_to_csv(output_path, genept_configs['custom_cell_attr_names'], np.array(embeddings), adata.obs)
                elif self.output_file_type == 'h5ad':
                    adata.obsm['X_' + self.model_name] = embeddings
                    adata.write(output_path)
                print(f"Output embedding in {output_path}")
            return None

        return print("Invalid model name")


emb_extractor = EmbeddingExtractor("Geneformer", output_file_type='h5ad')
emb_extractor.tokenize()
emb_extractor.extract_embeddings()
