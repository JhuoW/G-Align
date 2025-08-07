# adapted from github.com/Zehong-Wang/GFT

import torch
from sentence_transformers import SentenceTransformer
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
from torchmetrics import AveragePrecision, AUROC
import numpy as np
from tqdm.autonotebook import trange
import gc
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from torch_geometric.data import InMemoryDataset
import os.path as osp
from utils.utils import safe_mkdir, pth_safe_load
import torch_geometric as pyg
from collections.abc import Mapping
import pandas as pd

class SentenceEncoder:
    def __init__(self, name, root="cache_data/pretrained_llm_model", batch_size=1, multi_gpu=False):
        self.name = name
        self.root = root
        self.device, self.gpu_ids = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = None
        self.tokenizer = None
    
    def get_model(self):
        if self.name == "ST": # sentence-transformers
            print("Loading SentenceTransformer model ulti-qa-distilbert-cos-v1...")
            self.model = SentenceTransformer("multi-qa-distilbert-cos-v1", device=self.device, cache_folder=self.root)
            self.encode = self.ST_encode
        elif self.name == 'llama2_7b':
            model_name = "meta-llama/Llama-2-7b-hf"
            print(f"Loading Llama model {model_name}...")
            self.model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root).to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode
                
        elif self.name == "llama2_13b":
            # model_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'llama-2-7b')
            model_name = "meta-llama/Llama-2-13b-hf"
            print(f"Loading Llama model {model_name}...")
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=self.root)
            tokenizer.padding_side = "right"
            tokenizer.truncation_side = 'right'
            self.tokenizer = tokenizer
            self.encode = self.llama_encode           
        elif self.name == "e5":
            model_name = "intfloat/e5-large-v2"
            print(f"Loading E5 model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.root)
            model = AutoModel.from_pretrained(model_name, cache_dir=self.root)
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
            self.encode = self.e5_encode

        elif self.name == "roberta":
            print("Loading SentenceTransformer model roberta-base-nli-stsb-mean-tokens...")
            self.model = SentenceTransformer("sentence-transformers/roberta-base-nli-stsb-mean-tokens",
                device=self.device, cache_folder=self.root, )
            self.encode = self.ST_encode
        else:
            raise ValueError(f"Unknown language model: {self.name}.")

    def encode(self, texts, to_tensor=True):
        raise NotImplementedError("Not define llm encoder yet.")

    def ST_encode(self, sentences, to_tensor = True):
        if self.multi_gpu:
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(sentences, pool=pool, batch_size=self.batch_size, )
            embeddings = torch.from_numpy(embeddings)
        else:
            embeddings = self.model.encode(sentences, batch_size=self.batch_size, 
                                           show_progress_bar=True, convert_to_tensor=to_tensor, convert_to_numpy= not to_tensor)
        return embeddings
    
    def llama_encode(self, texts, to_tensor=True):

        # Add EOS token for padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                input_ids = self.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).input_ids.to(self.device)
                transformer_output = self.model(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"]
                # No gradients on word_embeddings
                word_embeddings = transformer_output[-1].detach()
                sentence_embeddings = word_embeddings.mean(dim=1)
                all_embeddings.append(sentence_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def e5_encode(self, texts, to_tensor=True):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                batch_dict = self.tokenizer(sentences_batch, padding="longest", truncation=True, return_tensors='pt')
                for item, value in batch_dict.items():
                    batch_dict[item] = value.to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = embeddings.detach()
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(all_embeddings.size())
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        if self.tokenizer is not None:
            self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def refine_dataset(dataset):
    # works for molecule graphs
    if dataset.data.get("node_embs") is not None:
        dataset.data.node_text_feat = dataset.data.node_embs
        dataset.data.node_embs = None
    if dataset.data.get("edge_embs") is not None:
        dataset.data.edge_text_feat = dataset.data.edge_embs
        dataset.data.edge_embs = None
    if dataset.data.get("pretrain_edge_index") is not None:
        dataset.data.edge_index = dataset.data.pretrain_edge_index
        dataset.data.pretrain_edge_index = None
    if dataset.data.get("y") is not None:
        dataset.data.y.squeeze_() 
    return dataset

def span_node_and_edge_idx(dataset):
    # Define node index
    if dataset.data.x.ndim == 1:
        return dataset
    
    if dataset.ds_name in ['cora', 'pubmed']:
        # num_nodes = dataset.data.x.shape[0]
        dataset.data.xn = dataset.data.x
    elif dataset.ds_name in ['arxiv', 'wikics']:
        dataset.data.xn = dataset.data.node_text_feat


    # num_nodes = dataset.data.x.shape[0]
    # dataset.data.x = torch.arange(num_nodes)

    # Define edge index
    if dataset.ds_name in ['cora', 'pubmed']:
        num_edge_types = 1
    elif dataset.ds_name in ['arxiv', 'wikics']:
        num_edge_types = dataset.data.edge_text_feat.shape[0]  # 1个边类型
    num_edges = dataset.data.edge_index.shape[1]   

    if num_edge_types == 1:
        dataset.data.xe = torch.zeros([num_edges], dtype=torch.long)  # 如果只有一个边类型，那么所有边的特征初始化为0
    else:
        dataset.data.xe = dataset.data.edge_types
    return dataset

def filter_unnecessary_attrs(dataset):
    keys = ['x',
            'xn',
            'xe',
            'edge_index',
            'y'
            ]
    for k, v in dataset.data.to_dict().items():
        if k not in keys:
            dataset.data[k] = None
    return dataset
