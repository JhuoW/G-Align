"""
Domain Embedding and FiLM Hyper-Network for Pre-training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch import Tensor
import copy
from pathlib import Path
import os
from torch_geometric.utils import subgraph
from model.FiLM import DomainFiLM
import numpy as np
import ast
try:
    from sklearn.decomposition import PCA
except ImportError: 
    PCA = None

def _iter_params(model, require_grad_only=True): 
    """Flatten / Unflatten model parameters."""
    for p in model.parameters():  # 迭代去除有梯度的参数
        if require_grad_only and not p.requires_grad:
            continue
        yield p

def flatten_grads(model: nn.Module, require_grad_only = True):
    grads = []
    for p in _iter_params(model, require_grad_only):  # 遍历所有有梯度的参数
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return torch.empty(0, device=next(model.parameters()).device)
    return torch.cat(grads, dim=0)


class GraphProbContrastLoss(nn.Module):
    def __init__(self, mask_ratio,
                       recon_weight,
                       neigh_weight,
                       detach_embed):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight
        self.neigh_weight = neigh_weight
        self.detach_embed = detach_embed
    
    def forward(self, data, embed):
        x, edge_index = data.x, data.edge_index
        device = x.device
        num_nodes, feat_dim = x.size()
        # masked reconstruction
        num_mask = max(1, int(self.mask_ratio * num_nodes))
        mask_idx = torch.randperm(num_nodes, device=device)[:num_mask]
        x_masked = x[mask_idx]
        h_masked = embed[mask_idx]
        if self.detach_embed:
            h_masked = h_masked.detach()
        recon = F.linear(h_masked, weight = torch.randn(feat_dim, h_masked.size(1), device=device) * 0.01)
        if edge_index.numel() > 0:
            row, col = edge_index
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1.0)
            neigh_mean = torch.zeros_like(embed)
            neigh_mean.index_add_(0, row, embed[col])
            neigh_mean = neigh_mean / deg.unsqueeze(-1)
        else:
            neigh_mean = embed
        recon_pred = neigh_mean[mask_idx][:,:feat_dim] if neigh_mean.size(1) >= feat_dim else F.pad(neigh_mean[mask_idx], (0, feat_dim - neigh_mean.size(1)))
        recon_loss = F.mse_loss(recon, recon_pred, reduction='mean')

        if edge_index.numel() == 0:
            neigh_loss = torch.tensor(0.0, device=device)
        else:
            row, col = edge_index
            diff = embed[row] - embed[col]
            neigh_loss = diff.pow(2).sum(dim=-1).mean()
        return self.recon_weight * recon_loss + self.neigh_weight * neigh_loss

class DomainEmbeddingExtractor:
    """Compute PCA fingerprints Embedding e_i for each domain (graph) in a combined Data.
    
    IMPORTANT: The backbone used here should be a FROZEN copy of the initial parameters θ₀.
    This ensures consistent domain embeddings across training and inference.
    """    
    def __init__(self, backbone:nn.Module, cfg):
        """
        backbone: Backbone GNN model to extract node embeddings.
        """
        self.backbone = backbone
        self.cfg = cfg
        pretrain_ds_names = cfg.pretrain.pretrain_datasets  # ['pubmed', 'arxiv', 'wikics']
        if cfg.Fingerprint.task == 'node_cls':
            if cfg.Fingerprint.loss_type == 'contrastive':
                self.prob_loss = GraphProbContrastLoss(
                    mask_ratio=cfg.Fingerprint.contrast_loss.probe_mask_ratio,
                    recon_weight=cfg.Fingerprint.contrast_loss.probe_recon_weight,
                    neigh_weight=cfg.Fingerprint.contrast_loss.probe_neigh_weight,
                    detach_embed=cfg.Fingerprint.detach_embed
                )
            elif cfg.Fingerprint.loss_type == 'ce':
                self.prob_loss = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type {cfg.Fingerprint.loss_type} for Fingerprint")

        if isinstance(pretrain_ds_names, list):
            ds_names_str = '_'.join(pretrain_ds_names) # dsname1_dsname2_...
        else:
            ds_names_str = '_'.join(ast.literal_eval(str(pretrain_ds_names)))

        fingerprint_dir = os.path.join(cfg.dirs.fingerprint_storage, ds_names_str)
        self.cache_dir = Path(fingerprint_dir)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached = False
        self._e = None
        self._B = None
        self._theta0 = None
        self._try_load_cache()  

    def _pth(self, name):
        return self.cache_dir/ name if self.cache_dir else None

    def _try_load_cache(self):
        if not self.cache_dir: 
            return
        if (self._pth('B.pt').exists() and 
            self._pth('PreDomainEmbedding.pt').exists() and 
            self._pth('theta0.pt').exists()):
            self._B = torch.load(self._pth('B.pt'), weights_only=False)
            self._e = torch.load(self._pth('PreDomainEmbedding.pt'), weights_only=False)
            self._theta0 = torch.load(self._pth('theta0.pt'), weights_only=False)
            self.backbone.load_state_dict(self._theta0, strict=False)
            self._cached = True


    def _device(self):
        if self.cfg.Fingerprint.device is not None:
            return torch.device(self.cfg.Fingerprint.device)
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _maybe_cache_path(self, pretrain_mode:str ,name:str): # persist fingerprints to disk
        if self.cfg.dirs.fingerprint_storage is None:
            return None
        os.makedirs(self.cfg.dirs.fingerprint_storage, exist_ok=True)
        return os.path.join(self.cfg.dirs.fingerprint_storage, pretrain_mode, name)
    
    def _save_cache(self):
        if not self.cache_dir: 
            return
        torch.save(self._B.cpu(), self._pth('B.pt'))
        torch.save(self._e.cpu(), self._pth('PreDomainEmbedding.pt'))
        torch.save(self._theta0, self._pth('theta0.pt'))
        self._cached = True

    @torch.no_grad()
    def _embed_nodes(self, data: Data) -> Tensor:
        """ Forward all nodes through the frozen backbone GNN to get node embeddings. 
        Returns: [N, h]
        """
        self.backbone.eval()
        device = self._device()
        data = data.to(device)
        h, g = self.backbone(data)
        return h.detach(), g.detach()

    def _domain_subgraph(self, data:Data, idx: int):
        graph_mask = (data.batch == idx) # extract for nodes in the idx-th graph
        edge_mask = graph_mask[data.edge_index[0]] & graph_mask[data.edge_index[1]]
        node_idx = graph_mask.nonzero(as_tuple=False).view(-1) # node idx in the combined graph
        if node_idx.numel() == 0:
            raise ValueError(f"Domain {idx} has no nodes")
        edge_index_i, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True) 
        x_i = data.x[graph_mask]
        y_i = data.y[graph_mask]
        xe_i = data.xe[edge_mask] if hasattr(data, 'xe') else None
        return x_i, edge_index_i, node_idx, y_i, x_i.shape[0], xe_i
    
    def _sample_nodes(self, x, edge_index, y, max_nodes):
        num_nodes = x.shape[0]
        if num_nodes <= max_nodes:
            return x, edge_index, y
        perm = torch.randperm(num_nodes, device=x.device)[:max_nodes]  # sample max_nodes 节点
        new_idx = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        new_idx[perm] = torch.arange(max_nodes, device=x.device)
        row,col = edge_index
        keep_mask = (new_idx[row] >= 0) & (new_idx[col] >= 0)  

        row = new_idx[row[keep_mask]]
        col = new_idx[col[keep_mask]]

        return x[perm], torch.stack([row, col], dim=0), y[perm]
    
    def _probe_grad4domain(self, data, H_all, domain_idx):
        """Compute gradient probe vector Δθ_i for domain_idx."""
        device = self._device()
        model = self.backbone.to(device)
        model.train(False)

        # for param in model.parameters():
        #     param.requires_grad = True

        if self._theta0 is None:
            self._theta0 = {k:v.clone().cpu() for k, v in self.backbone.state_dict().items()}


        x_i, edge_index_i, node_idx, y_i, num_nodes, xe_i = self._domain_subgraph(data, domain_idx)
        x_i, edge_index_i, y_i = self._sample_nodes(x_i, edge_index_i, y_i, self.cfg.Fingerprint.max_nodes if self.cfg.Fingerprint.get('max_nodes', None) is  not None else num_nodes)

        x_i = x_i.to(device)
        edge_index_i = edge_index_i.to(device)
        y_i = y_i.to(device)
        xe_i = xe_i.to(device) if xe_i is not None else None
        domain_graph = Data(x=x_i, 
                            edge_index=edge_index_i, 
                            y=y_i, 
                            xe = xe_i,
                            batch=torch.zeros(num_nodes, dtype=torch.long, device=device))
        H_i, _ = model(domain_graph)  # [N_i, num_classes]

        # compute probe Loss
        if self.cfg.Fingerprint.loss_type == 'ce':
            loss = self.prob_loss(H_i, y_i)
        elif self.cfg.Fingerprint.loss_type == 'contrastive':
            loss = self.prob_loss(domain_graph, H_i)
        else:
            raise ValueError(f"Unknown loss type {self.cfg.Fingerprint.loss_type} for Fingerprint")
        
        # if not loss.requires_grad:
        #     raise RuntimeError("Loss does not require gradients. Check model configuration.")
        loss.requires_grad = True  
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_vec = flatten_grads(model, self.cfg.Fingerprint.require_grad_only).detach()

        delta = -self.cfg.Fingerprint.probe_lr * grad_vec  # scale by probe_lr
        del model
        return delta.cpu()

    def _fit_pca(self, deltas):
        """Fit PCA on stacked deltas: return (B, e_matrix)."""
        M, d_theta = deltas.shape # M = number of parameters
        device = deltas.device

        # d_e = min(self.cfg.Fingerprint.compressed_dim, d_theta, M)
        d_e = self.cfg.Fingerprint.compressed_dim

        if PCA is not None and M > 10:
            pca = PCA(n_components = d_e,
                      whiten=self.cfg.Fingerprint.pca_whiten,
                      svd_solver = 'full' if self.cfg.Fingerprint.pca_svd_full else 'randomized',
                      random_state=self.cfg.Fingerprint.random_state)

            comps = pca.fit_transform(deltas.cpu().numpy())

            if self.cfg.Fingerprint.l2_normalize:
                comps = comps / (np.linalg.norm(comps, axis=1, keepdims=True) + 1e-8)

            B = torch.from_numpy(pca.components_).to(device, dtype = deltas.dtype)  # [d_e, d_theta]
            e = torch.from_numpy(comps).to(device, dtype = deltas.dtype)
            return B,e
        else: # Use torch SVD
            mean = deltas.mean(dim=0, keepdim=True) # [1, d_theta]
            Xc  =deltas - mean  # (3, d_theta)
            # U: (3,3)
            # S: (3,)
            # Vh: (3, d_theta)
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  
            B = Vh[:d_e]
            print(B.shape)
            e = (U[:, :d_e] * S[:d_e])
            print(B.shape)
            print(e.shape)
            if self.cfg.Fingerprint.pca_whiten:
                e = e / (S[:d_e] + 1e-8)
            if self.cfg.Fingerprint.l2_normalize:
                e = F.normalize(e, p=2, dim=-1)
            return B, e


    def compute_fingerprints(self, data):
        """Caches results on disk; compute gradient probes per domain; fits PCA to obtain low-dim domain embeddings"""
        if self._cached:
            return self._e, self._B
        
        device = self._device()
        data = data.to(device)
        M = int(data.batch.max().item()) + 1 # number of domains (pre-training graphs)

        H_all, _  = self._embed_nodes(data) # [N, d] 
        # print(H_all.shape)

        deltas = []
        for i in range(M):  # each domain
            delta_i = self._probe_grad4domain(data, H_all, i)
            deltas.append(delta_i)
        deltas = torch.stack(deltas, dim=0) # [3, 11829]
        B, e = self._fit_pca(deltas)
        ds_names_str = '_'.join(data.name_dict.keys()) # Cora_Citeseer_Pubmed_...
        path_B = self._maybe_cache_path(ds_names_str, "B.pt")
        path_e = self._maybe_cache_path(ds_names_str, "PreDomainEmbedding.pt")
        self._B, self._e= B, e
        self._save_cache()
        return e, B
    
    def get_cached(self):
        if not self._cached:
            return None
        return self._e, self._B

    @torch.no_grad()
    def fingerprint_unseen(self, data_new: Data):
        if self._B is None or self._theta0 is None:
            raise RuntimeError('Compute domain embedding on pre‑training corpus first.')

        device = self._device()
        self.backbone.load_state_dict(self._theta0)
        self.backbone.to(device).train(False)
        datat_new = data_new.to(device)
        # single probe step on whole unseen graph
        edge_index_new = datat_new.edge_index
        x = datat_new.x
        H, _ = self.backbone(datat_new)
        y = datat_new.y
        if self.cfg.Fingerprint.loss_type == 'ce':
            loss = self.prob_loss(H, y)
        elif self.cfg.Fingerprint.loss_type == 'contrastive':
            loss = self.prob_loss(data_new, H)
        self.backbone.zero_grad(set_to_none=True)
        loss.backward()
        grad = flatten_grads(self.backbone, self.cfg.Fingerprint.require_grad_only).detach()
        delta = -self.cfg.Fingerprint.probe_lr * grad.cpu()
        e_new = (self._B @ delta).to(device)
        if self.cfg.Fingerprint.l2_normalize:
            e_new = F.normalize(e_new, p=2, dim=-1)
        return e_new

class DomainEmbedder(nn.Module):
    def __init__(self, backboneGNN, cfg):  # backboneGNN is a frozen GNN model
        super().__init__()
        self.backboneGNN = backboneGNN
        self.cfg = cfg
        self.dm_extractor = DomainEmbeddingExtractor(backboneGNN, cfg)
        self.dm_film = DomainFiLM(cfg)

    def forward(self, data, device):
        
        e, B = self.dm_extractor.compute_fingerprints(data)
        e = e.to(device)
        B = B.to(device)
        gamma_f, beta_f, gamma_l, beta_l = self.dm_film(e)
        return e, (gamma_f, beta_f, gamma_l, beta_l), B
    
    @torch.no_grad()
    def fingerprint_unseen(self, data_new: Data):
        return self.dm_extractor.fingerprint_unseen(data_new)  # e_new



