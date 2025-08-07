# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

# # label embedding as function of graphs   graph conditional FiLM
# class DomainFiLM(nn.Module): 
#     def __init__(self, cfg):
#         super(DomainFiLM, self).__init__()
#         label_emb_dim = cfg.FiLM.label_emb_dim if hasattr(cfg.FiLM, 'label_emb_dim') else 64
#         hidden_dim = cfg.FiLM.hidden_dim
#         FiLM_dim = cfg.FiLM.FiLM_dim 
#         self.lin = nn.Sequential(nn.Linear(hidden_dim, FiLM_dim), nn.ReLU(),nn.Linear(FiLM_dim, label_emb_dim * 2))

#     def forward(self, h, batch):
#         """Return gamma and beta for each graph in the batch.
#            h: node embeddings, shape (N, h)
#            batch: graph id for each node, shape (N,)
#            Returns:
#                gamma: shape (B, h), where B is the number of graphs in the batch
#                beta: shape (B, h)
#         """
#         g_emb = global_mean_pool(h, batch)
#         gamma, beta = self.lin(g_emb).chunk(2, dim=-1)  # split into gamma and beta: each (G, h)
#         return gamma[batch], beta[batch]


# def make_conv(in_dim, out_dim, conv_type, heads = 1):
#     if conv_type == "GCN":
#         return GCNConv(in_dim, out_dim, add_self_loops=True)
#     elif conv_type == "GAT":
#         return GATConv(in_dim, out_dim//heads, heads=heads, concat = True)
#     elif conv_type == "SAGE":
#         return SAGEConv(in_dim, out_dim)
#     else:
#         raise ValueError(f"conv_type {conv_type} not recognized")

# class SharedEncoder(nn.Module):
#     """for all pretrained datasets, share the same encoder. Choice two layers GCN or GAT"""
#     def __init__(self, in_dim, cfg):
#         super(SharedEncoder, self).__init__()
#         self.n_layers = cfg.FeatAlign.n_layers
#         self.hidden_dim = cfg.FeatAlign.hidden_dim
#         self.num_proj_hidden = cfg.FeatAlign.num_proj_hidden
#         self.gnns = nn.ModuleList()
#         dims = [in_dim] + [self.hidden_dim] * self.n_layers
#         for l in range(self.n_layers):
#             self.convs.append(make_conv(dims[l], dims[l+1], conv_type=cfg.FeatAlign.conv_type, heads=cfg.FeatAlign.heads))

#         self.act = nn.ReLU()

#         self.fc1 = torch.nn.Linear(self.hidden_dim, self.num_proj_hidden)
#         self.fc2 = torch.nn.Linear(self.num_proj_hidden, self.hidden_dim)
#         self.proj_act = cfg.FeatAlign.proj_act if hasattr(cfg.FeatAlign, 'proj_act') else 'elu'
    
#     def projection(self, z: torch.Tensor):
#         z = getattr(F, self.proj_act)(self.fc1(z))
#         return self.fc2(z)

#     def forward(self, data):
#         x = data.x
#         edge_index = data.edge_index
#         batch = data.batch if hasattr(data, 'batch') else None
#         for conv in self.convs:
#             h = F.dropout(x, p=self.dropout, training=self.training)
#             h = conv(h, edge_index)
#             h = self.act(h)
#         return h

# class DomainSpecEncoder(nn.Module):
#     '''Graph-Conditioned Domain-Specific Encoder'''
#     def __init__(self, in_dim, cfg):
#         super(DomainSpecEncoder, self).__init__()
#         self.hidden_dim = cfg.FeatAlign.hidden_dim
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, self.hidden_dim), nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim))

#     def forward(self, x):
#         return self.net(x)               


# class DomainAligner(nn.Module):
#     def __init__(self, in_dim, num_labels, cfg):
#         super(DomainAligner, self).__init__()
#         self.shared_encoder = SharedEncoder(in_dim, cfg)
#         self.domain_film = DomainFiLM(cfg)
#         self.domain_spec_encoder = DomainSpecEncoder(in_dim, cfg)
#         self.dropout = cfg.FeatAlign.dropout if hasattr(cfg.FeatAlign, 'dropout') else 0.5
#         self.label_embedding_init = nn.Embedding(num_labels, cfg.FiLM.label_emb_dim)
#         self.proj = cfg.FeatAlign.proj
#         self.tau = cfg.FeatAlign.tau


#     def contrastive_loss(self, h_sh, h_dom, batch):
#         h_sh = F.normalize(h_sh, dim=-1)
#         h_dom = F.normalize(h_dom, dim=-1)
#         logits = torch.mm(h_sh, h_dom.t(), out=h_sh)/ self.tau
#         targets = torch.arange(h_sh.size(0), device=h_sh.device)  
#         return F.cross_entropy(logits, targets)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         if not self.proj:
#             h_sh  = self.shared_encoder(x, edge_index, batch)
#         else:
#             h_sh = self.shared_encoder(data)
#             h_sh = self.shared_encoder.projection(h_sh)

#         h_dom = self.domain_spec_encoder(data.x)









# class HyperFiLM(nn.Module):
#     def __init__(self, fp_dim, emb_dim, hidden_dim, dropout):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(fp_dim, hidden_dim), 
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 4 * emb_dim)
#         )
#         self.gate = nn.Parameter(torch.zeros(1))

#     def forward(self, fp):
#         gamma_f, beta_f, gamma_l, beta_l = self.net(fp).chunk(4, dim=-1)
#         return F.softplus(gamma_f), beta_f, F.softplus(gamma_l), beta_l

# class FingerprintEnc(nn.Module):
#     def __init__(self, in_dim, cfg, backbone: nn.Module, emb_dim: int):
#         super(FingerprintEnc, self).__init__()
#         self.probe_lr   = cfg.Fingerprint.probe_lr
#         theta = sum(p.numel() for p in backbone.parameters())
#         self.fp_proj_dim = cfg.Fingerprint.fp_proj_dim
#         # Fingerprint projection matrix B
#         self.register_buffer("P", torch.empty(self.fp_proj_dim, 0), persistent=False)
#         self.hyper   = HyperFiLM(self.fp_proj_dim, emb_dim, hidden_dim= cfg.Fingerprint.hidden_dim, dropout=cfg.Fingerprint.dropout)
#         self.proj    = nn.Linear(emb_dim, emb_dim)
#         self._film_cache = {}

#     @torch.no_grad()
#     def _fingerprint(self, data, graph_id):
#         clone = self._detach_model(self.backbone)
#         opt   = torch.optim.SGD(clone.parameters(), lr=self.probe_lr)

#         mask      = torch.rand(data.x.size(0), device=data.x.device) < 0.15
#         corrupted = data.x.clone();  corrupted[mask] = 0.
#         x_hat, _  = clone(corrupted, data.edge_index, data.batch)
#         loss      = F.mse_loss(x_hat[mask], data.x[mask])

#         opt.zero_grad(); loss.backward(); opt.step()

#         delta = torch.cat([
#             (p1 - p0).flatten()
#             for p1, p0 in zip(clone.parameters(), self.backbone.parameters())
#         ])
#         return delta

#     @staticmethod
#     def _detach_model(model):
#         from copy import deepcopy
#         m = deepcopy(model)
#         for p in m.parameters(): p.requires_grad = True
#         return m

#     # ------------------------------------------------------------------
#     # 3.2  ***NEW*** – build P by PCA over Δθ                         #
#     # ------------------------------------------------------------------
#     def build_p_matrix(self, graph_loader_dict: dict[str, torch.utils.data.DataLoader]):
#         """
#         graph_loader_dict  : mapping {graph_id: PyG DataLoader for that graph}
#         After this call:
#           • self.P is a (fp_proj_dim × |θ|) tensor with top‑d_e PCA vectors
#           • fingerprints & FiLM parameters are pre‑cached in _film_cache
#         """
#         device = next(self.backbone.parameters()).device
#         deltas = []
#         with torch.no_grad():
#             for g_id, loader in graph_loader_dict.items():
#                 batch = next(iter(loader)).to(device)
#                 delta = self._fingerprint(batch, g_id)
#                 deltas.append(delta.unsqueeze(0))

#         ΔΘ = torch.cat(deltas, dim=0)    # (M × |θ|)
#         ΔΘ = ΔΘ - ΔΘ.mean(0, keepdim=True)

#         # PCA via truncated SVD (torch.linalg.svd handles tall matrices well)
#         U, S, Vh = torch.linalg.svd(ΔΘ, full_matrices=False)
#         P = Vh[:self.fp_proj_dim]         # (d_e × |θ|)

#         # Register the PCA projection matrix
#         self.P = nn.Parameter(P, requires_grad=False)

#         # Cache fingerprints & FiLM vectors
#         for g_id, delta in zip(graph_loader_dict.keys(), deltas):
#             fp = delta @ self.P.T        # (d_e,)
#             self._film_cache[g_id] = self.hyper(fp)

#     # ------------------------------------------------------------------
#     # 3.3  Forward (unchanged except P check)
#     # ------------------------------------------------------------------
#     def forward(self, data, graph_id: str):
#         assert self.P.numel() > 0, "Run build_p_matrix(...) once before training!"
#         if graph_id not in self._film_cache:
#             # if training after PCA, new graphs won't appear; still safe
#             fp = (self._fingerprint(data, graph_id) @ self.P.T)
#             self._film_cache[graph_id] = self.hyper(fp)

#         γ, β, γ_lab, β_lab = self._film_cache[graph_id]
#         x, _ = self.backbone(data.x, data.edge_index, data.batch)
#         z    = γ * x + β
#         return z, (γ_lab, β_lab)






# # domain_embedding.py
# # -----------------------------------------------------------
# # Author: ChatGPT (o3)
# # Date  : 2025‑07‑17
# # -----------------------------------------------------------
# # Core dependencies
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GINConv, global_mean_pool
# from torch_geometric.data import Data

# ################################################################################
# # 1.  Backbone GNN (replaceable)                                                #
# ################################################################################
# class GINBackbone(nn.Module):
#     """5‑layer GIN ▸ h‐dim hidden, ReLU, BatchNorm."""
#     def __init__(self, in_dim: int, hidden: int = 256, num_layers: int = 5):
#         super().__init__()
#         mlps = []
#         dims = [in_dim] + [hidden] * num_layers
#         for d_in, d_out in zip(dims[:-1], dims[1:]):
#             mlps.append(
#                 GINConv(
#                     nn.Sequential(
#                         nn.Linear(d_in, d_out), nn.ReLU(),
#                         nn.Linear(d_out, d_out)
#                     ),
#                     train_eps=True
#                 )
#             )
#         self.layers = nn.ModuleList(mlps)
#         self.bns    = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in mlps])

#     def forward(self, x, edge_index, batch):
#         for conv, bn in zip(self.layers, self.bns):
#             x = F.relu(bn(conv(x, edge_index)))
#         g = global_mean_pool(x, batch)              # graph embedding g_i
#         return x, g                                 # node & graph reps

# ################################################################################
# # 2.  Hyper‑network to map fingerprint → FiLM                                   #
# ################################################################################
# class HyperFiLM(nn.Module):
#     def __init__(self, fp_dim, emb_dim, hidden=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(fp_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, 4 * emb_dim)         # γ,β | γ_lab,β_lab
#         )
#         self.emb_dim = emb_dim

#     def forward(self, fingerprint):
#         vec = self.net(fingerprint)                # (4h,)
#         γ, β, γ_lab, β_lab = vec.chunk(4, dim=-1)
#         γ       = F.softplus(γ)                    # positive scale
#         γ_lab   = F.softplus(γ_lab)
#         return γ, β, γ_lab, β_lab

# ################################################################################
# # 3.  Domain‑aware wrapper                                                      #
# ################################################################################
# class DomainAwareEncoder(nn.Module):
#     """
#     Wraps a backbone GNN with:
#       • fingerprint probe (1 gradient step)
#       • JL projection  P  to compress Δθ
#       • HyperFiLM      h_φ to produce per‑graph FiLM vectors
#     """
#     def __init__(
#         self,
#         backbone: nn.Module,
#         in_dim: int,
#         emb_dim: int,
#         fp_proj_dim: int = 256,
#         probe_lr: float  = 5e‑4,
#     ):
#         super().__init__()
#         self.backbone   = backbone
#         self.emb_dim    = emb_dim
#         self.probe_lr   = probe_lr

#         # Fingerprint projection matrix  P  (frozen, JL Gaussian)
#         dθ = sum(p.numel() for p in backbone.parameters())
#         self.register_buffer(
#             "P",
#             torch.randn(fp_proj_dim, dθ) / (fp_proj_dim ** .5)
#         )

#         # Hyper‑network
#         self.hyper = HyperFiLM(fp_proj_dim, emb_dim)

#         # Graph‑id ↦ FiLM cache (populated on first forward)
#         self._film_cache = {}  # graph_id -> (γ,β,γ_lab,β_lab)

#         # Feature→label projection  g
#         self.proj = nn.Linear(emb_dim, emb_dim)

#     ########################################################################
#     # 3.1  PRIVATE helpers                                                 #
#     ########################################################################
#     @torch.no_grad()
#     def _fingerprint(self, data: Data):
#         """One unsup probe step on *this* graph to compute Δθ."""
#         clone = DomainAwareEncoder._detach_model(self.backbone)
#         opt   = torch.optim.SGD(clone.parameters(), lr=self.probe_lr)

#         # unsup objective: masked‑feature reconstruction
#         mask = torch.rand(data.x.size(0)) < 0.15
#         corrupted = data.x.clone()
#         corrupted[mask] = 0.

#         x_, _ = clone(corrupted, data.edge_index, data.batch)
#         loss  = F.mse_loss(x_[mask], data.x[mask])
#         opt.zero_grad(); loss.backward(); opt.step()

#         Δθ = torch.cat([ (p1‑p0).flatten()
#                          for p1,p0 in zip(clone.parameters(),
#                                           self.backbone.parameters()) ])
#         return torch.matmul(self.P, Δθ)             # (fp_proj_dim,)

#     @staticmethod
#     def _detach_model(model: nn.Module):
#         from copy import deepcopy
#         clone = deepcopy(model)
#         for p in clone.parameters(): p.requires_grad = True
#         return clone

#     ########################################################################
#     # 3.2  PUBLIC forward                                                  #
#     ########################################################################
#     def forward(self, data: Data, graph_id: str):
#         """
#         • data.batch must exist (PyG convention)
#         • graph_id is a stable string key for the graph the batch belongs to
#         """
#         if graph_id not in self._film_cache:
#             fp              = self._fingerprint(data)
#             γ, β, γ_lab, β_lab = self.hyper(fp)
#             self._film_cache[graph_id] = (γ, β, γ_lab, β_lab)

#         γ, β, γ_lab, β_lab = self._film_cache[graph_id]
#         x, g = self.backbone(data.x, data.edge_index, data.batch)
#         z    = γ * x + β                          # aligned node feature
#         return z, (γ_lab, β_lab)                  # node reps, label‑FiLM

# ################################################################################
# # 4.  Prompt builder & ICL inference                                           #
# ################################################################################
# def encode_support(encoder, data_supp, graph_id, y_supp):
#     """Return support token matrix Z_sup and label matrix U_sup."""
#     z_s, (γ_lab, β_lab) = encoder(data_supp, graph_id)
#     u_lab = γ_lab * encoder.proj.weight.T + β_lab  # every row of E^lab projected
#     u_s   = u_lab[y_supp]                          # pick rows
#     return z_s[y_supp_mask := (data_supp.y >= 0)], u_s

# def encode_query(encoder, data_q, graph_id):
#     z_q, _ = encoder(data_q, graph_id)
#     return z_q

# def pama_attention(Q, K, V, d):
#     α = torch.softmax(Q @ K.T / d**0.5, dim=-1)
#     return α @ V

# ################################################################################
# # 5.  Example usage (citation, social, synthesis graphs)                       #
# ################################################################################
# if __name__ == "__main__":
#     # pseudo‑code – replace with real PyG datasets
#     from torch_geometric.datasets import Planetoid, Reddit, Amazon
#     datasets = {
#         "cit": Planetoid(root="data/Cora",        name="Cora"),
#         "soc": Reddit(root="data/Reddit"),
#         "misc": Amazon(root="data/Amazon", domain="photo")
#     }

#     # shared backbone + encoder
#     enc = DomainAwareEncoder(
#         backbone = GINBackbone(in_dim=datasets["cit"].num_features, hidden=256),
#         in_dim   = datasets["cit"].num_features,
#         emb_dim  = 256
#     ).cuda()

#     # pre‑training loop (episodic)
#     # … build DataLoader per graph, call enc.forward(…) & cross‑entropy as explained …

#     # downstream ICL on a new graph 'bio'
#     # 1) run fingerprint probe internally (enc will cache)
#     # 2) build k‑shot / m‑way prompt, call pama_attention to classify
# """
# # write file
# Path("domain_embedding.py").write_text(code)
# print("✅  domain_embedding.py created")

# ::contentReference[oaicite:0]{index=0}
#  ​:contentReference[oaicite:1]{index=1}​






















# # class DomainFiLM(nn.Module):
# #     """Graph-Conditioned FiLM"""
# #     def __init__(self, num_labels, num_domain, cfg):
# #         super(DomainFiLM, self).__init__()
# #         label_emb_dim = cfg.label_emb_dim if hasattr(cfg, 'label_emb_dim') else 64
# #         self.gamma = nn.Embedding(num_domain, label_emb_dim)
# #         self.beta = nn.Embedding(num_domain, label_emb_dim)
# #         # initial_label_mlp = torch.nn.Linear(bert_dim, self.emb_dim)
# #         self.learned_label_embedding = nn.Embedding(num_labels, label_emb_dim)
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         nn.init.ones_(self.gamma.weight)
# #         nn.init.zeros_(self.beta.weight)
# #         nn.init.xavier_uniform_(self.learned_label_embedding.weight)
    
# #     def forward(self, domain_id, label):
# #         """Return label embeddings shaped
# #            * (L, h) if `label is None` (matrix mode)
# #            * (B, h) if `label` is not None (lookup mode)"""   
# #         gamma_d = self.gamma(domain_id)
# #         beta_d = self.beta(domain_id)
# #         if label is None:
# #             e = self.learned_label_embedding.weight
# #             return gamma_d * e + beta_d
# #         else:
# #             e = self.learned_label_embedding(label)
# #             return gamma_d * e + beta_d