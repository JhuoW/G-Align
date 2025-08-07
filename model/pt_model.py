import torch
import torch.nn as nn
import pytorch_lightning as pl
from data_process.data import EpisodeDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model.fingerprint import DomainEmbedder
import torch.nn.functional as F
import copy

class DataModule(pl.LightningDataModule):
    def __init__(self, data, k:int, m:int, n:int, batch_size:int=None, t:int=1):
        super().__init__()
        self.train_ds = EpisodeDataset(data = data, k = k, m = m, n = n, t = t)
        self.val_ds = EpisodeDataset(data = data, k = k, m = m, n = n//5, t = t)  # Fewer episodes for validation
        self.test_ds = EpisodeDataset(data = data, k = k, m = m, n = n//5, t = t)
        self.batch_size = (n*int(data.ptr.numel()-1)) if batch_size is None else batch_size
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False, 
                         num_workers=0, pin_memory=True, collate_fn= lambda x: x)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                         num_workers=0, pin_memory=True, collate_fn= lambda x: x)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                         num_workers=0, pin_memory=True, collate_fn= lambda x: x)


class PAMA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.Fingerprint.hidden_dim
        self.d = cfg.PAMA.d_attn if hasattr(cfg.PAMA, 'd_attn') else cfg.Fingerprint.hidden_dim
        self.heads = cfg.PAMA.heads if hasattr(cfg.PAMA, 'heads') else 1
        self.W_Q = nn.Linear(self.h, self.d * self.heads)
        self.W_K = nn.Linear(self.h, self.d * self.heads)
        self.W_V = nn.Linear(self.h, self.d * self.heads)
        self.dropout = cfg.PAMA.dropout if hasattr(cfg.PAMA, 'dropout') else 0.5
        self.g = nn.Sequential(
            nn.Linear(self.d * self.heads, self.d),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d, self.h)
        )
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward_feature(self, z_q, Z_sup):
        batch_size = 1
        mk = Z_sup.shape[0]
        Q_feat = self.W_Q(z_q.unsqueeze(0)).view(batch_size, self.heads, -1)  # [1 x heads x d]
        K_feat = self.W_K(Z_sup).view(mk, self.heads, -1).transpose(0, 1)  # [heads x mk x d]
        V_feat = self.W_V(Z_sup).view(mk, self.heads, -1).transpose(0, 1)  # [heads x mk x d]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q_feat, K_feat.transpose(-2, -1)) / (self.d ** 0.5)  # [1 x heads x mk]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [1 x heads x mk]
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        z_out = torch.matmul(attn_weights, V_feat)  # [1 x heads x d]
        z_out = z_out.view(batch_size, -1).squeeze(0)  # [heads*d]
        return z_out
    
    # def forward_label(self, u_q, U_sup):
    #     batch_size = 1
    #     m = U_sup.shape[0]

    #     Q_label = self.W_Q(u_q.unsqueeze(0)).view(batch_size, self.heads, -1)  # [1 x heads x d]
    #     K_label = self.W_K(U_sup).view(m, self.heads, -1).transpose(0, 1)  # [heads x m x d]
    #     V_label = self.W_V(U_sup).view(m, self.heads, -1).transpose(0, 1)  # [heads x m x d]

    #     # Compute attention scores
    #     attn_scores = torch.matmul(Q_label, K_label.transpose(-2, -1)) / (self.d ** 0.5)  # [1 x heads x m]
    #     attn_weights = torch.softmax(attn_scores, dim=-1)  # [1 x heads x m]
        
    #     # Apply attention to values
    #     u_out = torch.matmul(attn_weights, V_label)  # [1 x heads x d]
    #     u_out = u_out.view(batch_size, -1).squeeze(0)  # [heads*d]
    #     return u_out
    
    def forward(self, z_q, Z_sup, U_sup):

        # Feature-side attention
        z_out = self.forward_feature(z_q, Z_sup)  # [d]
        
        # Map to label space
        z_hat = self.g(z_out)  # [d]
        
        # Compute logits against support class label embeddings
        V_label = self.W_V(U_sup)  # [m x d]
        V_label = V_label.view(U_sup.shape[0], self.heads, -1).mean(dim=1) 
        logits = torch.matmul(z_hat.unsqueeze(0), V_label.transpose(-1, -2)).squeeze(0)
        return logits



class GFM(pl.LightningModule):
    def __init__(self, cfg, L_max: int, comb_pretrained_graphs, backboneGNN, domain_embedder: DomainEmbedder):
        super().__init__()
        self.save_hyperparameters(ignore= ['comb_pretrained_graphs', 'backboneGNN', 'domain_embedder'])
        self.GNNEnc = backboneGNN
        
        self.frozen_backbone = copy.deepcopy(backboneGNN)
        for param in self.frozen_backbone.parameters():
            param.requires_grad = False

        self.cfg = cfg
        self.comb_pretrained_graphs = comb_pretrained_graphs
        self.de = domain_embedder
        self.de.dm_extractor.backbone = self.frozen_backbone

        self.E_lab = nn.Parameter(torch.randn(L_max, self.cfg.Fingerprint.hidden_dim))
        self.pama = PAMA(cfg)

        self.register_buffer('domain_embeddings', None)
        self.register_buffer('gamma_f', None)
        self.register_buffer('beta_f', None)
        self.register_buffer('gamma_l', None)
        self.register_buffer('beta_l', None)
        self.register_buffer('batch', None)
        self.register_buffer('ptr', None)


    def align(self, x, E, gamma_f, beta_f, gamma_l, beta_l, batch):
        z = x * gamma_f[batch] + beta_f[batch]
        u = (E*gamma_l.unsqueeze(1))+beta_l.unsqueeze(1)
        return z, u

    def on_train_epoch_start(self):
        comb_pretrained_graphs = self.comb_pretrained_graphs.to(self.device)

        with torch.no_grad():
            e, film, B = self.de(comb_pretrained_graphs)
            self.gamma_f, self.beta_f, self.gamma_l, self.beta_l = film
            self.domain_embeddings = e
            # self.H = self.backboneGNN.encode(comb_pretrained_graphs.x, 
            #                                  comb_pretrained_graphs.edge_index, 
            #                                  comb_pretrained_graphs.xe if hasattr(comb_pretrained_graphs, 'xe') else None)[0]
        self.register_buffer('batch', comb_pretrained_graphs.batch)
        self.register_buffer('ptr', comb_pretrained_graphs.ptr)

    def configure_optimizers(self):
        params = [
            {'params': self.GNNEnc.parameters(), 'lr': self.cfg.PTModel.lr},  # Backbone IS trained
            {'params': self.pama.parameters(), 'lr': self.cfg.PTModel.lr},
            {'params': self.E_lab, 'lr': self.cfg.PTModel.lr * 0.1},  # Lower LR for label embeddings
            {'params': self.de.dm_film.parameters(), 'lr': self.cfg.PTModel.lr * 0.5}  # FiLM network
        ]
        optimizer = optim.AdamW(params, weight_decay=self.cfg.PTModel.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs if hasattr(self, 'trainer') else 100,
            eta_min=1e-6
        )        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def forward_backbone(self, x, edge_index, edge_attr=None, batch=None):
        """Forward through the TRAINABLE backbone GNN"""
        # This uses the main backbone that IS being trained
        H, g = self.GNNEnc.encode(x, edge_index, edge_attr, batch)
        return H, g

    def training_step(self, batch, batch_idx):         
        losses = []
        accuracies = []

        comb_graphs = self.comb_pretrained_graphs.to(self.device)
        H, _ = self.forward_backbone(
            comb_graphs.x, 
            comb_graphs.edge_index,
            comb_graphs.xe if hasattr(comb_graphs, 'xe') else None,
            comb_graphs.batch
        )        
        for ep in batch:
            idx_sup = ep['support'].to(self.device)
            idx_qry = ep['query'].to(self.device)
            gid = ep['graph']
            classes = ep['classes'].to(self.device)

            gamma_f, beta_f = self.gamma_f[gid], self.beta_f[gid]

            z_sup = gamma_f * self.H[idx_sup] + beta_f
            z_qry = gamma_f * self.H[idx_qry] + beta_f

            gamma_l, beta_l = self.gamma_l[gid], self.beta_l[gid]
            U_sup_base = self.E_lab[classes]
            U_sup = gamma_l * U_sup_base + beta_l

            k = len(idx_sup) // len(classes)
            m = len(classes)

            episode_losses = []
            correct = 0
            for i , q_idx in enumerate(idx_qry):
                q_class = i % m
                z_q = z_qry[i]

                logits = self.pama(z_q, z_sup, U_sup)

                tgt = torch.tensor(q_class, device=self.device, dtype=torch.long)

                loss_q = F.cross_entropy(logits.unsqueeze(0), tgt.unsqueeze(0))
                episode_losses.append(loss_q)

                # Track accuracy
                pred = logits.argmax().item()
                if pred == q_class:
                    correct += 1

            episode_loss = torch.stack(episode_losses).mean()
            losses.append(episode_loss)
            accuracies.append(correct / len(idx_qry))
        
        total_loss = torch.stack(losses).mean()
        avg_acc = torch.tensor(accuracies).mean()

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', avg_acc, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses = []
        accuracies = []
        comb_graphs = self.comb_pretrained_graphs.to(self.device)
        with torch.no_grad():
            H, _ = self.forward_backbone(
                comb_graphs.x, 
                comb_graphs.edge_index,
                comb_graphs.xe if hasattr(comb_graphs, 'xe') else None,
                comb_graphs.batch
            )
        
        for ep in batch:
            idx_sup = ep['support'].to(self.device)
            idx_qry = ep['query'].to(self.device)
            gid = ep['graph']
            classes = ep['classes'].to(self.device)
    
            gamma_f, beta_f = self.gamma_f[gid], self.beta_f[gid]
            z_sup = gamma_f * self.H[idx_sup] + beta_f
            z_qry = gamma_f * self.H[idx_qry] + beta_f

            gamma_l, beta_l = self.gamma_l[gid], self.beta_l[gid]
            U_sup_base = self.E_lab[classes]
            U_sup = gamma_l * U_sup_base + beta_l

            k = len(idx_sup) // len(classes)
            m = len(classes)

            episode_losses = []
            correct = 0
            total = 0

            for i, q_idx in enumerate(idx_qry):
                q_class = i % m
                z_q = z_qry[i]
                
                logits = self.pama(z_q, z_sup, U_sup)
                pred = logits.argmax()
                
                target = torch.tensor(q_class, device=self.device, dtype=torch.long)
                loss_q = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                episode_losses.append(loss_q)
                
                if pred == q_class:
                    correct += 1
                total += 1
            episode_loss = torch.stack(episode_losses).mean()
            episode_acc = correct / total
            
            losses.append(episode_loss)
            accuracies.append(episode_acc)

        val_loss = torch.stack(losses).mean()
        val_acc = torch.tensor(accuracies).mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    

    def on_ssave_checkpoint(self, checkpoint):
        """Save both the trained backbone and the frozen initial backbone"""
        checkpoint['frozen_backbone_state'] = self.frozen_backbone.state_dict()
        checkpoint['domain_embeddings'] = self.domain_embeddings
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load both backbones when restoring from checkpoint"""
        if 'frozen_backbone_state' in checkpoint:
            self.frozen_backbone.load_state_dict(checkpoint['frozen_backbone_state'])
        if 'domain_embeddings' in checkpoint:
            self.domain_embeddings = checkpoint['domain_embeddings']

    # def predict_step(self, batch, batch_idx):
    #     predictions = []
        
    #     for ep in batch:
    #         idx_sup = ep['support'].to(self.device)
    #         idx_qry = ep['query'].to(self.device)
    #         gid = ep['graph']
    #         classes = ep['classes'].to(self.device)

    #         # Domain-aligned features
    #         gamma_f, beta_f = self.gamma_f[gid], self.beta_f[gid]
    #         z_sup = gamma_f * self.H[idx_sup] + beta_f
    #         z_qry = gamma_f * self.H[idx_qry] + beta_f

    #         # Domain-aligned label embeddings
    #         gamma_l, beta_l = self.gamma_l[gid], self.beta_l[gid]
    #         U_sup_base = self.E_label[classes]
    #         U_sup = gamma_l * U_sup_base + beta_l

    #         # Make predictions
    #         episode_preds = []
    #         for i, q_idx in enumerate(idx_qry):
    #             z_q = z_qry[i]
    #             logits = self.pama(z_q, z_sup, U_sup)
    #             pred = logits.argmax().item()
    #             episode_preds.append(pred)
            
    #         predictions.append({
    #             'graph': gid,
    #             'query_indices': idx_qry.cpu(),
    #             'predictions': episode_preds,
    #             'classes': classes.cpu()
    #         })
        
    #     return predictions