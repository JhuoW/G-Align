import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)  # /home/zhuowei/Code/G-Align
from omegaconf import DictConfig
from utils.logging import timer
import hydra
from utils.logging import logger
from utils.exp import init_exp
from utils.utils import seed_setting
from data_process.data import CombineDataset
import wandb
from omegaconf import OmegaConf
from data_process.task_constructor import train_task_constructor, UnifiedTaskConstructor
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs
from model.base import BackboneGNN
from model.fingerprint import DomainEmbedder
from model.pt_model import GFM, DataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os
import os.path as osp
import copy

def init_wandb(cfg):
    wandb.init(project=cfg.wandb.project,
               name = "Pretrain on tasks = {}".format(cfg.pretrain.pretrain_tasks),
               mode="disabled" if cfg.wandb.debug else "online",
               config=OmegaConf.to_object(cfg),)


def get_pretrain_data(cfg, pretrain_device):
    pretrain_ds_names = cfg.pretrain.pretrain_datasets
    if isinstance(pretrain_ds_names, str):
        pretrain_ds_names_lst = [a.strip() for a in pretrain_ds_names.split(",")]
    else:
        pretrain_ds_names_lst = pretrain_ds_names
    pretrain_tasks = cfg.pretrain.train_tasks  # ['pubmed_link','pubmed_node','arxiv','wikics']
    if isinstance(pretrain_tasks, str):
        pretrain_tasks = [a.strip() for a in pretrain_tasks.split(",")]
    pretrain_task_names = ','.join(pretrain_tasks)
    logger.info(f"Pretrain on the following tasks: {pretrain_task_names}")  # pubmed_link,pubmed_node,arxiv,wikics
    tasks: UnifiedTaskConstructor = train_task_constructor(data_path = cfg.dirs.data_storage, cfg = cfg, pretrain_tasks = pretrain_tasks)
    pretrain_dataset_dict = {}
    data_config_lookup = cfg.data_config
    # 获取预训练数据集
    for ds_name in pretrain_ds_names_lst: # ['pubmed', 'arxiv', 'wikics']
        if ds_name not in pretrain_dataset_dict:
            data_config = data_config_lookup[ds_name]
            dataset = tasks.get_dataset(cfg, data_config)
            dataset = refine_dataset(dataset)
            dataset = span_node_and_edge_idx(dataset)
            dataset = filter_unnecessary_attrs(dataset)
            pretrain_dataset_dict[ds_name] = dataset
    
    # Data(x=[200761, 64], edge_index=[2, 2835972], y=[200761], batch=[200761], ptr=[4],name_dict={pubmed=0,arxiv=1,wikics=2,})
    combined_pretrained_dataset = CombineDataset(cfg=cfg, pretrain_ds_dict = pretrain_dataset_dict, pretrain_device=pretrain_device).combine_graph()
    combined_pretrained_data = combined_pretrained_dataset[0]
    return combined_pretrained_data, pretrain_tasks


@timer()
@hydra.main(config_path=f"{root}/configs", config_name="main", version_base=None)
def main(cfg:DictConfig):
    cfg = init_exp(cfg) 
    seed_setting(cfg.pretrain.seed)
    if torch.cuda.is_available() and cfg.preprocess_device == "gpu":
        pretrain_device = torch.device(f"cuda:{cfg.pretrain.gpu}")
    else:
        pretrain_device = torch.device("cpu")

    # train on ['pubmed_link','pubmed_node','arxiv','wikics']
    # test on ['cora_node']
    comb_graphs, pretrain_tasks = get_pretrain_data(cfg, pretrain_device)
    cfg.pretrain.pretrain_tasks = pretrain_tasks  # [pubmed_link, pubmed_node, arxiv, wikics]
    # init_wandb(cfg)

    logger.info("Starting pretraining phase...")
    L_max = int(comb_graphs.y.max().item()) + 1
    logger.info(f"Maximum label count across datasets: {L_max}")

    input_dim = comb_graphs.x.size(-1)
    backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=L_max, cfg=cfg).to(pretrain_device)

    frozen_backbone = copy.deepcopy(backbone_gnn)
    frozen_backbone.n_layers = 1
    frozen_backbone.readout_proj = False
    # for param in frozen_backbone.parameters():
    #     param.requires_grad = False


    domain_embedder = DomainEmbedder(backboneGNN=frozen_backbone, cfg=cfg).to(pretrain_device)

    model = GFM(cfg = cfg,
                L_max= L_max,
                comb_pretrained_graphs= comb_graphs,
                backboneGNN=backbone_gnn,
                domain_embedder=domain_embedder)
    k_shot = cfg.pretrain.k_shot
    m_way = min(cfg.pretrain.m_way, L_max)
    t_query = cfg.pretrain.t_query
    n_episodes = cfg.pretrain.n_eqisodes
    data_module = DataModule(data=comb_graphs, k=k_shot, m=m_way, n=n_episodes, t=t_query)

    # Setup Callbacks
    checkpoint_dir = osp.join(cfg.dirs.checkpoint_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                          filename = "gfm-{epoch:02d}-{val_loss:.4f}",
                                          monitor="val_loss",
                                          mode="min",
                                          save_top_k=3,
                                          save_last=True,
                                          verbose=True)
    
    early_stopping_callback = EarlyStopping(monitor="val_loss",        
                                            min_delta=cfg.pretrain.min_delta,
                                            patience=cfg.pretrain.patience,
                                            verbose=True,
                                            mode="min")
    wandb_logger = WandbLogger(project=cfg.wandb.project,
                               name = "Pretrain on tasks = {}".format(cfg.pretrain.pretrain_tasks),
                               save_dir=cfg.dirs.temp,
                               mode="disabled" if cfg.wandb.debug else "online",
                               config=OmegaConf.to_object(cfg)) if not cfg.wandb.debug else None
    
    trainer = pl.Trainer(max_epochs=cfg.pretrain.pretrain_epochs,
                         accelerator="gpu" if torch.cuda.is_available() and cfg.pretrain.gpu is not None else "cpu",
                         devices=[cfg.pretrain.gpu] if torch.cuda.is_available() and cfg.pretrain.gpu is not None else "auto",
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         log_every_n_steps=cfg.pretrain.log_every_n_steps,
                         check_val_every_n_epoch=cfg.pretrain.check_val_every_n_epoch,
                         enable_progress_bar= True,
                         deterministic=True)
    
    logger.info("Starting training...")

    trainer.fit(model, datamodule=data_module)

    final_model_path = osp.join(cfg.dirs.output, "final_gfm_model.pt")

    # Ensure domain embedder has computed embeddings
    if not domain_embedder.dm_extractor._cached:
        logger.info("Computing domain embeddings...")
        with torch.no_grad():
            comb_graphs = comb_graphs.to(pretrain_device)
            e, B = domain_embedder.dm_extractor.compute_fingerprints(comb_graphs)

    model_state = {
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': backbone_gnn.state_dict(),  # Trained backbone
        'frozen_backbone_state_dict': frozen_backbone.state_dict(),  # Frozen initial backbone θ₀
        'domain_embedder_B': domain_embedder.dm_extractor._B,
        'domain_embedder_theta0': domain_embedder.dm_extractor._theta0,
        'domain_embedder_e': domain_embedder.dm_extractor._e,
        'config': cfg,
        'L_max': L_max,
        'pretrain_datasets': cfg.pretrain.pretrain_datasets,
        'pretrain_tasks': pretrain_tasks,
        'combined_graphs_info': {
            'num_nodes': comb_graphs.x.shape[0],
            'num_features': comb_graphs.x.shape[1],
            'num_graphs': int(comb_graphs.ptr.numel() - 1),
            'ptr': comb_graphs.ptr,
            'name_dict': comb_graphs.name_dict
        }        
    }
    torch.save(model_state, final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info("Running final validation...")
    test_results = trainer.test(model, datamodule=data_module)    

    logger.info("Pretraining completed successfully!")
    logger.info(f"Final test results: {test_results}")
    
    wandb.finish()
    return model, final_model_path

if __name__ == "__main__":
    main()











# before filter
# pubmed: Data(x=[19717], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717], pca_x=[19717, 64], num_classes=3, num_features=500, xe=[88648])
# arxiv: Data(num_nodes=169343, edge_index=[2, 2315598], x=[169343], node_year=[169343, 1], y=[169343, 1], train_mask=[169343], val_mask=[169343], test_mask=[169343], node_text_feat=[169343, 768], edge_text_feat=[1, 768], class_node_text_feat=[40, 768], xe=[2315598])
# wikics: Data(x=[11701], edge_index=[2, 431726], y=[11701], train_mask=[11701, 20], val_mask=[11701, 20], test_mask=[11701], stopping_mask=[11701, 20], node_text_feat=[11701, 768], edge_text_feat=[1, 768], class_node_text_feat=[10, 768], xe=[431726])

# after filter
# pubmed
# Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], xn=[19717, 500], xe=[88648])
# arxiv
# Data(x=[169343, 128], edge_index=[2, 2315598], y=[169343], xn=[169343, 768], xe=[2315598])
# wikics
# Data(x=[11701, 300], edge_index=[2, 431726], y=[11701], xn=[11701, 768], xe=[431726])