import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from omegaconf import DictConfig, OmegaConf
from utils.logging import logger, timer
from data_process.data import SingleGraphDataset
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs, SentenceEncoder
from model.base import BackboneGNN
from model.fingerprint import DomainEmbedder
from model.pt_model import GFM, PAMA
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
import os.path as osp
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.decomposition import PCA


class InContextLearner:
    """In-context learning interface for G-Align pretrained models."""
    
    def __init__(self,args):
        """
        Initialize the in-context learner.
        
        Args:
            model_path: Path to the saved pretrained model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
        self.model_path = args.model_path
        self.args = args
        # Load pretrained model state
        logger.info(f"Loading pretrained model from {self.model_path}")
        self.model_state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract configuration and metadata
        self.cfg = self.model_state['config']
        self.L_max = self.model_state['L_max']
        self.pretrain_datasets = self.model_state['pretrain_datasets']
        self.pretrain_tasks = self.model_state.get('pretrain_tasks', [])
        
        # Manually resolve hydra interpolations that can't be resolved in non-hydra context
        self._resolve_config_paths()
        
        # Initialize components
        self._setup_model()
        
        logger.info("In-context learner initialized successfully")
    
    def _resolve_config_paths(self):
        """Manually resolve hydra interpolations in the configuration."""
        import os
        from omegaconf import OmegaConf
        
        # Get the current working directory as project_root
        project_root = os.getcwd()
        
        # Convert to structured config to avoid hydra interpolation errors
        try:
            # Create a new config with resolved paths
            config_dict = OmegaConf.to_container(self.cfg, resolve=False)
            
            # Manually resolve the known interpolations
            if 'dirs' in config_dict and config_dict['dirs']:
                dirs = config_dict['dirs']
                
                # Replace fingerprint_storage path
                if 'fingerprint_storage' in dirs:
                    dirs['fingerprint_storage'] = os.path.join(project_root, 'generated_files', 'fingerprint')
                
                # Replace data_storage path
                if 'data_storage' in dirs:
                    dirs['data_storage'] = os.path.join(project_root, 'datasets')
                
                # Replace output path
                if 'output' in dirs:
                    model_name = 'G-Align'
                    if 'model' in config_dict and 'name' in config_dict['model']:
                        model_name = config_dict['model']['name']
                    dirs['output'] = os.path.join(project_root, 'generated_files', 'output', model_name)
            
            # Convert back to OmegaConf
            self.cfg = OmegaConf.create(config_dict)
            
        except Exception as e:
            logger.warning(f"Failed to resolve config paths: {e}. Using fallback paths.")
            # Fallback: create basic paths structure
            from omegaconf import DictConfig
            if not hasattr(self.cfg, 'dirs'):
                self.cfg.dirs = DictConfig({})
            
            # Set fallback paths
            self.cfg.dirs.fingerprint_storage = os.path.join(project_root, 'generated_files', 'fingerprint')
            self.cfg.dirs.data_storage = os.path.join(project_root, 'datasets')
            self.cfg.dirs.output = os.path.join(project_root, 'generated_files', 'output', 'G-Align')


    def _setup_model(self):
        """Setup the model components from saved state."""
        # Initialize backbone GNN (the trained one)
        input_dim = self.model_state['combined_graphs_info']['num_features']
        self.backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=self.L_max, cfg=self.cfg)
        self.backbone_gnn.n_layers = 2
        self.backbone_gnn.load_state_dict(self.model_state['backbone_state_dict'])
        self.backbone_gnn.to(self.device).eval()
        
        # Initialize frozen backbone (θ₀) for domain fingerprinting
        self.frozen_backbone = BackboneGNN(in_dim=input_dim, num_classes=self.L_max, cfg=self.cfg)
        self.frozen_backbone.n_layers = 1
        self.frozen_backbone.readout_proj = False
        if 'frozen_backbone_state_dict' in self.model_state:
            self.frozen_backbone.load_state_dict(self.model_state['frozen_backbone_state_dict'], strict=False)
        self.frozen_backbone.to(self.device).eval()
        
        # Initialize domain embedder
        self.domain_embedder = DomainEmbedder(self.frozen_backbone, self.cfg)
        
        # Load the convolutional projection network state
        if 'domain_embedder_projection_state' in self.model_state:
            self.domain_embedder.dm_extractor.projection.load_state_dict(
                self.model_state['domain_embedder_projection_state']
            )
            logger.info("Loaded convolutional projection network")
        
        # Load cached components
        self.domain_embedder.dm_extractor._theta0 = self.model_state.get('domain_embedder_theta0')
        self.domain_embedder.dm_extractor._e = self.model_state.get('domain_embedder_e')
        
        # Load delta matrices and padding info if saved
        if 'domain_embedder_delta_matrices' in self.model_state:
            self.domain_embedder.dm_extractor._delta_matrices = self.model_state['domain_embedder_delta_matrices']
            # Also need d_c_max for padding new graphs
            if isinstance(self.domain_embedder.dm_extractor._delta_matrices, list):
                # Compute d_c_max from saved matrices
                d_c_max = max(dm.shape[1] for dm in self.domain_embedder.dm_extractor._delta_matrices)
                self.domain_embedder.dm_extractor._d_c_max = d_c_max
                logger.info(f"Maximum number of classes in pretraining: {d_c_max}")
        
        self.domain_embedder.dm_extractor._cached = True
        self.domain_embedder.to(self.device).eval()
        
        # Create a dummy combined graph for GFM initialization
        dummy_graphs = Data(
            x=torch.randn(1, input_dim),
            edge_index=torch.tensor([[0], [0]]),
            y=torch.tensor([0]),
            batch=torch.tensor([0]),
            ptr=torch.tensor([0, 1])
        )
        
        # Initialize GFM model
        self.model = GFM(
            cfg=self.cfg,
            L_max=self.L_max,
            comb_pretrained_graphs=dummy_graphs,
            backboneGNN=self.backbone_gnn,
            domain_embedder=self.domain_embedder
        )
        self.model.load_state_dict(self.model_state['model_state_dict'])
        self.model.to(self.device).eval()
        
        # Get PAMA and label embeddings
        self.pama = self.model.pama
        self.E_label = self.model.E_lab
        
        logger.info("Model components initialized successfully")
    
    def load_downstream_graph(self, 
                             dataset_name: str,
                             data_path: str = None) -> Data:
        """
        Load a downstream graph dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cora', 'citeseer')
            data_path: Optional custom data path
            
        Returns:
            Processed graph data
        """
        logger.info(f"Loading downstream dataset: {dataset_name}")
        
        # Use data path from config if not provided
        if data_path is None:
            data_path = self.cfg.dirs.data_storage
        
        # Create encoder (will not actually encode since cora doesn't need it)
        llm_encoder = SentenceEncoder(self.cfg.llm_name, batch_size=self.cfg.llm_b_size)
        
        # Load the dataset
        dataset = SingleGraphDataset(
            self.cfg, 
            name=dataset_name,
            llm_encoder=llm_encoder,
            load_text=False  # Cora doesn't need text encoding
        )
        
        # Process dataset
        dataset = refine_dataset(dataset)
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
        
        # for cora: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], xn=[2708, 1433], xe=[10556])
        pca_x = self.dimension_align(dataset)
        xe    = dataset.data.xe.unsqueeze(1)
        graph_data = Data(x = pca_x,
                          edge_index = dataset.edge_index,
                          y = dataset.labels,
                          xe = xe,
                          train_mask = dataset.train_mask,
                          test_mask = dataset.test_mask,
                          val_mask = dataset.val_mask)


        # Ensure proper attributes
        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
        
        # Handle edge attributes - if edge_attr doesn't match node features dimension,
        # create dummy edge features or adjust them
        num_edges = graph_data.edge_index.shape[1]
        node_feature_dim = graph_data.x.shape[1]
        
        logger.info(f"Graph info - Edges: {num_edges}, Node features: {node_feature_dim}")
        
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            logger.info(f"Original edge_attr shape: {graph_data.edge_attr.shape}")
            # Check if edge_attr has wrong dimensions
            if graph_data.edge_attr.shape != (num_edges, node_feature_dim):
                logger.info(f"Edge attributes dimension mismatch. Expected: ({num_edges}, {node_feature_dim}), Got: {graph_data.edge_attr.shape}")
                # Create zero edge features with correct dimensions
                graph_data.edge_attr = torch.zeros(num_edges, node_feature_dim, dtype=graph_data.x.dtype)
                logger.info(f"Created dummy edge attributes with shape: {graph_data.edge_attr.shape}")
        else:
            # If no edge attributes, create zero edge features  
            graph_data.edge_attr = torch.zeros(num_edges, node_feature_dim, dtype=graph_data.x.dtype)
            logger.info(f"No edge attributes found, created dummy edge attributes with shape: {graph_data.edge_attr.shape}")
        
        logger.info(f"Final edge_attr shape: {graph_data.edge_attr.shape}, dtype: {graph_data.edge_attr.dtype}")
        
        return graph_data.to(self.device)

    def dimension_align(self, ds):
        unify_dim = self.cfg.unify_dim if self.cfg.unify_dim else 50
        pca_cache_path = osp.join(ds.processed_dir, f"pca_{unify_dim}.pt")
        if osp.exists(pca_cache_path):
            pca_x = torch.load(pca_cache_path, weights_only=False)
        else:
            if ds.num_features == unify_dim:
                pca_x = ds.data.xn.clone()
            else:
                x_np = ds.data.xn.cpu().numpy()
                pca = PCA(n_components=unify_dim)
                projected = pca.fit_transform(x_np)
                pca_x  = torch.from_numpy(projected).float()
                torch.save(pca_x, pca_cache_path)
        return pca_x



    @torch.no_grad()
    def compute_domain_embedding(self, graph_data: Data) -> torch.Tensor:
        """
        Compute domain embedding for a new graph using in-context learning.
        
        Args:
            graph_data: The new graph data
            
        Returns:
            Domain embedding tensor of shape (d_e,)
        """
        logger.info("Computing domain embedding for new graph...")
        
        # Use the domain embedder to compute fingerprint for unseen graph
        domain_embedding = self.domain_embedder.fingerprint_unseen(graph_data)
        
        logger.info(f"Domain embedding shape: {domain_embedding.shape}")
        return domain_embedding
    
    @torch.no_grad()
    def few_shot_inference(self, 
                          graph_data: Data, 
                          k_shot: int = 5,
                          seed: int = 42) -> Dict:
        """
        Perform few-shot in-context learning on a new graph.
        
        Args:
            graph_data: The target graph data
            k_shot: Number of examples per class for prompting
            seed: Random seed for sampling support set
            
        Returns:
            Dictionary containing predictions and metrics
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Performing {k_shot}-shot in-context learning...")
        
        # Step 1: Compute domain embedding for the new graph
        domain_embedding = self.compute_domain_embedding(graph_data)
        
        # Step 2: Compute domain-specific FiLM parameters
        gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
        gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
        gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
        
        logger.info("Computed FiLM parameters for domain adaptation")
        
        # Step 3: Get node embeddings from the TRAINED backbone GNN
        H, _ = self.backbone_gnn.encode(
            graph_data.x, 
            graph_data.edge_index, 
            graph_data.xe if hasattr(graph_data, 'xe') else None,
            graph_data.batch if hasattr(graph_data, 'batch') else None
        )
        
        # Step 4: Domain-align all node features
        z_all = gamma_f * H + beta_f
        
        # Step 5: Sample support and query sets
        labels = graph_data.y
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels.to(self.device)
        num_classes = len(unique_labels)
        
        logger.info(f"Number of classes: {num_classes}")
        
        # Use train/test masks if available
        if hasattr(graph_data, 'train_mask') and hasattr(graph_data, 'test_mask'):
            train_mask = graph_data.train_mask
            test_mask = graph_data.test_mask
        else:
            # Create random split
            n_nodes = labels.shape[0]
            perm = torch.randperm(n_nodes)
            split_idx = int(0.8 * n_nodes)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            train_mask[perm[:split_idx]] = True
            test_mask[perm[split_idx:]] = True
        
        # Sample k-shot support set from train nodes
        support_indices = []
        support_labels = []
        
        for class_label in unique_labels:
            # Get train nodes for this class
            class_train_mask = train_mask & (labels == class_label)
            class_train_indices = class_train_mask.nonzero(as_tuple=False).squeeze()
            
            if len(class_train_indices) < k_shot:
                logger.warning(f"Class {class_label} has only {len(class_train_indices)} train samples, using all")
                selected = class_train_indices
            else:
                # Randomly sample k nodes
                perm = torch.randperm(len(class_train_indices))[:k_shot]
                selected = class_train_indices[perm]
            
            support_indices.append(selected)
            support_labels.extend([class_label.item()] * len(selected))
        
        support_indices = torch.cat(support_indices)
        support_labels = torch.tensor(support_labels, device=self.device)
        
        logger.info(f"Support set: {len(support_indices)} nodes from {num_classes} classes")
        
        # Step 6: Get domain-aligned label embeddings
        U_base = self.E_label[unique_labels]  # (num_classes, d)
        U_aligned = gamma_l * U_base + beta_l  # Domain-aligned label embeddings
        
        # Step 7: Build support set features
        Z_sup = z_all[support_indices]  # (k*num_classes, d)
        
        # Step 8: Perform inference on all test nodes
        test_indices = test_mask.nonzero(as_tuple=False).squeeze()
        predictions = []
        confidences = []
        true_labels = []
        
        logger.info(f"Evaluating on {len(test_indices)} test nodes...")
        
        for idx in tqdm(test_indices, desc="Inference"):
            z_q = z_all[idx]  # Query node feature
            
            # Use PAMA for attention-based prediction
            logits = self.pama(z_q, Z_sup, U_aligned)  # (num_classes,)
            
            # Get prediction
            probs = F.softmax(logits, dim=-1)
            pred_idx = logits.argmax().item()
            predicted_label = unique_labels[pred_idx].item()
            confidence = probs[pred_idx].item()
            
            predictions.append(predicted_label)
            confidences.append(confidence)
            true_labels.append(labels[idx].item())
        
        # Step 9: Compute metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        confidences = np.array(confidences)
        
        accuracy = (predictions == true_labels).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for class_label in unique_labels:
            class_mask = true_labels == class_label.item()
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == true_labels[class_mask]).mean()
                per_class_acc[class_label.item()] = class_acc
        
        results = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'mean_confidence': confidences.mean(),
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'num_test_samples': len(test_indices),
            'num_classes': num_classes,
            'k_shot': k_shot
        }
        
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info(f"Mean Confidence: {confidences.mean():.4f}")
        
        return results
    
    def evaluate_multiple_runs(self,
                              graph_data: Data,
                              k_shot: int = 5,
                              n_runs: int = 10) -> Dict:
        """
        Evaluate with multiple random support sets.
        
        Args:
            graph_data: Target graph
            k_shot: Number of support examples per class
            n_runs: Number of random runs
            
        Returns:
            Aggregated results
        """
        logger.info(f"Running {n_runs} evaluations with different support sets...")
        
        all_accuracies = []
        all_confidences = []
        
        for run in range(n_runs):
            results = self.few_shot_inference(graph_data, k_shot=k_shot, seed=42+run)
            all_accuracies.append(results['accuracy'])
            all_confidences.append(results['mean_confidence'])
        
        return {
            'mean_accuracy': np.mean(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'mean_confidence': np.mean(all_confidences),
            'std_confidence': np.std(all_confidences),
            'all_accuracies': all_accuracies,
            'k_shot': k_shot,
            'n_runs': n_runs
        }


def main():
    parser = argparse.ArgumentParser(description="G-Align In-Context Learning")
    parser.add_argument('--model_path', type=str, default='generated_files/output/G-Align/final_gfm_model.pt',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str, default='cora',
                       help='Downstream dataset name')
    parser.add_argument('--k_shot', type=int, default=3,
                       help = 'Number of support examples per class')
    parser.add_argument('--n_runs', type=int, default=10,
                       help = 'Number of evaluation runs')
    parser.add_argument('--gpu_id', type=int, default=2,
                       help = 'GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help = 'Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize the in-context learner
    logger.info("="*60)
    logger.info("G-Align In-Context Learning")
    logger.info("="*60)
    
    learner = InContextLearner(args)
    
    # Load downstream dataset
    graph_data = learner.load_downstream_graph(args.dataset)
    logger.info(f"Loaded {args.dataset} dataset:")
    logger.info(f"  Nodes: {graph_data.x.shape[0]}")
    logger.info(f"  Edges: {graph_data.edge_index.shape[1]}")
    logger.info(f"  Features: {graph_data.x.shape[1]}")
    logger.info(f"  Classes: {int(graph_data.y.max().item()) + 1}")
    
    # Single run evaluation
    logger.info("\n" + "="*60)
    logger.info(f"Single Run Evaluation ({args.k_shot}-shot)")
    logger.info("="*60)
    
    single_results = learner.few_shot_inference(
        graph_data, 
        k_shot=args.k_shot,
        seed=args.seed
    )
    
    print(f"\nSingle Run Results:")
    print(f"  Accuracy: {single_results['accuracy']:.4f}")
    print(f"  Confidence: {single_results['mean_confidence']:.4f}")
    print(f"  Test Samples: {single_results['num_test_samples']}")
    
    print(f"\nPer-Class Accuracy:")
    for class_id, acc in single_results['per_class_accuracy'].items():
        print(f"  Class {class_id}: {acc:.4f}")
    
    # Multiple runs evaluation
    if args.n_runs > 1:
        logger.info("\n" + "="*60)
        logger.info(f"Multiple Runs Evaluation ({args.n_runs} runs)")
        logger.info("="*60)
        
        multi_results = learner.evaluate_multiple_runs(
            graph_data,
            k_shot=args.k_shot,
            n_runs=args.n_runs
        )
        
        print(f"\nMultiple Runs Results ({args.n_runs} runs):")
        print(f"  Mean Accuracy: {multi_results['mean_accuracy']:.4f} ± {multi_results['std_accuracy']:.4f}")
        print(f"  Mean Confidence: {multi_results['mean_confidence']:.4f} ± {multi_results['std_confidence']:.4f}")
        
        # Show distribution
        print(f"\n  Accuracy Distribution:")
        for i, acc in enumerate(multi_results['all_accuracies']):
            print(f"    Run {i+1}: {acc:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()