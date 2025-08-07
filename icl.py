"""
In-Context Learning Interface for G-Align
Performs inference on downstream graphs from unknown domains using the pretrained GFM model.
"""

import torch
import torch.nn.functional as F
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from omegaconf import DictConfig, OmegaConf
from utils.logging import logger
from data_process.data import SingleGraphDataset
from data_process.datahelper import refine_dataset, span_node_and_edge_idx, filter_unnecessary_attrs
from model.base import BackboneGNN
from model.fingerprint import DomainEmbedder
from model.pt_model import GFM, PAMA
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
import os.path as osp


class InContextLearner:
    """In-context learning interface for G-Align pretrained models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the in-context learner.
        
        Args:
            model_path: Path to the saved pretrained model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Load pretrained model state
        logger.info(f"Loading pretrained model from {model_path}")
        self.model_state = torch.load(model_path, map_location=self.device)
        
        # Extract configuration and metadata
        self.cfg = self.model_state['config']
        self.L_max = self.model_state['L_max']
        self.pretrain_datasets = self.model_state['pretrain_datasets']
        
        # Initialize components
        self._setup_model()
        
        logger.info("In-context learner initialized successfully")
    
    def _setup_model(self):
        """Setup the model components from saved state."""
        # Initialize backbone GNN
        input_dim = self.model_state['combined_graphs_info']['num_features']
        self.backbone_gnn = BackboneGNN(in_dim=input_dim, num_classes=self.L_max, cfg=self.cfg)
        self.backbone_gnn.load_state_dict(self.model_state['backbone_state_dict'])
        self.backbone_gnn.to(self.device).eval()
        
        # Initialize domain embedder with cached components
        self.domain_embedder = DomainEmbedder(self.backbone_gnn, self.cfg)
        self.domain_embedder.dm_extractor._B = self.model_state['domain_embedder_B']
        self.domain_embedder.dm_extractor._theta0 = self.model_state['domain_embedder_theta0']
        self.domain_embedder.dm_extractor._cached = True
        
        # Initialize GFM model
        # Create dummy combined graphs for initialization
        dummy_graphs = Data(
            x=torch.randn(1, input_dim),
            edge_index=torch.tensor([[0], [0]]),
            y=torch.tensor([0]),
            batch=torch.tensor([0]),
            ptr=torch.tensor([0, 1])
        )
        
        self.model = GFM(
            cfg=self.cfg,
            L_max=self.L_max,
            comb_pretrained_graphs=dummy_graphs,
            backboneGNN=self.backbone_gnn,
            domain_embedder=self.domain_embedder
        )
        self.model.load_state_dict(self.model_state['model_state_dict'])
        self.model.to(self.device).eval()
        
        # Initialize PAMA for inference
        self.pama = PAMA(self.cfg).to(self.device).eval()
        self.pama.load_state_dict({k.replace('pama.', ''): v for k, v in self.model_state['model_state_dict'].items() if k.startswith('pama.')})
    
    def load_downstream_graph(self, dataset_name: str, cfg_override: Optional[Dict] = None) -> Data:
        """
        Load a downstream graph dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cora', 'citeseer')
            cfg_override: Optional configuration overrides
            
        Returns:
            Processed graph data
        """
        # Create a temporary config for the downstream dataset
        temp_cfg = OmegaConf.create(self.cfg)
        if cfg_override:
            temp_cfg = OmegaConf.merge(temp_cfg, cfg_override)
        
        # Load the dataset
        dataset = SingleGraphDataset(temp_cfg, dataset_name)
        dataset = refine_dataset(dataset)
        dataset = span_node_and_edge_idx(dataset)
        dataset = filter_unnecessary_attrs(dataset)
        
        return dataset.data.to(self.device)
    
    def compute_domain_embedding(self, graph_data: Data) -> torch.Tensor:
        """
        Compute domain embedding for a new graph using in-context learning.
        
        Args:
            graph_data: The new graph data
            
        Returns:
            Domain embedding tensor
        """
        logger.info("Computing domain embedding for new graph...")
        
        # Use the domain embedder to compute fingerprint for unseen graph
        domain_embedding = self.domain_embedder.fingerprint_unseen(graph_data)
        
        return domain_embedding
    
    def few_shot_inference(self, 
                          graph_data: Data, 
                          support_indices: List[int], 
                          support_labels: List[int],
                          query_indices: List[int],
                          k_shot: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform few-shot inference on a new graph.
        
        Args:
            graph_data: The target graph data
            support_indices: Node indices for support set
            support_labels: Labels for support nodes
            query_indices: Node indices for queries
            k_shot: Number of shots per class
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        with torch.no_grad():
            # Compute domain embedding for the new graph
            domain_embedding = self.compute_domain_embedding(graph_data)
            
            # Compute domain-specific FiLM parameters
            gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
            gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
            gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
            
            # Get node embeddings from backbone GNN
            node_embeddings = self.backbone_gnn.encode(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.xe if hasattr(graph_data, 'xe') else None
            )[0]
            
            # Domain-align node features
            z_sup = gamma_f * node_embeddings[support_indices] + beta_f
            z_qry = gamma_f * node_embeddings[query_indices] + beta_f
            
            # Get unique classes and create label embeddings
            unique_labels = torch.tensor(list(set(support_labels)), device=self.device)
            U_sup_base = self.model.E_label[unique_labels]
            U_sup = gamma_l * U_sup_base + beta_l
            
            # Perform inference for each query
            predictions = []
            confidence_scores = []
            
            for i, q_idx in enumerate(query_indices):
                z_q = z_qry[i]
                
                # Compute logits using PAMA
                logits = self.pama(z_q, z_sup, U_sup)
                
                # Get prediction and confidence
                probs = F.softmax(logits, dim=-1)
                pred_class_idx = logits.argmax().item()
                confidence = probs[pred_class_idx].item()
                
                # Map back to original label space
                predicted_label = unique_labels[pred_class_idx].item()
                
                predictions.append(predicted_label)
                confidence_scores.append(confidence)
        
        return torch.tensor(predictions), torch.tensor(confidence_scores)
    
    def zero_shot_inference(self, 
                           graph_data: Data, 
                           query_indices: List[int],
                           class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform zero-shot inference using only class name information.
        
        Args:
            graph_data: The target graph data
            query_indices: Node indices for queries
            class_names: Names/descriptions of the classes
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        with torch.no_grad():
            # Compute domain embedding
            domain_embedding = self.compute_domain_embedding(graph_data)
            
            # Get domain-specific parameters
            gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
            gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
            gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
            
            # Get node embeddings
            node_embeddings = self.backbone_gnn.encode(
                graph_data.x, 
                graph_data.edge_index, 
                graph_data.xe if hasattr(graph_data, 'xe') else None
            )[0]
            
            # Domain-align query features
            z_qry = gamma_f * node_embeddings[query_indices] + beta_f
            
            # Create label embeddings for the given classes
            # For zero-shot, we use a subset of the pretrained label embeddings
            num_classes = len(class_names)
            U_sup_base = self.model.E_label[:num_classes]  # Use first N label embeddings
            U_sup = gamma_l * U_sup_base + beta_l
            
            # Dummy support set (not used in zero-shot but required by PAMA)
            z_sup = torch.zeros(num_classes, z_qry.shape[1], device=self.device)
            
            predictions = []
            confidence_scores = []
            
            for i, q_idx in enumerate(query_indices):
                z_q = z_qry[i]
                
                # Compute similarity scores directly
                z_q_mapped = self.pama.g(z_q)
                V_label = self.pama.W_V(U_sup)
                logits = torch.matmul(z_q_mapped, V_label.T)
                
                probs = F.softmax(logits, dim=-1)
                pred_class_idx = logits.argmax().item()
                confidence = probs[pred_class_idx].item()
                
                predictions.append(pred_class_idx)
                confidence_scores.append(confidence)
        
        return torch.tensor(predictions), torch.tensor(confidence_scores)
    
    def evaluate_on_dataset(self, 
                           dataset_name: str, 
                           k_shot: int = 5, 
                           n_episodes: int = 100,
                           split_ratio: float = 0.8) -> Dict[str, float]:
        """
        Evaluate the model on a downstream dataset using episodic evaluation.
        
        Args:
            dataset_name: Name of the dataset to evaluate on
            k_shot: Number of support examples per class
            n_episodes: Number of episodes to run
            split_ratio: Ratio of nodes to use for support vs query
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} dataset...")
        
        # Load the dataset
        graph_data = self.load_downstream_graph(dataset_name)
        
        accuracies = []
        confidences = []
        
        # Get unique classes
        unique_labels = graph_data.y.unique()
        n_classes = len(unique_labels)
        
        for episode in range(n_episodes):
            # Sample classes for this episode
            if n_classes > 5:  # Multi-way if enough classes
                sampled_classes = unique_labels[torch.randperm(n_classes)[:min(5, n_classes)]]
            else:
                sampled_classes = unique_labels
            
            support_indices = []
            support_labels = []
            query_indices = []
            query_labels = []
            
            # Sample support and query sets for each class
            for class_label in sampled_classes:
                class_nodes = (graph_data.y == class_label).nonzero(as_tuple=False).squeeze()
                
                if len(class_nodes.shape) == 0:  # Single node
                    class_nodes = class_nodes.unsqueeze(0)
                
                # Randomly split into support and query
                n_nodes = class_nodes.size(0)
                n_support = min(k_shot, max(1, int(n_nodes * split_ratio)))
                
                perm = torch.randperm(n_nodes)
                support_nodes = class_nodes[perm[:n_support]]
                query_nodes = class_nodes[perm[n_support:n_support+2]]  # Take 2 query nodes max
                
                support_indices.extend(support_nodes.tolist())
                support_labels.extend([class_label.item()] * len(support_nodes))
                query_indices.extend(query_nodes.tolist())
                query_labels.extend([class_label.item()] * len(query_nodes))
            
            if len(query_indices) == 0:  # Skip if no query nodes
                continue
            
            # Perform inference
            predictions, confidence = self.few_shot_inference(
                graph_data, support_indices, support_labels, query_indices
            )
            
            # Compute accuracy
            query_labels_tensor = torch.tensor(query_labels, device=self.device)
            episode_accuracy = (predictions == query_labels_tensor).float().mean().item()
            
            accuracies.append(episode_accuracy)
            confidences.extend(confidence.tolist())
        
        # Compute final metrics
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_confidence = np.mean(confidences)
        
        results = {
            'accuracy_mean': mean_accuracy,
            'accuracy_std': std_accuracy,
            'confidence_mean': mean_confidence,
            'n_episodes': len(accuracies)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results


def load_pretrained_model(model_path: str, device: str = "cuda") -> InContextLearner:
    """
    Load a pretrained G-Align model for in-context learning.
    
    Args:
        model_path: Path to the saved model
        device: Device to run on
        
    Returns:
        InContextLearner instance
    """
    return InContextLearner(model_path, device)


# Example usage
if __name__ == "__main__":
    # Load pretrained model
    model_path = "output/G-Align/final_gfm_model.pt"  # Update with actual path
    
    if osp.exists(model_path):
        learner = load_pretrained_model(model_path)
        
        # Evaluate on downstream dataset
        results = learner.evaluate_on_dataset("cora", k_shot=5, n_episodes=50)
        print(f"Evaluation results: {results}")
        
        # Example of few-shot inference
        graph_data = learner.load_downstream_graph("cora")
        support_indices = [0, 1, 2, 3, 4]  # Example indices
        support_labels = [0, 0, 1, 1, 2]    # Example labels
        query_indices = [10, 11, 12]        # Example query nodes
        
        predictions, confidences = learner.few_shot_inference(
            graph_data, support_indices, support_labels, query_indices
        )
        
        print(f"Predictions: {predictions}")
        print(f"Confidences: {confidences}")
    else:
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please run pretraining first: python pretrain.py")