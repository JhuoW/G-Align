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
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
import os.path as osp
from tqdm import tqdm


class InContextLearner:
    """In-context learning interface for G-Align pretrained models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize the in-context learner.
        
        Args:
            model_path: Path to the saved pretrained model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Load pretrained model state
        logger.info(f"Loading pretrained model from {model_path}")
        self.model_state = torch.load(model_path, map_location=self.device)
        
        # Extract configuration and metadata
        self.cfg = self.model_state['config']
        self.L_max = self.model_state['L_max']
        self.pretrain_datasets = self.model_state['pretrain_datasets']
        self.pretrain_tasks = self.model_state.get('pretrain_tasks', [])
        
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
        
        # Load the convolutional projection network state
        if 'domain_embedder_projection_state' in self.model_state:
            self.domain_embedder.dm_extractor.projection.load_state_dict(
                self.model_state['domain_embedder_projection_state']
            )
        
        # Load other cached components
        self.domain_embedder.dm_extractor._theta0 = self.model_state['domain_embedder_theta0']
        self.domain_embedder.dm_extractor._e = self.model_state.get('domain_embedder_e')
        
        # Optional: load delta matrices if saved
        if 'domain_embedder_delta_matrices' in self.model_state:
            self.domain_embedder.dm_extractor._delta_matrices = self.model_state['domain_embedder_delta_matrices']
        
        self.domain_embedder.dm_extractor._cached = True
        self.domain_embedder.to(self.device).eval()
        
        # Initialize GFM model
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
        self.pama = self.model.pama
        self.E_label = self.model.E_lab
    
    def load_downstream_graph(self, 
                             dataset_name: str = None, 
                             graph_data: Data = None,
                             cfg_override: Optional[Dict] = None) -> Data:
        """
        Load a downstream graph dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cora', 'citeseer')
            graph_data: Pre-loaded graph data
            cfg_override: Optional configuration overrides
            
        Returns:
            Processed graph data
        """
        if graph_data is not None:
            return graph_data.to(self.device)
        
        if dataset_name is None:
            raise ValueError("Either dataset_name or graph_data must be provided")
        
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
    
    @torch.no_grad()
    def compute_domain_embedding(self, graph_data: Data) -> torch.Tensor:
        """
        Compute domain embedding for a new graph using in-context learning.
        
        Args:
            graph_data: The new graph data
            
        Returns:
            Domain embedding tensor
        """
        logger.info("Computing domain embedding for new graph...")
        
        # Ensure graph has required attributes
        if not hasattr(graph_data, 'batch'):
            graph_data.batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=self.device)
        
        # Use the domain embedder to compute fingerprint for unseen graph
        domain_embedding = self.domain_embedder.fingerprint_unseen(graph_data)
        
        return domain_embedding
    
    @torch.no_grad()
    def few_shot_inference(self, 
                          graph_data: Data, 
                          support_indices: Union[List[int], torch.Tensor], 
                          support_labels: Union[List[int], torch.Tensor],
                          query_indices: Union[List[int], torch.Tensor],
                          return_probs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform few-shot inference on a new graph.
        
        Args:
            graph_data: The target graph data
            support_indices: Node indices for support set
            support_labels: Labels for support nodes
            query_indices: Node indices for queries
            return_probs: Whether to return probability distributions
            
        Returns:
            Tuple of (predictions, confidence_scores/probabilities)
        """
        # Convert to tensors if needed
        support_indices = torch.tensor(support_indices, device=self.device) if not isinstance(support_indices, torch.Tensor) else support_indices.to(self.device)
        support_labels = torch.tensor(support_labels, device=self.device) if not isinstance(support_labels, torch.Tensor) else support_labels.to(self.device)
        query_indices = torch.tensor(query_indices, device=self.device) if not isinstance(query_indices, torch.Tensor) else query_indices.to(self.device)
        
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
        unique_labels = torch.unique(support_labels)
        U_sup_base = self.E_label[unique_labels]
        U_sup = gamma_l * U_sup_base + beta_l
        
        # Create label mapping
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        
        # Perform inference for each query
        predictions = []
        confidence_scores = []
        all_probs = []
        
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
            if return_probs:
                all_probs.append(probs)
        
        predictions = torch.tensor(predictions, device=self.device)
        confidence_scores = torch.tensor(confidence_scores, device=self.device)
        
        if return_probs:
            return predictions, torch.stack(all_probs)
        else:
            return predictions, confidence_scores
    
    @torch.no_grad()
    def zero_shot_inference(self, 
                           graph_data: Data, 
                           query_indices: Union[List[int], torch.Tensor],
                           num_classes: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform zero-shot inference using pretrained label embeddings.
        
        Args:
            graph_data: The target graph data
            query_indices: Node indices for queries
            num_classes: Number of classes in the target task
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        query_indices = torch.tensor(query_indices, device=self.device) if not isinstance(query_indices, torch.Tensor) else query_indices.to(self.device)
        
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
        
        # Use all or specified number of label embeddings
        if num_classes is None:
            num_classes = min(self.L_max, int(graph_data.y.max().item()) + 1)
        
        U_sup_base = self.E_label[:num_classes]
        U_sup = gamma_l * U_sup_base + beta_l
        
        predictions = []
        confidence_scores = []
        
        for i, q_idx in enumerate(query_indices):
            z_q = z_qry[i]
            
            # Map to label space
            z_q_mapped = self.pama.g(z_q)
            
            # Compute similarity scores
            V_label = self.pama.W_V(U_sup)
            V_label = V_label.view(num_classes, self.pama.heads, -1).mean(dim=1)
            logits = torch.matmul(z_q_mapped.unsqueeze(0), V_label.T).squeeze(0)
            
            probs = F.softmax(logits, dim=-1)
            pred_class_idx = logits.argmax().item()
            confidence = probs[pred_class_idx].item()
            
            predictions.append(pred_class_idx)
            confidence_scores.append(confidence)
        
        return torch.tensor(predictions, device=self.device), torch.tensor(confidence_scores, device=self.device)
    
    def evaluate_on_dataset(self, 
                           dataset_name: str = None,
                           graph_data: Data = None,
                           k_shot: int = 5, 
                           n_episodes: int = 100,
                           m_way: int = None,
                           eval_mode: str = 'few-shot') -> Dict[str, float]:
        """
        Evaluate the model on a downstream dataset using episodic evaluation.
        
        Args:
            dataset_name: Name of the dataset to evaluate on
            graph_data: Pre-loaded graph data
            k_shot: Number of support examples per class
            n_episodes: Number of episodes to run
            m_way: Number of classes per episode (None = use all)
            eval_mode: 'few-shot' or 'zero-shot'
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating in {eval_mode} mode...")
        
        # Load the dataset
        graph_data = self.load_downstream_graph(dataset_name, graph_data)
        
        # Get labels and split information
        labels = graph_data.y
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        if m_way is None:
            m_way = min(5, n_classes)
        
        # Use masks if available, otherwise create random splits
        if hasattr(graph_data, 'train_mask') and hasattr(graph_data, 'test_mask'):
            train_indices = graph_data.train_mask.nonzero(as_tuple=False).squeeze()
            test_indices = graph_data.test_mask.nonzero(as_tuple=False).squeeze()
        else:
            n_nodes = labels.shape[0]
            perm = torch.randperm(n_nodes)
            split_idx = int(0.8 * n_nodes)
            train_indices = perm[:split_idx]
            test_indices = perm[split_idx:]
        
        accuracies = []
        confidences = []
        
        for episode in tqdm(range(n_episodes), desc="Episodes"):
            # Sample classes for this episode
            if n_classes > m_way:
                sampled_classes = unique_labels[torch.randperm(n_classes)[:m_way]]
            else:
                sampled_classes = unique_labels
            
            if eval_mode == 'few-shot':
                support_indices = []
                support_labels = []
                query_indices = []
                query_labels = []
                
                # Sample support and query sets for each class
                for class_label in sampled_classes:
                    # Get train nodes for this class
                    class_train_mask = (labels[train_indices] == class_label)
                    class_train_nodes = train_indices[class_train_mask]
                    
                    # Get test nodes for this class
                    class_test_mask = (labels[test_indices] == class_label)
                    class_test_nodes = test_indices[class_test_mask]
                    
                    if len(class_train_nodes) < k_shot or len(class_test_nodes) < 1:
                        continue
                    
                    # Sample k support nodes from train set
                    support_perm = torch.randperm(len(class_train_nodes))[:k_shot]
                    support_nodes = class_train_nodes[support_perm]
                    
                    # Sample query nodes from test set
                    query_perm = torch.randperm(len(class_test_nodes))[:min(5, len(class_test_nodes))]
                    query_nodes = class_test_nodes[query_perm]
                    
                    support_indices.extend(support_nodes.tolist())
                    support_labels.extend([class_label.item()] * len(support_nodes))
                    query_indices.extend(query_nodes.tolist())
                    query_labels.extend([class_label.item()] * len(query_nodes))
                
                if len(query_indices) == 0:
                    continue
                
                # Perform inference
                predictions, confidence = self.few_shot_inference(
                    graph_data, support_indices, support_labels, query_indices
                )
                
                # Compute accuracy
                query_labels_tensor = torch.tensor(query_labels, device=self.device)
                episode_accuracy = (predictions == query_labels_tensor).float().mean().item()
                
            else:  # zero-shot
                # Sample query nodes from test set
                query_indices = []
                query_labels = []
                
                for class_label in sampled_classes:
                    class_test_mask = (labels[test_indices] == class_label)
                    class_test_nodes = test_indices[class_test_mask]
                    
                    if len(class_test_nodes) < 1:
                        continue
                    
                    query_perm = torch.randperm(len(class_test_nodes))[:min(10, len(class_test_nodes))]
                    query_nodes = class_test_nodes[query_perm]
                    
                    query_indices.extend(query_nodes.tolist())
                    query_labels.extend([class_label.item()] * len(query_nodes))
                
                if len(query_indices) == 0:
                    continue
                
                # Perform zero-shot inference
                predictions, confidence = self.zero_shot_inference(
                    graph_data, query_indices, num_classes=len(sampled_classes)
                )
                
                # Map predictions to sampled classes
                predictions_mapped = sampled_classes[predictions]
                query_labels_tensor = torch.tensor(query_labels, device=self.device)
                episode_accuracy = (predictions_mapped == query_labels_tensor).float().mean().item()
            
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
            'n_episodes': len(accuracies),
            'eval_mode': eval_mode,
            'k_shot': k_shot if eval_mode == 'few-shot' else 0,
            'm_way': m_way
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def cross_dataset_evaluation(self,
                                test_datasets: List[str],
                                k_shot: int = 5,
                                n_episodes: int = 50,
                                eval_mode: str = 'few-shot') -> Dict[str, Dict[str, float]]:
        """
        Evaluate on multiple downstream datasets.
        
        Args:
            test_datasets: List of dataset names to evaluate on
            k_shot: Number of support examples per class
            n_episodes: Number of episodes per dataset
            eval_mode: 'few-shot' or 'zero-shot'
            
        Returns:
            Dictionary mapping dataset names to evaluation results
        """
        all_results = {}
        
        for dataset_name in test_datasets:
            logger.info(f"\nEvaluating on {dataset_name}...")
            try:
                results = self.evaluate_on_dataset(
                    dataset_name=dataset_name,
                    k_shot=k_shot,
                    n_episodes=n_episodes,
                    eval_mode=eval_mode
                )
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate on {dataset_name}: {e}")
                all_results[dataset_name] = {'error': str(e)}
        
        # Compute average metrics
        valid_results = [r for r in all_results.values() if 'accuracy_mean' in r]
        if valid_results:
            avg_accuracy = np.mean([r['accuracy_mean'] for r in valid_results])
            avg_std = np.mean([r['accuracy_std'] for r in valid_results])
            all_results['average'] = {
                'accuracy_mean': avg_accuracy,
                'accuracy_std': avg_std,
                'n_datasets': len(valid_results)
            }
        
        return all_results


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


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="G-Align In-Context Learning")
    parser.add_argument('--model_path', type=str, default="output/G-Align/final_gfm_model.pt",
                       help='Path to pretrained model')
    parser.add_argument('--dataset', type=str, default="cora",
                       help='Dataset to evaluate on')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--n_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--eval_mode', type=str, choices=['few-shot', 'zero-shot'], 
                       default='few-shot', help='Evaluation mode')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    if osp.exists(args.model_path):
        # Load pretrained model
        learner = load_pretrained_model(args.model_path, args.device)
        
        # Single dataset evaluation
        logger.info(f"Evaluating on {args.dataset} in {args.eval_mode} mode...")
        results = learner.evaluate_on_dataset(
            dataset_name=args.dataset,
            k_shot=args.k_shot,
            n_episodes=args.n_episodes,
            eval_mode=args.eval_mode
        )
        
        print("\n" + "="*50)
        print(f"Results on {args.dataset}:")
        print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  Confidence: {results['confidence_mean']:.4f}")
        print("="*50)
        
        # Cross-dataset evaluation
        if args.dataset == "cora":
            # Test on other citation networks
            test_datasets = ["citeseer", "pubmed"]
            logger.info("\nPerforming cross-dataset evaluation...")
            cross_results = learner.cross_dataset_evaluation(
                test_datasets=test_datasets,
                k_shot=args.k_shot,
                n_episodes=args.n_episodes // 2,
                eval_mode=args.eval_mode
            )
            
            print("\nCross-Dataset Results:")
            print("-"*50)
            for dataset, res in cross_results.items():
                if 'accuracy_mean' in res:
                    print(f"{dataset:15s}: {res['accuracy_mean']:.4f} ± {res.get('accuracy_std', 0):.4f}")
            print("-"*50)
        
        # Example of manual few-shot inference
        logger.info("\nExample of manual few-shot inference...")
        graph_data = learner.load_downstream_graph(args.dataset)
        
        # Sample some nodes for demonstration
        n_nodes = graph_data.x.shape[0]
        sample_indices = torch.randperm(n_nodes)[:100]
        
        support_indices = sample_indices[:20].tolist()
        support_labels = graph_data.y[support_indices].tolist()
        query_indices = sample_indices[20:30].tolist()
        
        predictions, confidences = learner.few_shot_inference(
            graph_data, support_indices, support_labels, query_indices
        )
        
        print(f"\nManual inference example:")
        print(f"  Query nodes: {query_indices[:5]}")
        print(f"  Predictions: {predictions[:5].tolist()}")
        print(f"  Confidences: {confidences[:5].tolist()}")
        
    else:
        logger.error(f"Model file not found: {args.model_path}")
        logger.info("Please run pretraining first: python pretrain.py")