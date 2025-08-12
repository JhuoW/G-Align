#!/usr/bin/env python
"""
Enhanced prototype-based in-context learning for G-Align.
Combines multiple strategies for improved performance:
1. Weighted prototypes based on confidence
2. Graph structure-aware prototypes
3. Adaptive distance metrics
4. Semi-supervised refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from icl import PrototypeInContextLearner
from utils.logging import logger


class GraphAwarePrototypeLearner(PrototypeInContextLearner):
    """Enhanced prototype learner with graph-aware features."""
    
    def __init__(self, args):
        super().__init__(args)
        self.label_propagation = LabelPropagation(num_layers=3, alpha=0.9)
        
    def compute_graph_aware_prototypes(self, 
                                      features: torch.Tensor,
                                      labels: torch.Tensor,
                                      edge_index: torch.Tensor,
                                      unique_labels: torch.Tensor,
                                      mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute prototypes with graph structure awareness.
        
        Args:
            features: Node features [n_nodes, d]
            labels: Node labels [n_nodes]
            edge_index: Graph edges [2, n_edges]
            unique_labels: Unique class labels
            mask: Support set mask
            
        Returns:
            Graph-aware prototypes [n_classes, d]
        """
        n_nodes = features.shape[0]
        n_classes = len(unique_labels)
        
        # Initialize soft labels
        soft_labels = torch.zeros(n_nodes, n_classes, device=features.device)
        
        # Set hard labels for support set
        if mask is not None:
            for idx, label in enumerate(unique_labels):
                class_mask = mask & (labels == label)
                if class_mask.sum() > 0:
                    soft_labels[class_mask, idx] = 1.0
        
        # Propagate labels through graph
        propagated_labels = self.label_propagation(soft_labels, edge_index)
        
        # Compute weighted prototypes
        prototypes = []
        for idx, label in enumerate(unique_labels):
            # Get propagated weights for this class
            class_weights = propagated_labels[:, idx]
            
            # Weight threshold
            weight_threshold = 0.1
            valid_mask = class_weights > weight_threshold
            
            if valid_mask.sum() > 0:
                # Weighted average of features
                weighted_features = features[valid_mask] * class_weights[valid_mask].unsqueeze(-1)
                prototype = weighted_features.sum(dim=0) / class_weights[valid_mask].sum()
            else:
                # Fallback to simple mean
                class_mask = labels == label
                if mask is not None:
                    class_mask = class_mask & mask
                if class_mask.sum() > 0:
                    prototype = features[class_mask].mean(dim=0)
                else:
                    prototype = torch.zeros_like(features[0])
            
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        return prototypes
    
    def adaptive_distance(self, 
                         query_features: torch.Tensor,
                         prototypes: torch.Tensor,
                         confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute adaptive distance based on feature statistics.
        
        Args:
            query_features: Query features [n_query, d]
            prototypes: Class prototypes [n_classes, d]
            confidence: Optional confidence weights
            
        Returns:
            Distance scores [n_query, n_classes]
        """
        # Normalize features
        query_norm = F.normalize(query_features, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        
        # Cosine similarity
        cos_sim = torch.mm(query_norm, proto_norm.t())
        
        # Euclidean distance (normalized)
        eucl_dist = torch.cdist(query_norm, proto_norm, p=2)
        eucl_sim = 1.0 / (1.0 + eucl_dist)  # Convert to similarity
        
        # Adaptive combination based on feature variance
        query_var = query_features.var(dim=-1, keepdim=True)
        alpha = torch.sigmoid(query_var)  # High variance -> prefer cosine
        
        combined_sim = alpha * cos_sim + (1 - alpha) * eucl_sim
        
        # Apply confidence weighting if available
        if confidence is not None:
            combined_sim = combined_sim * confidence.unsqueeze(0)
        
        return combined_sim
    
    def semi_supervised_refinement(self,
                                   features: torch.Tensor,
                                   initial_predictions: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   support_mask: torch.Tensor,
                                   num_iterations: int = 5) -> torch.Tensor:
        """
        Refine predictions using semi-supervised learning.
        
        Args:
            features: Node features [n_nodes, d]
            initial_predictions: Initial class predictions [n_nodes]
            edge_index: Graph edges
            support_mask: Mask for support set
            num_iterations: Number of refinement iterations
            
        Returns:
            Refined predictions [n_nodes]
        """
        n_nodes = features.shape[0]
        n_classes = initial_predictions.max().item() + 1
        
        # Initialize pseudo-labels
        pseudo_labels = torch.zeros(n_nodes, n_classes, device=features.device)
        for i in range(n_nodes):
            if support_mask[i]:
                # Keep support set labels fixed
                pseudo_labels[i, initial_predictions[i]] = 1.0
            else:
                # Soft labels for query set
                pseudo_labels[i, initial_predictions[i]] = 0.8
        
        # Iterative refinement
        for _ in range(num_iterations):
            # Propagate labels
            propagated = self.label_propagation(pseudo_labels, edge_index)
            
            # Update non-support nodes
            pseudo_labels[~support_mask] = 0.9 * propagated[~support_mask] + 0.1 * pseudo_labels[~support_mask]
            
            # Normalize
            pseudo_labels = F.softmax(pseudo_labels, dim=-1)
        
        # Get final predictions
        refined_predictions = pseudo_labels.argmax(dim=-1)
        
        return refined_predictions
    
    @torch.no_grad()
    def enhanced_prototype_inference(self,
                                    graph_data,
                                    k_shot: int = 5,
                                    seed: int = 42,
                                    use_graph_aware: bool = True,
                                    use_adaptive_distance: bool = True,
                                    use_refinement: bool = True) -> Dict:
        """
        Enhanced prototype-based inference with multiple strategies.
        
        Args:
            graph_data: Target graph
            k_shot: Number of support examples per class
            seed: Random seed
            use_graph_aware: Use graph-aware prototypes
            use_adaptive_distance: Use adaptive distance metric
            use_refinement: Use semi-supervised refinement
            
        Returns:
            Results dictionary
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info("Running enhanced prototype-based inference")
        logger.info(f"Settings: graph_aware={use_graph_aware}, adaptive={use_adaptive_distance}, refinement={use_refinement}")
        
        # Compute domain embedding and FiLM parameters
        domain_embedding = self.compute_domain_embedding(graph_data)
        gamma_f, beta_f, gamma_l, beta_l = self.domain_embedder.dm_film(domain_embedding.unsqueeze(0))
        gamma_f, beta_f = gamma_f.squeeze(0), beta_f.squeeze(0)
        gamma_l, beta_l = gamma_l.squeeze(0), beta_l.squeeze(0)
        
        # Get node embeddings
        H, _ = self.backbone_gnn.encode(
            graph_data.x,
            graph_data.edge_index,
            graph_data.xe if hasattr(graph_data, 'xe') else None,
            graph_data.batch if hasattr(graph_data, 'batch') else None
        )
        
        # Apply domain alignment
        z_all = gamma_f * H + beta_f
        z_all = F.normalize(z_all, p=2, dim=-1)
        
        # Setup for few-shot learning
        labels = graph_data.y
        unique_labels = torch.unique(labels).sort()[0]
        num_classes = len(unique_labels)
        
        # Get train/test masks
        if hasattr(graph_data, 'train_mask') and hasattr(graph_data, 'test_mask'):
            train_mask = graph_data.train_mask
            test_mask = graph_data.test_mask
        else:
            n_nodes = labels.shape[0]
            perm = torch.randperm(n_nodes)
            split_idx = int(0.8 * n_nodes)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
            train_mask[perm[:split_idx]] = True
            test_mask[perm[split_idx:]] = True
        
        # Sample support set
        support_mask = torch.zeros_like(train_mask)
        support_labels = []
        
        for class_label in unique_labels:
            class_train_mask = train_mask & (labels == class_label)
            class_train_indices = class_train_mask.nonzero(as_tuple=False).squeeze(-1)
            
            if class_train_indices.numel() > 0:
                n_samples = min(k_shot, len(class_train_indices))
                perm = torch.randperm(len(class_train_indices))[:n_samples]
                selected = class_train_indices[perm]
                support_mask[selected] = True
                support_labels.extend([class_label.item()] * n_samples)
        
        logger.info(f"Support set: {support_mask.sum().item()} nodes from {num_classes} classes")
        
        # Compute prototypes
        if use_graph_aware:
            prototypes = self.compute_graph_aware_prototypes(
                z_all, labels, graph_data.edge_index, unique_labels, support_mask
            )
        else:
            support_features = z_all[support_mask]
            support_labels_tensor = labels[support_mask]
            prototypes = self.compute_prototypes(support_features, support_labels_tensor, unique_labels)
        
        # Get initial predictions for all nodes
        if use_adaptive_distance:
            similarities = self.adaptive_distance(z_all, prototypes)
        else:
            similarities = self.prototype_distance(z_all, prototypes, 'cosine')
        
        initial_predictions = similarities.argmax(dim=1)
        
        # Map to original labels
        predicted_labels = torch.zeros_like(labels)
        for i, label in enumerate(unique_labels):
            mask = initial_predictions == i
            predicted_labels[mask] = label
        
        # Apply refinement if requested
        if use_refinement and graph_data.edge_index.shape[1] > 0:
            refined_predictions = self.semi_supervised_refinement(
                z_all, predicted_labels, graph_data.edge_index, support_mask, num_iterations=5
            )
            predicted_labels[~support_mask] = refined_predictions[~support_mask]
        
        # Evaluate on test set
        test_predictions = predicted_labels[test_mask].cpu().numpy()
        test_true = labels[test_mask].cpu().numpy()
        
        accuracy = (test_predictions == test_true).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for class_label in unique_labels:
            class_mask = test_true == class_label.item()
            if class_mask.sum() > 0:
                class_acc = (test_predictions[class_mask] == test_true[class_mask]).mean()
                per_class_acc[class_label.item()] = float(class_acc)
        
        # Get confidence scores
        test_similarities = similarities[test_mask]
        test_probs = F.softmax(test_similarities / 0.1, dim=1)
        confidences = test_probs.max(dim=1)[0].cpu().numpy()
        
        results = {
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'mean_confidence': float(confidences.mean()),
            'num_test_samples': test_mask.sum().item(),
            'num_classes': num_classes,
            'k_shot': k_shot,
            'use_graph_aware': use_graph_aware,
            'use_adaptive_distance': use_adaptive_distance,
            'use_refinement': use_refinement
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Mean confidence: {confidences.mean():.4f}")
        
        return results


class LabelPropagation(MessagePassing):
    """Label propagation layer for semi-supervised learning."""
    
    def __init__(self, num_layers: int = 3, alpha: float = 0.9):
        super().__init__(aggr='add')
        self.num_layers = num_layers
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Propagate labels through graph.
        
        Args:
            x: Node label distributions [n_nodes, n_classes]
            edge_index: Graph edges [2, n_edges]
            
        Returns:
            Propagated labels [n_nodes, n_classes]
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Store original labels
        x_orig = x.clone()
        
        # Propagate
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = self.alpha * x + (1 - self.alpha) * x_orig
        
        return x
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced G-Align Prototype ICL")
    parser.add_argument('--model_path', type=str, default='generated_files/output/G-Align/Aug11-17:00-9929422f/final_gfm_model.pt')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Initialize enhanced learner
    learner = GraphAwarePrototypeLearner(args)
    
    # Load dataset
    graph_data = learner.load_downstream_graph(args.dataset)
    
    logger.info("="*60)
    logger.info("Enhanced Prototype-based In-Context Learning")
    logger.info("="*60)
    
    # Test different configurations
    configs = [
        (False, False, False),  # Baseline prototypes
        (True, False, False),   # Graph-aware only
        (False, True, False),   # Adaptive distance only
        (False, False, True),   # Refinement only
        (True, True, False),    # Graph + Adaptive
        (True, True, True),     # All enhancements
    ]
    
    config_names = [
        "Baseline",
        "Graph-aware",
        "Adaptive distance",
        "Refinement",
        "Graph + Adaptive",
        "All enhancements"
    ]
    
    results_summary = []
    
    for (use_graph, use_adaptive, use_refine), name in zip(configs, config_names):
        logger.info(f"\nTesting: {name}")
        
        accuracies = []
        for run in range(min(5, args.n_runs)):
            results = learner.enhanced_prototype_inference(
                graph_data,
                k_shot=args.k_shot,
                seed=42+run,
                use_graph_aware=use_graph,
                use_adaptive_distance=use_adaptive,
                use_refinement=use_refine
            )
            accuracies.append(results['accuracy'])
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        results_summary.append({
            'config': name,
            'mean_acc': mean_acc,
            'std_acc': std_acc
        })
        
        print(f"  {name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - Best Configuration:")
    print("="*60)
    
    best_config = max(results_summary, key=lambda x: x['mean_acc'])
    print(f"Best: {best_config['config']}")
    print(f"Accuracy: {best_config['mean_acc']:.4f} ± {best_config['std_acc']:.4f}")


if __name__ == "__main__":
    main()