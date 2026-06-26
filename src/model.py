"""
OmniGen - GNN Model Module
Heterogeneous Graph Neural Network for drug repurposing.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional


class OmniGenGNN(nn.Module):
    """
    Heterogeneous GNN for drug repurposing using Graph Attention Networks.
    Predicts missing Compound-Disease treatment edges.
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        edge_types: Optional[list] = None,
        num_nodes_dict: Optional[Dict[str, int]] = None
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_types = edge_types or []
        self.num_nodes_dict = num_nodes_dict or {}
        
        # Initialize node embeddings (learnable)
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in self.num_nodes_dict.items()
        })
        
        # Message passing layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
        
        # Batch normalization
        self.bns = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(hidden_channels)
                for node_type in self.num_nodes_dict.keys()
            })
            for _ in range(num_layers)
        ])
        
        # Edge prediction decoder
        self.decoder = nn.Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 1)
        )
        
    def encode(self, x_dict: Dict[str, torch.Tensor], 
               edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Message passing: compute node embeddings.
        """
        # Use learnable embeddings
        h_dict = {
            node_type: self.node_embeddings[node_type](
                torch.arange(num_nodes, device=self.node_embeddings[node_type].weight.device)
            )
            for node_type, num_nodes in self.num_nodes_dict.items()
        }
        
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            for node_type in h_dict.keys():
                h_dict[node_type] = F.relu(h_dict[node_type])
                if h_dict[node_type].size(0) > 1:
                    h_dict[node_type] = self.bns[i][node_type](h_dict[node_type])
            
            h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) 
                     for k, v in h_dict.items()}
        
        return h_dict
    
    def decode(self, z_dict: Dict[str, torch.Tensor],
               edge_label_index: torch.Tensor,
               src_type: str = 'Compound',
               dst_type: str = 'Disease') -> torch.Tensor:
        """
        Score drug-disease pairs using learned embeddings.
        """
        src = z_dict[src_type][edge_label_index[0]]
        dst = z_dict[dst_type][edge_label_index[1]]
        
        pair_features = torch.cat([src, dst], dim=-1)
        return self.decoder(pair_features).squeeze(-1)
    
    def forward(self, data: HeteroData, 
                edge_label_index: torch.Tensor,
                src_type: str = 'Compound',
                dst_type: str = 'Disease') -> torch.Tensor:
        """
        Full forward pass: encode + decode.
        """
        z_dict = self.encode(data.x_dict, data.edge_index_dict)
        return self.decode(z_dict, edge_label_index, src_type, dst_type)
    
    def predict(self, data: HeteroData, 
                drug_idx: torch.Tensor,
                disease_idx: torch.Tensor) -> torch.Tensor:
        """
        Predict treatment probability for specific drug-disease pairs.
        """
        self.eval()
        with torch.no_grad():
            z_dict = self.encode(data.x_dict, data.edge_index_dict)
            edge_label_index = torch.stack([drug_idx, disease_idx])
            scores = self.decode(z_dict, edge_label_index)
            return torch.sigmoid(scores)


class DrugRepurposingTrainer:
    """
    Training pipeline for drug repurposing model.
    """
    
    def __init__(self, model: OmniGenGNN, lr: float = 0.001, 
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {'train_loss': [], 'val_auc': []}
        
    def train_epoch(self, data: HeteroData, 
                    pos_edge_index: torch.Tensor,
                    neg_edge_index: torch.Tensor) -> float:
        """
        Train for one epoch with positive and negative edges.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        num_pos = pos_edge_index.size(1)
        num_neg = neg_edge_index.size(1)
        
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat([
            torch.ones(num_pos),
            torch.zeros(num_neg)
        ])
        
        out = self.model(data, edge_label_index)
        loss = self.criterion(out, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return float(loss.detach())
    
    @torch.no_grad()
    def evaluate(self, data: HeteroData,
                 pos_edge_index: torch.Tensor,
                 neg_edge_index: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        """
        self.model.eval()
        
        num_pos = pos_edge_index.size(1)
        num_neg = neg_edge_index.size(1)
        
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        labels = torch.cat([
            torch.ones(num_pos),
            torch.zeros(num_neg)
        ])
        
        out = self.model(data, edge_label_index)
        probs = torch.sigmoid(out)
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        auc = roc_auc_score(labels_np, probs_np)
        ap = average_precision_score(labels_np, probs_np)
        
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean().item()
        
        return {
            'auc': auc,
            'ap': ap,
            'accuracy': acc,
            'loss': float(self.criterion(out, labels))
        }
    
    def train(self, data: HeteroData,
              pos_train: torch.Tensor,
              neg_train: torch.Tensor,
              pos_val: torch.Tensor,
              neg_val: torch.Tensor,
              epochs: int = 100,
              patience: int = 10) -> Dict:
        """
        Full training loop with early stopping.
        """
        os.makedirs("models", exist_ok=True)
        
        best_val_auc = 0
        patience_counter = 0
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Training edges: {pos_train.size(1):,} pos, {neg_train.size(1):,} neg")
        print(f"Validation edges: {pos_val.size(1):,} pos, {neg_val.size(1):,} neg")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(data, pos_train, neg_train)
            self.history['train_loss'].append(train_loss)
            
            val_metrics = self.evaluate(data, pos_val, neg_val)
            self.history['val_auc'].append(val_metrics['auc'])
            
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val AUC={val_metrics['auc']:.4f}, "
                      f"Val AP={val_metrics['ap']:.4f}")
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                torch.save(self.model.state_dict(), "models/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        print(f"\nBest Val AUC: {best_val_auc:.4f}")
        return self.history


if __name__ == "__main__":
    print("OmniGen GNN Model loaded successfully!")
    print("Use DrugRepurposingTrainer for training.")