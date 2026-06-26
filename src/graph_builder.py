"""
OmniGen - Graph Construction Module
Builds heterogeneous graph from DRKG triplets for drug repurposing.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
import warnings


class DRKGGraphBuilder:
    """
    Builds a PyTorch Geometric HeteroData object from DRKG triplets.
    Focuses on drug repurposing: Compound -> Disease prediction.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
        # Entity mappings: {entity_type: {original_id: index}}
        self.entity_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_entity: Dict[str, Dict[int, str]] = {}
        
        # Node feature dimensions
        self.hidden_dim = 128
        
    def _extract_entity_type(self, entity: str) -> str:
        """Extract type from 'Type::ID' format."""
        return entity.split("::")[0]
    
    def _get_main_types(self) -> List[str]:
        """Return main entity types for drug repurposing."""
        return ["Compound", "Disease", "Gene"]
    
    def _is_relevant_relation(self, rel_name: str) -> bool:
        """Check if relation is useful for drug repurposing."""
        relevant_keywords = [
            "treat", "therap", "indicat", "palliat", "prevent",
            "interact", "target", "bind", "inhibit", "activat",
            "agonist", "antagonist", "substrate", "enzyme", "carrier",
            "transporter", "associated", "correlated"
        ]
        rel_lower = rel_name.lower()
        return any(kw in rel_lower for kw in relevant_keywords)
    
    def build_entity_mappings(self, entity_types: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        """
        Create index mappings for each entity type.
        """
        if entity_types is None:
            entity_types = self._get_main_types()
        
        print(f"Building mappings for: {entity_types}")
        
        for etype in entity_types:
            # Get all entities of this type from both head and tail
            mask_head = self.df["head"].str.startswith(f"{etype}::")
            mask_tail = self.df["tail"].str.startswith(f"{etype}::")
            
            entities = set(self.df[mask_head]["head"]) | set(self.df[mask_tail]["tail"])
            entities = sorted(list(entities))
            
            self.entity_to_idx[etype] = {e: i for i, e in enumerate(entities)}
            self.idx_to_entity[etype] = {i: e for e, i in self.entity_to_idx[etype].items()}
            
            print(f"  {etype}: {len(entities):,} entities")
        
        return self.entity_to_idx
    
    def build_hetero_data(self, sample_size: Optional[int] = None, 
                          filter_relations: bool = True) -> HeteroData:
        """
        Build PyTorch Geometric HeteroData object.
        
        Args:
            sample_size: If set, sample N triplets for quick testing (None = full graph)
            filter_relations: If True, only keep relations relevant to drug repurposing
        """
        if sample_size:
            print(f"Sampling {sample_size:,} triplets for testing...")
            df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            df = self.df
            print(f"Using full dataset: {len(df):,} triplets")
        
        # Filter to relevant relations for drug repurposing
        if filter_relations:
            print("Filtering to drug repurposing-relevant relations...")
            df = df[df["relation"].apply(self._is_relevant_relation)].reset_index(drop=True)
            print(f"Filtered to {len(df):,} relevant triplets")
        
        # Build mappings
        self.build_entity_mappings()
        
        data = HeteroData()
        
        # Add node features (learnable embeddings)
        for etype, mapping in self.entity_to_idx.items():
            num_nodes = len(mapping)
            # Random initialization - will be trained
            data[etype].x = torch.randn(num_nodes, self.hidden_dim)
            # Store original IDs for reference
            data[etype].node_ids = list(mapping.keys())
            print(f"Added {num_nodes:,} nodes for {etype}")
        
        # Add edges by relation type
        relation_groups = df.groupby("relation")
        
        edge_count = 0
        for rel_name, rel_df in relation_groups:
            # Parse relation to get head and tail types
            head_type = self._extract_entity_type(rel_df.iloc[0]["head"])
            tail_type = self._extract_entity_type(rel_df.iloc[0]["tail"])
            
            # Skip if types not in our mappings
            if head_type not in self.entity_to_idx or tail_type not in self.entity_to_idx:
                continue
            
            # Map entities to indices
            src = []
            dst = []
            for _, row in rel_df.iterrows():
                s = self.entity_to_idx[head_type].get(row["head"])
                t = self.entity_to_idx[tail_type].get(row["tail"])
                if s is not None and t is not None:
                    src.append(s)
                    dst.append(t)
            
            if not src:
                continue
            
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            
            # Clean relation name for PyG (no special chars)
            clean_rel = rel_name.replace("::", "_").replace(":", "_").replace(" ", "_")
            edge_type = (head_type, clean_rel, tail_type)
            
            data[edge_type].edge_index = edge_index
            edge_count += len(src)
            
            if len(src) > 100:  # Only print significant edge types
                print(f"  Added {len(src):,} edges: {edge_type}")
        
        print(f"\nTotal edges added: {edge_count:,}")
        print(f"Edge types: {len(data.edge_types)}")
        
        # Store metadata
        data.num_nodes_dict = {k: len(v) for k, v in self.entity_to_idx.items()}
        
        return data
    
    def get_target_edges(self, data: HeteroData, target_pattern: str = "treat") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract positive Compound-Disease pairs for training.
        """
        # Find Compound-Disease edges matching pattern
        target_edge_types = [
            et for et in data.edge_types 
            if et[0] == "Compound" and et[2] == "Disease" and target_pattern in et[1].lower()
        ]
        
        if not target_edge_types:
            print("Warning: No direct Compound-Disease treatment edges found")
            print("Available Compound-Disease edges:", [
                et for et in data.edge_types 
                if et[0] == "Compound" and et[2] == "Disease"
            ])
            return None, None
        
        # Use first matching edge type
        target_et = target_edge_types[0]
        print(f"\nTarget edge type: {target_et}")
        
        edge_index = data[target_et].edge_index
        labels = torch.ones(edge_index.size(1))
        
        print(f"Positive pairs: {edge_index.size(1):,}")
        
        return edge_index, labels
    
    def create_negative_samples(self, num_negatives: int, data: HeteroData) -> torch.Tensor:
        """
        Create negative samples: random Compound-Disease pairs that don't exist.
        """
        num_compounds = data["Compound"].num_nodes
        num_diseases = data["Disease"].num_nodes
        
        neg_src = torch.randint(0, num_compounds, (num_negatives,))
        neg_dst = torch.randint(0, num_diseases, (num_negatives,))
        
        return torch.stack([neg_src, neg_dst], dim=0)


if __name__ == "__main__":
    # Test with larger sample
    from data_loader import DRKGLoader
    
    print("=" * 60)
    print("Testing Graph Builder")
    print("=" * 60)
    
    loader = DRKGLoader("data/raw")
    df = loader.load("drkg.tsv")
    
    # Use larger sample and filter relations
    builder = DRKGGraphBuilder(df)
    data = builder.build_hetero_data(sample_size=100000, filter_relations=True)
    
    print(f"\nGraph built!")
    print(f"Node types: {list(data.node_types)}")
    print(f"Total edge types: {len(data.edge_types)}")
    
    # Test target edge extraction
    edge_index, labels = builder.get_target_edges(data)
    if edge_index is not None:
        print(f"Labels shape: {labels.shape}")
        
        # Test negative sampling
        neg_edges = builder.create_negative_samples(1000, data)
        print(f"Negative samples: {neg_edges.shape}")