import sys
sys.path.append("src")

import torch
from src.data_loader import DRKGLoader
from src.graph_builder import DRKGGraphBuilder
from src.model import OmniGenGNN, DrugRepurposingTrainer

print("=" * 60)
print("Testing OmniGen GNN Model")
print("=" * 60)

# Load data
loader = DRKGLoader("data/raw")
df = loader.load("drkg.tsv")

# Build graph (use 50K for faster testing)
builder = DRKGGraphBuilder(df)
data = builder.build_hetero_data(sample_size=50000, filter_relations=True)

# Get positive edges
pos_edges, labels = builder.get_target_edges(data)
if pos_edges is None:
    print("No target edges found. Exiting.")
    sys.exit(1)

print(f"\nTotal positive edges: {pos_edges.size(1)}")

# Split into train/val
num_pos = pos_edges.size(1)
num_train = int(0.8 * num_pos)

# Shuffle
perm = torch.randperm(num_pos)
pos_train = pos_edges[:, perm[:num_train]]
pos_val = pos_edges[:, perm[num_train:]]

# Create negative samples
neg_train = builder.create_negative_samples(num_train, data)
neg_val = builder.create_negative_samples(num_pos - num_train, data)

print(f"Train: {pos_train.size(1)} pos, {neg_train.size(1)} neg")
print(f"Val: {pos_val.size(1)} pos, {neg_val.size(1)} neg")

# Initialize model with num_nodes_dict
model = OmniGenGNN(
    hidden_channels=64,
    out_channels=32,
    num_layers=2,
    num_heads=2,
    dropout=0.3,
    edge_types=data.edge_types,
    num_nodes_dict=data.num_nodes_dict
)

print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Train
trainer = DrugRepurposingTrainer(model, lr=0.005)
history = trainer.train(
    data, pos_train, neg_train,
    pos_val, neg_val,
    epochs=50, patience=10
)

print("\nTraining complete!")