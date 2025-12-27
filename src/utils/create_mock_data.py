"""
Create a small mock processed dataset for quick testing.
This bypasses the expensive BERT preprocessing step.
"""
import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.gcn_dataset import GCNDataset


def create_mock_processed_data(output_path, num_graphs=20, avg_nodes=10, num_features=774):
    """
    Create mock processed dataset for testing.

    Args:
        output_path: Path to save the processed data file
        num_graphs: Number of graphs to generate
        avg_nodes: Average number of nodes per graph
        num_features: Number of features per node (774 = 768 BERT + 6 time features)
    """
    print(f"Creating mock dataset with {num_graphs} graphs, ~{avg_nodes} nodes each...")
    print(f"Feature dimension: {num_features}")

    data_list = []

    for i in range(num_graphs):
        # Random number of nodes around avg_nodes
        num_nodes = max(3, int(avg_nodes + np.random.randn() * 3))

        # Generate random node features
        x = torch.randn(num_nodes, num_features)

        # Generate random edges (sparse graph)
        num_edges = int(num_nodes * 1.5)  # ~1.5 edges per node
        edges = []
        for _ in range(num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append([src, dst])

        if not edges:
            edges = [[0, 1], [1, 0]]  # At least one edge

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Random label (0 or 1)
        y = torch.tensor(np.random.randint(0, 2), dtype=torch.long)

        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_graphs} graphs")

    # Collate into single dataset
    print("\nCollating graphs...")
    data, slices = GCNDataset.collate(data_list)

    # Save processed data
    print(f"Saving to: {output_path}")
    torch.save((data, slices), output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total graphs: {num_graphs}")
    print(f"  Total nodes: {sum(g.num_nodes for g in data_list)}")
    print(f"  Total edges: {sum(g.num_edges for g in data_list)}")
    label_dist = {0: 0, 1: 0}
    for g in data_list:
        label_dist[g.y.item()] += 1
    print(f"  Label distribution: {label_dist}")
    print(f"  Feature dimension: {num_features}")

    return data, slices


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_dir = os.path.join(root_path, "data", "CED_Processed")
    processed_file = os.path.join(processed_dir, "CED_data_features.pt")

    # Create directory if not exists
    os.makedirs(processed_dir, exist_ok=True)

    # Create mock data
    create_mock_processed_data(
        output_path=processed_file,
        num_graphs=20,      # Small number for quick testing
        avg_nodes=10,        # Small graphs
        num_features=774     # Matches actual BERT + time features
    )

    print("\n" + "="*60)
    print("Mock dataset created successfully!")
    print("Now you can run:")
    print("  python src/trainers/train_gcn.py --quick-test")
    print("="*60)


if __name__ == "__main__":
    main()
