#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
import seaborn as sns
from viral_centrality import viral_centrality

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================================
# Data Loading and Preprocessing
# =====================================================================

def load_congress_data():
    """Load the congress network data from JSON file."""
    with open('congress_network_data.json', 'r') as f:
        data = json.load(f)

    inList = data[0]['inList']
    inWeight = data[0]['inWeight']
    outList = data[0]['outList']
    outWeight = data[0]['outWeight']
    usernameList = data[0]['usernameList']

    return inList, inWeight, outList, outWeight, usernameList

def create_networkx_graph(inList, inWeight, outList, outWeight, usernameList):
    """Create a NetworkX graph from the congress network data."""
    G = nx.DiGraph()

    # Add nodes with username attributes
    for i, username in enumerate(usernameList):
        G.add_node(i, username=username)

    # Add edges with weights
    for source, targets in enumerate(outList):
        for idx, target in enumerate(targets):
            weight = outWeight[source][idx]
            G.add_edge(source, target, weight=weight)

    return G

def create_pyg_data(G):
    """Convert NetworkX graph to PyTorch Geometric Data object."""
    # Convert to PyG data
    data = from_networkx(G)

    # Extract edge weights as edge features
    edge_weights = torch.tensor([G[u][v]['weight'] for u, v in G.edges()], dtype=torch.float)
    data.edge_attr = edge_weights.view(-1, 1)

    # Create node features based on network properties
    # Using degree, in-degree, out-degree, and clustering coefficient as features
    degrees = torch.tensor([G.degree(i) for i in range(len(G.nodes()))], dtype=torch.float)
    in_degrees = torch.tensor([G.in_degree(i) for i in range(len(G.nodes()))], dtype=torch.float)
    out_degrees = torch.tensor([G.out_degree(i) for i in range(len(G.nodes()))], dtype=torch.float)

    # Calculate clustering coefficients (for undirected version of the graph)
    G_undirected = G.to_undirected()
    clustering = torch.tensor([nx.clustering(G_undirected, i) for i in range(len(G.nodes()))], dtype=torch.float)

    # Combine features
    node_features = torch.stack([degrees, in_degrees, out_degrees, clustering], dim=1)

    # Normalize features
    scaler = StandardScaler()
    node_features_np = scaler.fit_transform(node_features.numpy())
    data.x = torch.tensor(node_features_np, dtype=torch.float)

    return data

# =====================================================================
# Influence Modeling
# =====================================================================

class InfluenceGNN(nn.Module):
    """GNN model for predicting influence spread."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(InfluenceGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Apply edge weights as attention
        edge_weight = edge_attr.view(-1)

        # Graph convolutions
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)

        # Final prediction
        x = self.lin(x)

        return x

def train_influence_model(data, viral_centrality_scores, epochs=200):
    """Train the influence model using viral centrality as target."""
    # Prepare target: viral centrality scores
    data.y = torch.tensor(viral_centrality_scores, dtype=torch.float)

    # Split data into train and test
    num_nodes = data.x.size(0)
    indices = list(range(num_nodes))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    # Initialize model
    model = InfluenceGNN(in_channels=data.x.size(1),
                         hidden_channels=64,
                         out_channels=1).to(device)

    # Move data to device
    data = data.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask].squeeze(), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index, data.edge_attr)
                test_loss = criterion(pred[data.test_mask].squeeze(), data.y[data.test_mask])

                # Convert to numpy for RMSE calculation
                pred_np = pred[data.test_mask].squeeze().cpu().numpy()
                y_np = data.y[data.test_mask].cpu().numpy()
                rmse = np.sqrt(mean_squared_error(y_np, pred_np))

                print(f'Epoch: {epoch+1:03d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}')
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred_np = pred.squeeze().cpu().numpy()

    return model, pred_np

def visualize_influence_prediction(true_values, predicted_values, usernameList):
    """Visualize the predicted influence vs. true influence."""
    plt.figure(figsize=(12, 8))

    # Scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.6)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')

    # Highlight top influencers
    top_indices = np.argsort(true_values)[-10:]
    for idx in top_indices:
        plt.annotate(usernameList[idx],
                     (true_values[idx], predicted_values[idx]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('True Viral Centrality')
    plt.ylabel('Predicted Influence')
    plt.title('GNN Predicted Influence vs. Viral Centrality')
    plt.grid(True, alpha=0.3)

    # Calculate correlation
    correlation = np.corrcoef(true_values, predicted_values)[0, 1]
    plt.annotate(f'Correlation: {correlation:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('influence_prediction.png')
    plt.close()

# =====================================================================
# Centrality Analysis
# =====================================================================

class CentralityGNN(nn.Module):
    """GNN model for learning node centrality."""
    def __init__(self, in_channels, hidden_channels):
        super(CentralityGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        # Apply edge weights
        edge_weight = edge_attr.view(-1)

        # Graph convolutions
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # Final centrality score
        x = self.conv3(x, edge_index, edge_weight)

        return x

def compute_centrality_measures(G, data, usernameList):
    """Compute various centrality measures for comparison."""
    # Traditional centrality measures
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)

    # Handle eigenvector centrality for potentially disconnected graphs
    try:
        # Try to compute eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    except nx.AmbiguousSolution:
        print("Warning: Graph is disconnected. Using PageRank as an alternative to eigenvector centrality.")
        # Use PageRank as an alternative (works well for disconnected graphs)
        eigenvector_centrality = nx.pagerank(G, weight='weight')
    except Exception as e:
        print(f"Error computing eigenvector centrality: {e}")
        print("Using degree centrality as a fallback.")
        eigenvector_centrality = degree_centrality

    # Convert to numpy arrays
    degree_cent = np.array([degree_centrality[i] for i in range(len(G.nodes()))])
    in_degree_cent = np.array([in_degree_centrality[i] for i in range(len(G.nodes()))])
    out_degree_cent = np.array([out_degree_centrality[i] for i in range(len(G.nodes()))])
    eigenvector_cent = np.array([eigenvector_centrality[i] for i in range(len(G.nodes()))])

    # Train GNN-based centrality model
    model = CentralityGNN(in_channels=data.x.size(1), hidden_channels=64).to(device)
    data = data.to(device)

    # Simple training to learn structure (unsupervised)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)

        # Use degree centrality as a weak supervision signal
        target = torch.tensor(in_degree_cent, dtype=torch.float).view(-1, 1).to(device)
        loss = F.mse_loss(out, target)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')

    # Get GNN centrality scores
    model.eval()
    with torch.no_grad():
        gnn_centrality = model(data.x, data.edge_index, data.edge_attr).cpu().numpy().flatten()

    # Normalize GNN centrality
    gnn_centrality = (gnn_centrality - gnn_centrality.min()) / (gnn_centrality.max() - gnn_centrality.min())

    # Compute viral centrality
    inList, inWeight, outList, _ = load_congress_data()[0:4]
    viral_cent = viral_centrality(inList, inWeight, outList, Niter=-1, tol=0.001)

    # Normalize viral centrality
    viral_cent = (viral_cent - viral_cent.min()) / (viral_cent.max() - viral_cent.min())

    # Create a DataFrame for comparison
    centrality_data = {
        'Node': list(range(len(G.nodes()))),
        'Username': usernameList,
        'Degree': degree_cent,
        'In-Degree': in_degree_cent,
        'Out-Degree': out_degree_cent,
        'Eigenvector': eigenvector_cent,
        'GNN': gnn_centrality,
        'Viral': viral_cent
    }

    return centrality_data, gnn_centrality, viral_cent

def visualize_centrality_comparison(centrality_data, gnn_centrality, viral_centrality, usernameList):
    """Visualize and compare different centrality measures."""
    # Correlation matrix
    centrality_measures = np.column_stack([
        centrality_data['Degree'],
        centrality_data['In-Degree'],
        centrality_data['Out-Degree'],
        centrality_data['Eigenvector'],
        gnn_centrality,
        viral_centrality
    ])

    measure_names = ['Degree', 'In-Degree', 'Out-Degree', 'Eigenvector', 'GNN', 'Viral']
    corr_matrix = np.corrcoef(centrality_measures.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=measure_names, yticklabels=measure_names)
    plt.title('Correlation Between Centrality Measures')
    plt.tight_layout()
    plt.savefig('centrality_correlation.png')
    plt.close()

    # Scatter plot: GNN vs Viral Centrality
    plt.figure(figsize=(12, 8))
    plt.scatter(viral_centrality, gnn_centrality, alpha=0.6)

    # Highlight top nodes
    top_indices = np.argsort(viral_centrality)[-10:]
    for idx in top_indices:
        plt.annotate(usernameList[idx],
                     (viral_centrality[idx], gnn_centrality[idx]),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Viral Centrality')
    plt.ylabel('GNN Centrality')
    plt.title('GNN Centrality vs. Viral Centrality')
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = np.corrcoef(viral_centrality, gnn_centrality)[0, 1]
    plt.annotate(f'Correlation: {correlation:.4f}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('gnn_vs_viral_centrality.png')
    plt.close()

    # Return top influencers by different measures
    top_by_viral = np.argsort(viral_centrality)[-10:][::-1]
    top_by_gnn = np.argsort(gnn_centrality)[-10:][::-1]

    print("\nTop 10 Influencers by Viral Centrality:")
    for i, idx in enumerate(top_by_viral):
        print(f"{i+1}. {usernameList[idx]} (Score: {viral_centrality[idx]:.4f})")

    print("\nTop 10 Influencers by GNN Centrality:")
    for i, idx in enumerate(top_by_gnn):
        print(f"{i+1}. {usernameList[idx]} (Score: {gnn_centrality[idx]:.4f})")

    return top_by_viral, top_by_gnn

# =====================================================================
# Community Detection
# =====================================================================

class CommunityGAT(nn.Module):
    """Graph Attention Network for community detection."""
    def __init__(self, in_channels, hidden_channels, embedding_dim, heads=8):
        super(CommunityGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=0.3)
        self.conv3 = GATConv(hidden_channels, embedding_dim, heads=1, dropout=0.3)

    def forward(self, x, edge_index, edge_attr=None):
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # Final embedding layer
        x = self.conv3(x, edge_index)

        return x

def detect_communities(data, G, usernameList, n_clusters=5, embedding_dim=16):
    """Detect communities using GAT embeddings and K-means clustering."""
    # Initialize and train the GAT model
    model = CommunityGAT(in_channels=data.x.size(1),
                         hidden_channels=64,
                         embedding_dim=embedding_dim).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train with a simple reconstruction loss
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()

        # Get node embeddings
        embeddings = model(data.x, data.edge_index)

        # Compute pairwise similarities
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Create target adjacency matrix (with edge weights)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        adj_matrix = torch.zeros((data.num_nodes, data.num_nodes), device=device)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_matrix[src, dst] = edge_attr[i, 0]  # Use tensor indexing and extract the scalar value

        # Compute loss (reconstruction of weighted adjacency)
        loss = F.mse_loss(sim_matrix, adj_matrix)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')

    # Get embeddings for clustering
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Evaluate clustering quality
    silhouette_avg = silhouette_score(embeddings, clusters)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    # Visualize communities
    visualize_communities(G, clusters, usernameList, embeddings, n_clusters)

    return clusters, embeddings

def visualize_communities(G, clusters, usernameList, embeddings, n_clusters):
    """Visualize the detected communities."""
    # Add community information to the graph
    for i, comm in enumerate(clusters):
        G.nodes[i]['community'] = int(comm)

    # Create a spring layout
    pos = nx.spring_layout(G, seed=42)

    # Plot the graph with communities
    plt.figure(figsize=(16, 12))

    # Define colors for communities
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    # Draw nodes colored by community
    for i in range(n_clusters):
        node_list = [node for node in G.nodes() if G.nodes[node]['community'] == i]
        nx.draw_networkx_nodes(G, pos,
                              nodelist=node_list,
                              node_color=[colors[i]],
                              node_size=100,
                              alpha=0.8,
                              label=f'Community {i+1}')

    # Draw edges with transparency based on weight
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color=edge_colors, edge_cmap=plt.cm.Blues)

    # Add labels for important nodes
    # Find top nodes by degree in each community
    top_nodes = []
    for comm in range(n_clusters):
        comm_nodes = [node for node in G.nodes() if G.nodes[node]['community'] == comm]
        comm_nodes_by_degree = sorted(comm_nodes, key=lambda x: G.degree(x), reverse=True)
        top_nodes.extend(comm_nodes_by_degree[:3])  # Top 3 nodes from each community

    # Create labels dictionary for top nodes
    labels = {node: usernameList[node] for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title('Congress Network Communities Detected by GAT')
    plt.legend(scatterpoints=1, loc='lower right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('community_detection.png', dpi=300)
    plt.close()

    # Visualize embeddings with t-SNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[i]], label=f'Community {i+1}', alpha=0.7)

    # Add labels for important nodes
    for node in top_nodes:
        plt.annotate(usernameList[node],
                    (embeddings_2d[node, 0], embeddings_2d[node, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9)

    plt.title('t-SNE Visualization of Node Embeddings')
    plt.legend()
    plt.tight_layout()
    plt.savefig('community_embeddings.png', dpi=300)
    plt.close()

    # Analyze community characteristics
    community_stats = []
    for comm in range(n_clusters):
        comm_nodes = [node for node in G.nodes() if G.nodes[node]['community'] == comm]

        # Calculate average degree and other metrics
        avg_degree = np.mean([G.degree(node) for node in comm_nodes])
        avg_in_degree = np.mean([G.in_degree(node) for node in comm_nodes])
        avg_out_degree = np.mean([G.out_degree(node) for node in comm_nodes])

        # Get top nodes by degree
        top_nodes_by_degree = sorted(comm_nodes, key=lambda x: G.degree(x), reverse=True)[:5]
        top_usernames = [usernameList[node] for node in top_nodes_by_degree]

        community_stats.append({
            'Community': comm + 1,
            'Size': len(comm_nodes),
            'Avg Degree': avg_degree,
            'Avg In-Degree': avg_in_degree,
            'Avg Out-Degree': avg_out_degree,
            'Top Members': top_usernames
        })

    # Print community statistics
    print("\nCommunity Statistics:")
    for stats in community_stats:
        print(f"\nCommunity {stats['Community']}:")
        print(f"  Size: {stats['Size']} members")
        print(f"  Avg Degree: {stats['Avg Degree']:.2f}")
        print(f"  Avg In-Degree: {stats['Avg In-Degree']:.2f}")
        print(f"  Avg Out-Degree: {stats['Avg Out-Degree']:.2f}")
        print(f"  Top Members: {', '.join(stats['Top Members'])}")

    return community_stats

# =====================================================================
# Main Function
# =====================================================================

def main():
    """Main function to run all analyses."""
    print("Loading congress network data...")
    inList, inWeight, outList, outWeight, usernameList = load_congress_data()

    print("Creating NetworkX graph...")
    G = create_networkx_graph(inList, inWeight, outList, outWeight, usernameList)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    print("Converting to PyTorch Geometric format...")
    data = create_pyg_data(G)

    # Task 1: Influence Modeling
    print("\n=== Task 1: Influence Modeling ===")
    print("Computing viral centrality as ground truth...")
    viral_cent = viral_centrality(inList, inWeight, outList, Niter=-1, tol=0.001)

    print("Training influence prediction model...")
    influence_model, predicted_influence = train_influence_model(data, viral_cent)

    print("Visualizing influence prediction results...")
    visualize_influence_prediction(viral_cent, predicted_influence, usernameList)

    # Task 2: Centrality Analysis
    print("\n=== Task 2: Centrality Analysis ===")
    print("Computing various centrality measures...")
    centrality_data, gnn_centrality, viral_cent = compute_centrality_measures(G, data, usernameList)

    print("Comparing centrality measures...")
    top_viral, top_gnn = visualize_centrality_comparison(centrality_data, gnn_centrality, viral_cent, usernameList)

    # Task 3: Community Detection
    print("\n=== Task 3: Community Detection ===")
    print("Detecting communities using Graph Attention Network...")
    clusters, embeddings = detect_communities(data, G, usernameList)

    print("\nAnalysis complete! Results saved as PNG files.")

if __name__ == "__main__":
    main()
