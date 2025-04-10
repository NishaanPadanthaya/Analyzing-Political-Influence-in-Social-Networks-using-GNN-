# Congressional Network Analysis with Graph Neural Networks

## Project Overview

This project applies advanced Graph Neural Network (GNN) techniques to analyze the Congressional Twitter network, providing insights into political communication patterns, influence dynamics, and community structures. The implementation addresses three key analytical tasks:

1. **Influence Modeling**: Using GNNs to predict which Congressional members amplify messages most effectively
2. **Centrality Analysis**: Comparing GNN-based node importance scores with traditional and viral centrality measures
3. **Community Detection**: Identifying political factions and ideological groups using Graph Attention Networks

## Dataset

The project uses the Congressional Twitter network dataset from:
- "A centrality measure for quantifying spread on weighted, directed networks" (Fink et. al., Physica A, 2023)
- "A Congressional Twitter network dataset quantifying pairwise probability of influence" (Fink et. al., Data in Brief, 2023)

The dataset contains:
- **Nodes**: Congressional Twitter accounts
- **Edges**: Interactions between accounts (retweets, mentions, replies)
- **Edge Weights**: Transmission probabilities representing influence strength

## Learning Goals

This implementation addresses several learning goals:

1. **Understanding Influence Propagation**:
   - Discover the dynamics and direction of political messaging
   - Identify important influencers within the political discourse
   - Analyze which influencers may not be active on Twitter platforms

2. **Examining Partisan Structures**:
   - Investigate engagement patterns between political actors
   - Analyze community detection within the party network
   - Examine the scope of engagement within and across ideological divides

3. **Information Spread Modeling**:
   - Implement propagation models for predicting message spread
   - Apply GNN techniques to model information diffusion
   - Analyze the effectiveness of different network structures for information dissemination

## Implementation Details

### Influence Modeling

The implementation uses a Graph Convolutional Network (GCN) to predict influence spread:
- **Input Features**: Node network properties and edge weights (transmission probabilities)
- **Target**: Viral centrality scores as ground truth for influence
- **Architecture**: 3-layer GCN with ReLU activations and dropout
- **Training**: Mean Squared Error loss with Adam optimizer
- **Evaluation**: RMSE and correlation with viral centrality

This model helps identify which Congressional members are most effective at amplifying messages through the network.

### Centrality Analysis

The implementation compares multiple centrality measures:
- **Traditional Measures**: Degree, In-degree, Out-degree, Eigenvector centrality
- **GNN-based Centrality**: Learned through a specialized GNN model
- **Viral Centrality**: Based on simulating information spread through the network

The comparison reveals different aspects of node importance and provides a more nuanced understanding of influence in the political network.

### Community Detection

The implementation uses a Graph Attention Network (GAT) for community detection:
- **Node Embeddings**: Learned through multi-head attention mechanisms
- **Clustering**: K-means applied to embeddings to identify communities
- **Visualization**: Network visualization with community coloring and t-SNE plots
- **Analysis**: Community characteristics, key members, and inter-community interactions

This analysis helps identify political factions, ideological groups, and cross-party interactions in the Congressional network.

## Requirements

The implementation requires the following Python packages:

```
numpy
matplotlib
networkx
scikit-learn
torch
torch-geometric
torch-scatter
seaborn
```

## Installation

### For Python Script

To install the required dependencies for running the Python script:

```python
# Basic dependencies
pip install numpy matplotlib networkx scikit-learn seaborn

# PyTorch (CPU version)
pip install torch torchvision

# PyTorch Geometric and related packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### For Jupyter Notebook

To install the required dependencies in a Jupyter notebook:

```python
# Basic dependencies
!pip install numpy matplotlib networkx scikit-learn seaborn

# PyTorch (CPU version)
!pip install torch torchvision

# PyTorch Geometric and related packages
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

For GPU support, modify the PyTorch Geometric installation to match your PyTorch and CUDA version:

```python
# For GPU support (example with PyTorch 2.0.0 and CUDA 11.8)
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Note: You may need to restart the Jupyter kernel after installation to use the newly installed packages.

## Usage

### Running the Python Script

To run the full analysis:

```
python congress_network_gnn.py
```

### Using in a Jupyter Notebook

To use in a Jupyter notebook:

1. Install dependencies (see Installation section)
2. Import required libraries
3. Copy the functions from `congress_network_gnn.py` into notebook cells
4. Execute the cells in sequence

## Output

The implementation generates several visualizations:

1. **Influence Prediction**:
   - Scatter plot of predicted influence vs. viral centrality
   - Highlighted top influencers
   - Correlation statistics

2. **Centrality Comparison**:
   - Correlation matrix of different centrality measures
   - Scatter plot comparing GNN centrality to viral centrality
   - Lists of top influencers according to different measures

3. **Community Detection**:
   - Network visualization with detected communities
   - t-SNE visualization of node embeddings
   - Community statistics and key members

## Troubleshooting

### Common Issues

1. **Disconnected Graph Error**:
   - The implementation handles disconnected graphs by using PageRank as an alternative to eigenvector centrality.
   - If you encounter other graph-related errors, consider preprocessing the graph to ensure connectivity.

2. **CUDA Out of Memory**:
   - If you encounter GPU memory issues, try reducing batch sizes or model complexity.
   - Alternatively, switch to CPU computation by setting `device = torch.device('cpu')`.

3. **Package Installation Issues**:
   - PyTorch Geometric requires specific versions that match your PyTorch and CUDA versions.
   - Refer to the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for detailed instructions.

4. **Jupyter Notebook Installation**:
   - When installing in a Jupyter notebook, use the `!pip install` command with the exclamation mark.
   - You may need to restart the kernel after installation to use the newly installed packages.

## Project Structure

```
congress_network/
├── congress_network_data.json    # Network dataset
├── congress.edgelist             # NetworkX format edgelist
├── viral_centrality.py           # Implementation of viral centrality
├── congress_network_gnn.py       # Main implementation file
├── README.md                     # This file
└── output/                       # Generated visualizations
    ├── influence_prediction.png
    ├── centrality_correlation.png
    ├── gnn_vs_viral_centrality.png
    ├── community_detection.png
    └── community_embeddings.png
```

## Practical Applications

This implementation enables several practical applications:

1. **Identifying Key Influencers**: Finding accounts that can effectively spread information
2. **Understanding Political Polarization**: Analyzing community structure and cross-party interactions
3. **Predicting Information Spread**: Modeling how messages propagate through the political network
4. **Strategic Communication**: Identifying optimal paths for message dissemination

## Future Work

Potential extensions to this project include:

1. **Temporal Analysis**: Incorporating time dynamics to track evolving trends
2. **Content Analysis**: Integrating text data from tweets to analyze message content
3. **Cross-Platform Analysis**: Extending to other social media platforms
4. **Predictive Modeling**: Forecasting viral content and information cascades
