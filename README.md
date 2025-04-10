# Analyzing Political Influence and Communication Patterns in Social Networks using GNN 

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
