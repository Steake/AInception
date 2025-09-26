import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Any, Tuple
import numpy as np

class PromiseGraphTransformer(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embeddings for promises/principles/agents
        self.node_embed = nn.Embedding(100, hidden_dim)  # Assuming <100 unique nodes
        
        # Transformer layers for graph propagation
        self.transformer_layers = nn.ModuleList([
            TransformerConv(hidden_dim, hidden_dim, heads=num_heads, concat=True, beta=True)
            for _ in range(num_layers)
        ])
        
        # Output heads for cooperation prediction and anomaly detection
        self.cooperation_head = nn.Linear(hidden_dim * num_heads, 1)  # Sigmoid for probability
        self.anomaly_head = nn.Linear(hidden_dim * num_heads, 1)  # Sigmoid for anomaly score

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # x: node features [num_nodes, hidden_dim]
        # edge_index: [2, num_edges]
        
        # Embed nodes if x is indices
        if x.dtype == torch.long:
            x = self.node_embed(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling if batch provided (for graph-level prediction)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Predictions
        cooperation = torch.sigmoid(self.cooperation_head(x))
        anomaly = torch.sigmoid(self.anomaly_head(x))
        
        return {
            'cooperation_prob': cooperation,
            'anomaly_score': anomaly,
            'embeddings': x
        }

    def simulate_social_dynamics(self, graph_data: Dict[str, Any], num_steps: int = 5) -> Dict[str, Any]:
        """
        Simulate multi-agent promise interactions over time steps.
        graph_data: {'nodes': list[node_ids], 'edges': list[(src, dst)], 'features': np.array}
        Returns: {'cooperation_history': list[float], 'emergent_events': list[str]}
        """
        # Convert to PyG Data
        x = torch.tensor(graph_data['features'], dtype=torch.float)
        edge_index = torch.tensor(graph_data['edges'], dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        
        # Batch if multiple graphs (for multi-agent)
        batch = Batch.from_data_list([data] * num_steps) if num_steps > 1 else data
        
        # Forward pass for each step (simple simulation)
        cooperation_history = []
        emergent_events = []
        
        for step in range(num_steps):
            out = self.forward(batch.x, batch.edge_index, batch.batch if hasattr(batch, 'batch') else None)
            coop_step = out['cooperation_prob'].mean().item()
            coop_anom = out['anomaly_score'].mean().item()
            
            cooperation_history.append(coop_step)
            
            # Detect emergent events (heuristic on anomaly)
            if coop_anom > 0.7:
                event = f"Step {step}: High cooperation anomaly detected (score: {coop_anom:.2f})"
            elif coop_step > 0.8:
                event = f"Step {step}: Strong alliance formed (cooperation: {coop_step:.2f})"
            else:
                event = f"Step {step}: Normal interaction (cooperation: {coop_step:.2f})"
            emergent_events.append(event)
            
            # Update features (simple evolution for POC)
            batch.x = F.relu(batch.x + out['embeddings'] * 0.1)  # Incremental update
            
        return {
            'cooperation_history': cooperation_history,
            'emergent_events': emergent_events,
            'final_embeddings': out['embeddings']
        }

# Example usage and dummy training data
if __name__ == "__main__":
    model = PromiseGraphTransformer(hidden_dim=32, num_layers=2)
    
    # Dummy graph: 3 nodes (agents), edges representing promise relationships
    num_nodes = 3
    x = torch.arange(num_nodes, dtype=torch.long)  # Node indices
    edge_index = torch.tensor([[0,1],[1,0],[1,2],[2,1],[0,2]], dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    
    # Simulate
    out = model(data.x, data.edge_index)
    print("Graph Embeddings Shape:", out['embeddings'].shape)
    print("Sample Cooperation Prob:", out['cooperation_prob'][0].item())
    print("Sample Anomaly Score:", out['anomaly_score'][0].item())
    
    # Simulate dynamics
    graph_data = {
        'nodes': [0,1,2],
        'edges': [[0,1],[1,0],[1,2],[2,1],[0,2]],
        'features': np.random.rand(3, 32).astype(np.float32)
    }
    dynamics = model.simulate_social_dynamics(graph_data, num_steps=3)
    print("Social Dynamics Simulation:", dynamics)
