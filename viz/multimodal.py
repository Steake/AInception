import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import Dict, Any, List
import json

class MultimodalPerception:
    def __init__(self):
        # Fallback: No CLIP loading, use rule-based visual description
        print("Using fallback visual encoder (CLIP unavailable due to network).")
        
        # Sentence transformer for text embeddings (for RAG)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # FAISS index for RAG (vector store)
        self.dimension = 384  # MiniLM dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []  # Store descriptions or states

        # Sample data for RAG (placeholder - load from DB in production)
        self._add_sample_data()

    def _add_sample_data(self):
        # Sample state descriptions for RAG
        samples = [
            "Agent at (0,0) with low energy, item nearby",
            "Agent carrying item, heading to goal (7,7)",
            "High temperature zone detected, energy low",
            "Promise violation risk at danger tile (3,3)"
        ]
        embeddings = self.text_encoder.encode(samples)
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(samples)

    def encode_visual_state(self, grid_state: Dict[str, Any]) -> np.ndarray:
        """
        Fallback rule-based visual encoding: Generate a simple feature vector from state.
        Returns a dummy embedding for compatibility.
        """
        # Simple rule-based features: position, carrying, energy as vector
        ax, ay = grid_state.get('agent_pos', (0, 0))
        carrying = 1.0 if grid_state.get('carrying', False) else 0.0
        energy = grid_state.get('energy', 0.5)
        goal_dist = abs(ax - 7) + abs(ay - 7)  # Manhattan to goal
        
        # Dummy embedding (384-dim for MiniLM compatibility)
        embedding = np.zeros(384)
        embedding[0] = ax / 8.0  # Normalized x
        embedding[1] = ay / 8.0  # Normalized y
        embedding[2] = carrying
        embedding[3] = energy
        embedding[4] = goal_dist / 14.0  # Max Manhattan in 8x8 grid
        return embedding

    def rag_retrieve(self, query_embedding: np.ndarray, k: int = 3) -> List[str]:
        """Retrieve similar states using FAISS for RAG."""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.metadata[i] for i in indices[0]]

    def describe_state(self, visual_state: Dict[str, Any], text_context: str = "") -> str:
        """Fallback visual description + text RAG for state description."""
        # Generate rule-based visual description
        ax, ay = visual_state.get('agent_pos', (0, 0))
        carrying = visual_state.get('carrying', False)
        energy = visual_state.get('energy', 0.5)
        goal = visual_state.get('goal', (7, 7))
        dist_to_goal = abs(ax - goal[0]) + abs(ay - goal[1])
        
        visual_desc = f"Agent at position ({ax}, {ay}), carrying item: {carrying}, energy level: {energy:.2f}, distance to goal: {dist_to_goal}"
        
        # For RAG, use text description of state as query
        state_text = visual_desc
        if text_context:
            state_text += f". Context: {text_context}"
        
        text_emb = self.text_encoder.encode([state_text])
        retrieved = self.rag_retrieve(text_emb, k=2)
        
        description = f"Rule-based visual analysis: {visual_desc}. Retrieved similar states: {', '.join(retrieved[:1])}"
        return description

# Example usage
if __name__ == "__main__":
    multimodal = MultimodalPerception()
    state = {'agent_pos': (3, 3), 'item_pos': (1, 1), 'goal': (7, 7), 'grid_size': 8, 'energy': 0.6}
    desc = multimodal.describe_state(state)
    print("Fallback State Description:", desc)
