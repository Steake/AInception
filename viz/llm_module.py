from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class LLMDecomposer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):  # Placeholder small model; replace with Llama 3 7B when ready
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)

    def decompose_goal(self, goal_description: str, state_context: str = "") -> str:
        prompt = f"""Decompose the following goal into sub-goals for an autonomous agent.

Goal: {goal_description}

Current state: {state_context}

Sub-goals:"""
        response = self.pipe(prompt, max_length=200, num_return_sequences=1, temperature=0.7, do_sample=True)
        return response[0]['generated_text']

    def generate_narrative(self, state: dict, action: str) -> str:
        prompt = f"""Generate a narrative description for the agent's action.

State: {state}
Action: {action}

Narrative:"""
        response = self.pipe(prompt, max_length=100, num_return_sequences=1, temperature=0.8)
        return response[0]['generated_text']

# Example usage
if __name__ == "__main__":
    decomposer = LLMDecomposer()
    subgoals = decomposer.decompose_goal("Deliver the item to the target location")
    print(subgoals)
