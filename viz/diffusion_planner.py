import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DModel
from typing import List, Dict, Any
import numpy as np

class SimpleTrajectoryDiffusion:
    def __init__(self, action_dim=4, timesteps=50):
        self.scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        self.unet = UNet2DModel(
            sample_size=10,  # Sequence length
            in_channels=action_dim,
            out_channels=action_dim,
            block_out_channels=(32, 64),
            layers_per_block=2,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet.to(self.device)
        self.scheduler.set_timesteps(timesteps)

    def generate_trajectory(self, start_pos: tuple, goal_pos: tuple, num_steps: int = 10, temperature: float = 1.0, principles: List[str] = None) -> List[Dict[str, Any]]:
        # Scale noise with temperature for creativity
        batch_size = 1
        noise_scale = temperature
        noise = torch.randn((batch_size, 4, num_steps, 1)).to(self.device) * noise_scale
        timesteps = self.scheduler.timesteps.to(self.device)

        for t in timesteps:
            # Predict noise
            noise_pred = self.unet(noise, t).sample
            # Denoise step
            noise = self.scheduler.step(noise_pred, t, noise).prev_sample

        # Convert to actions (simple heuristic for POC)
        trajectory = []
        current = list(start_pos)
        for i in range(num_steps):
            action_probs = torch.softmax(noise[0, :, i, 0], dim=0).cpu().numpy()
            action = np.argmax(action_probs)
            if action == 0: current[0] += 1  # right
            elif action == 1: current[0] -= 1  # left
            elif action == 2: current[1] += 1  # down
            elif action == 3: current[1] -= 1  # up
            step = {"type": "move", "pos": tuple(current), "dxdy": (action % 2 * 2 - 1, action // 2 * 2 - 1), "prob": float(action_probs[action])}
            trajectory.append(step)

        # Constitution conditioning: Filter out violating steps (simple check)
        if principles:
            filtered_traj = []
            for step in trajectory:
                # Placeholder: Reject if step goes to 'danger' (3,3) and 'do_no_harm' principle
                if principles and 'do_no_harm' in principles and step['pos'] == (3, 3):
                    continue  # Skip violating step
                filtered_traj.append(step)
            trajectory = filtered_traj[:num_steps]  # Pad or truncate if needed

        return trajectory

# Example
if __name__ == "__main__":
    planner = SimpleTrajectoryDiffusion()
    traj = planner.generate_trajectory((0, 0), (7, 7), num_steps=10, temperature=1.5, principles=['do_no_harm'])
    print("Advanced Trajectory (with creativity & conditioning):", traj)
