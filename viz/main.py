import sys
import os
import asyncio
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, 
                             QHBoxLayout, QLabel, QSlider, QProgressBar)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter import AInceptionAdapter, AgentVisualState
from agent.core import Agent
from llm_module import LLMDecomposer
from diffusion_planner import SimpleTrajectoryDiffusion
from multimodal import MultimodalPerception
from rlhf_module import RLHFTrainer
import gymnasium as gym
import tempfile
import numpy as np

class VisualWidget(QWidget):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.agent_pos = (0, 0)
        self.carrying = False
        self.grid_size = 8
        self.setMinimumSize(400, 400)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_from_adapter)
        self.timer.start(100)  # Update every 100ms

    def update_from_adapter(self):
        state = self.adapter.get_state()
        if state:
            self.agent_pos = state.position
            self.carrying = state.carrying
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(200, 230, 230))  # Background

        # Draw grid
        cell_size = min(self.width() // self.grid_size, self.height() // self.grid_size)
        painter.setPen(QColor(100, 100, 100))
        for i in range(self.grid_size + 1):
            x = i * cell_size
            painter.drawLine(x, 0, x, self.height())
            painter.drawLine(0, x, self.width(), x)

        # Draw agent
        x, y = self.agent_pos
        agent_color = QColor(0, 255, 0) if self.carrying else QColor(255, 0, 0)
        painter.setBrush(agent_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(x * cell_size + 2, y * cell_size + 2, cell_size - 4, cell_size - 4)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AInceptionViz - Basic GUI")
        self.setGeometry(100, 100, 800, 600)

        # Initialize agent and adapter
        temp_db_path = tempfile.NamedTemporaryFile(delete=False, suffix='.db').name
        self.agent = Agent(enable_journal_llm=False, db_path=temp_db_path)
        self.adapter = AInceptionAdapter(self.agent)

        # ML Components
        self.llm = LLMDecomposer()
        self.diffusion = SimpleTrajectoryDiffusion()
        self.multimodal = MultimodalPerception()
        self.rlhf = RLHFTrainer()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.visual_widget = VisualWidget(self.adapter)
        layout.addWidget(self.visual_widget)

        # Overlays in main layout
        self.energy_bar = QProgressBar()
        self.energy_bar.setRange(0, 100)
        self.energy_bar.setValue(70)
        self.energy_bar.setStyleSheet("QProgressBar { border: 2px solid gray; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #05B8CC; }")
        layout.addWidget(QLabel("Energy:"))
        layout.addWidget(self.energy_bar)

        self.thought_bubble = QLabel("Agent thinking...")
        self.thought_bubble.setStyleSheet("background-color: white; border: 1px solid black; padding: 5px; border-radius: 10px;")
        layout.addWidget(self.thought_bubble)

        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        self.step_btn = QPushButton("Step")
        self.traj_btn = QPushButton("Generate Trajectory")
        self.decompose_btn = QPushButton("Decompose Goal")
        self.describe_btn = QPushButton("Describe State")
        self.train_rlhf_btn = QPushButton("Train RLHF")
        self.policy_action_btn = QPushButton("Get Policy Action")
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.step_btn)
        controls_layout.addWidget(self.traj_btn)
        controls_layout.addWidget(self.decompose_btn)
        controls_layout.addWidget(self.describe_btn)
        controls_layout.addWidget(self.train_rlhf_btn)
        controls_layout.addWidget(self.policy_action_btn)
        layout.addLayout(controls_layout)

        # Creativity slider
        self.creativity_slider = QSlider(Qt.Orientation.Horizontal)
        self.creativity_slider.setMinimum(5)
        self.creativity_slider.setMaximum(20)
        self.creativity_slider.setValue(10)
        self.creativity_slider.setTickInterval(5)
        self.creativity_label = QLabel("Creativity: 1.0")
        controls_layout.addWidget(QLabel("Creativity:"))
        controls_layout.addWidget(self.creativity_slider)
        controls_layout.addWidget(self.creativity_label)
        self.creativity_slider.valueChanged.connect(self.update_creativity_label)

        # Connect buttons
        self.play_btn.clicked.connect(self.play_simulation)
        self.pause_btn.clicked.connect(self.pause_simulation)
        self.reset_btn.clicked.connect(self.reset_simulation)
        self.step_btn.clicked.connect(self.single_step)
        self.traj_btn.clicked.connect(self.generate_trajectory)
        self.decompose_btn.clicked.connect(self.test_llm_decomposition)
        self.describe_btn.clicked.connect(self.describe_current_state)
        self.train_rlhf_btn.clicked.connect(self.train_rlhf)
        self.policy_action_btn.clicked.connect(self.get_policy_action)

        self.is_playing = False
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.simulation_step)
        self.simulation_timer.setInterval(500)

        self.trajectory = []
        self.traj_index = 0
        self.traj_timer = QTimer()
        self.traj_timer.timeout.connect(self.animate_trajectory)
        self.traj_timer.setInterval(200)

        self.current_temperature = 1.0

        # Update timer for overlays
        self.overlay_timer = QTimer()
        self.overlay_timer.timeout.connect(self.update_overlays)
        self.overlay_timer.start(100)

    def update_creativity_label(self, value):
        self.current_temperature = value / 10.0
        self.creativity_label.setText(f"Creativity: {self.current_temperature:.1f}")

    def update_overlays(self):
        # Update energy bar
        try:
            energy = self.agent.drives.drives['energy'].current
            self.energy_bar.setValue(int(energy * 100))
        except (AttributeError, KeyError):
            self.energy_bar.setValue(70)

        # Update thought bubble
        try:
            narrative = self.llm.generate_narrative({'energy': self.energy_bar.value() / 100.0}, "observing environment")
            self.thought_bubble.setText(narrative[:100] + "..." if len(narrative) > 100 else narrative)
        except:
            self.thought_bubble.setText("Agent thinking...")

    def play_simulation(self):
        if not self.is_playing:
            self.is_playing = True
            self.simulation_timer.start()
        else:
            self.simulation_timer.stop()
            self.is_playing = False

    def pause_simulation(self):
        self.simulation_timer.stop()
        self.is_playing = False

    def reset_simulation(self):
        self.simulation_timer.stop()
        self.is_playing = False
        self.traj_timer.stop()
        # Reset agent
        self.agent.drives.set_drive('energy', 0.7)
        self.visual_widget.agent_pos = (0, 0)
        self.visual_widget.carrying = False
        self.visual_widget.update()
        self.energy_bar.setValue(70)
        self.thought_bubble.setText("Agent reset and ready.")

    def single_step(self):
        obs = self._get_current_obs()
        asyncio.run(self.adapter.step(obs))

    def simulation_step(self):
        obs = self._get_current_obs()
        asyncio.run(self.adapter.step(obs))

    def _get_current_obs(self):
        try:
            energy = self.agent.drives.drives['energy'].current
        except (AttributeError, KeyError):
            energy = 0.7
        return {
            'agent_pos': self.visual_widget.agent_pos,
            'goal': (7, 7),
            'energy': energy,
            'carrying': self.visual_widget.carrying
        }

    def generate_trajectory(self):
        current_pos = self.visual_widget.agent_pos
        traj = self.diffusion.generate_trajectory(
            current_pos, (7, 7), num_steps=10, 
            temperature=self.current_temperature,
            principles=['do_no_harm']
        )
        self.trajectory = traj
        self.traj_index = 0
        print(f"Generated trajectory with temperature {self.current_temperature}:", traj)
        self.traj_timer.start()

    def animate_trajectory(self):
        if self.traj_index < len(self.trajectory):
            step = self.trajectory[self.traj_index]
            self.visual_widget.agent_pos = step['pos']
            self.visual_widget.update()
            self.traj_index += 1
        else:
            self.traj_timer.stop()

    def test_llm_decomposition(self):
        goal = "Deliver the item to the target location while conserving energy"
        subgoals = self.llm.decompose_goal(goal)
        print("LLM Goal Decomposition:")
        print(subgoals)

    def describe_current_state(self):
        state = {
            'agent_pos': self.visual_widget.agent_pos,
            'carrying': self.visual_widget.carrying,
            'goal': (7, 7),
            'grid_size': 8,
            'energy': self.energy_bar.value() / 100.0
        }
        description = self.multimodal.describe_state(state)
        print("Multimodal State Description:")
        print(description)
        
        enhanced = self.llm.generate_narrative(state, "observing environment")
        print("LLM Enhanced Narrative:")
        print(enhanced)

    def train_rlhf(self):
        # Sample feedback data
        feedback_data = [
            (np.array([0.1, 0.2, 0.0, 0.0]), 0, 1.0, 0.8),  # Good: left
            (np.array([0.5, 0.6, 0.0, 0.0]), 1, -1.0, -0.5),  # Bad: right
            (np.array([0.0, 0.0, 0.1, 0.2]), 0, 1.0, 0.9)   # Good: left
        ]
        self.rlhf.train_with_feedback(100, feedback_data)
        print("RLHF Training Complete - Model updated with feedback!")

    def get_policy_action(self):
        # Get current state as observation (simplified)
        obs = np.array([self.visual_widget.agent_pos[0]/8.0, self.visual_widget.agent_pos[1]/8.0, 
                       self.visual_widget.carrying, self.energy_bar.value() / 100.0])
        action = self.rlhf.get_policy_action(obs)
        print("RLHF Policy Action for current state:", action)
        # Apply action to agent (simplified)
        self.single_step_with_policy(action)

    def single_step_with_policy(self, action):
        obs = self._get_current_obs()
        asyncio.run(self.adapter.step(obs))
        print("Applied RLHF policy action:", action)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())