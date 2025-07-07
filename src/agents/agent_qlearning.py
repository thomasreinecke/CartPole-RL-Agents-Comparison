# src/agents/agent_qlearning.py

import numpy as np
import os
from .base_agent import BaseAgent

# A classic Q-Learning agent that uses a discretized state space.
class QLearningAgent(BaseAgent):
    """A classic Q-Learning agent that implements the BaseAgent interface."""
    def __init__(self, state_size, action_space, **kwargs):
        # Initialize the parent class.
        super().__init__(state_size, action_space)
        
        # Agent hyperparameters, allowing overrides from kwargs.
        self.lr = kwargs.get('learning_rate', 0.1)
        self.gamma = kwargs.get('discount_factor', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.999)
        self.min_epsilon = kwargs.get('min_epsilon', 0.01)

        # Defines the boundaries for discretizing the continuous state space.
        self.state_bins = [
            np.linspace(-2.4, 2.4, 9),      # Cart Position
            np.linspace(-3.0, 3.0, 9),      # Cart Velocity
            np.linspace(-0.209, 0.209, 9),  # Pole Angle
            np.linspace(-3.0, 3.0, 9),      # Pole Angular Velocity
        ]
        
        # Calculates the dimensions of the Q-table based on the number of bins.
        q_table_size = [len(bins) + 1 for bins in self.state_bins] + [self.action_space.n]
        # Initializes the Q-table with all zeros.
        self.q_table = np.zeros(q_table_size)

    def _discretize_state(self, state):
        """Internal helper to convert a continuous state to a discrete one."""
        # Finds the correct bin index for each component of the state vector.
        return tuple(np.digitize(s, self.state_bins[i]) for i, s in enumerate(state))

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        # The agent must discretize the continuous state received from the environment.
        discretized_state = self._discretize_state(state)
        
        if np.random.random() < self.epsilon:
            return self.action_space.sample() # Explore: take a random action.
        else:
            return np.argmax(self.q_table[discretized_state]) # Exploit: take the best known action.

    def step(self, state, action, reward, next_state, done):
        """
        Performs the Q-learning update for a single environment step.
        """
        # Discretize the current and next states to find their positions in the Q-table.
        discretized_state = self._discretize_state(state)
        next_discretized_state = self._discretize_state(next_state)
        
        # Retrieve the current Q-value and the maximum Q-value for the next state.
        old_value = self.q_table[discretized_state][action]
        next_max = np.max(self.q_table[next_discretized_state])
        
        # Apply the Bellman equation to calculate the new Q-value.
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[discretized_state][action] = new_value

        # Decay epsilon at the end of each episode.
        if done:
            self._decay_epsilon()

    def _decay_epsilon(self):
        """Internal helper to decay the exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def get_status(self):
        """Returns the agent's current epsilon value for logging."""
        return {"Epsilon": self.epsilon}

    def save_model(self, filepath, verbose=True):
        """Saves the Q-table to a .npy file."""
        os.makedirs(filepath.parent, exist_ok=True)
        np.save(filepath, self.q_table)
        if verbose:
            print(f"Q-table saved to {filepath}")

    def load_model(self, filepath):
        """Loads the Q-table from a .npy file."""
        self.q_table = np.load(filepath)
        print(f"Q-table loaded from {filepath}")