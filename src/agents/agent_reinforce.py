# src/agents/agent_reinforce.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .base_agent import BaseAgent

# Defines the neural network architecture that represents the policy.
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

# The REINFORCE agent, which learns via Monte Carlo Policy Gradients.
class ReinforceAgent(BaseAgent):
    def __init__(self, state_size, action_space, **kwargs):
        super().__init__(state_size, action_space)
        
        # Agent hyperparameters, allowing overrides from kwargs.
        self.gamma = kwargs.get('gamma', 0.99)
        learning_rate = kwargs.get('learning_rate', 0.001)

        # Set the computation device (e.g., MPS for Apple Silicon, or CPU).
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"ReinforceAgent is using device: {self.device}")

        # Instantiate the policy network and optimizer.
        self.policy_net = PolicyNetwork(self.state_size, self.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Memory to store data for the current episode.
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        
        # Attributes to store the latest status for external logging.
        self.latest_policy_loss = 0.0
        self.latest_avg_entropy = 0.0

    def start_episode(self):
        """A helper method to clear memory at the start of a new episode."""
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()

    def choose_action(self, state):
        """Chooses an action by sampling from the policy distribution."""
        # Converts state to a tensor and gets action probabilities from the network.
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        logits = self.policy_net(state_tensor)
        
        # Creates a categorical distribution and samples an action.
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        
        # Store the log probability and entropy for the learning update.
        self.log_probs.append(action_dist.log_prob(action))
        self.entropies.append(action_dist.entropy())
        
        return action.item()

    def step(self, state, action, reward, next_state, done):
        """For REINFORCE, the step method's only job is to record the reward."""
        self.rewards.append(reward)

    def finish_episode(self):
        """
        Called by the training loop at the end of an episode to perform the
        learning update for the entire sequence of events.
        """
        # Calculate discounted returns (G_t) by iterating backwards through the rewards.
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns for training stability.
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Calculate the policy loss for each step.
        policy_loss = []
        for log_prob, R_t in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R_t)

        # Perform the optimization step.
        self.optimizer.zero_grad()
        # Sum the losses for all steps into a single value and backpropagate.
        policy_loss_tensor = torch.stack(policy_loss).sum()
        policy_loss_tensor.backward()
        self.optimizer.step()

        # Store status metrics for logging.
        self.latest_policy_loss = policy_loss_tensor.item()
        self.latest_avg_entropy = torch.stack(self.entropies).mean().item()
        
    def get_status(self):
        """Returns the policy loss and entropy from the last learning update."""
        return {
            "Policy Loss": self.latest_policy_loss,
            "Avg Entropy": self.latest_avg_entropy
        }
        
    def save_model(self, filepath, verbose=True):
        """Saves the policy network's state dictionary."""
        os.makedirs(filepath.parent, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        if verbose:
            print(f"REINFORCE model saved to {filepath}")

    def load_model(self, filepath):
        """Loads the policy network's state dictionary."""
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"REINFORCE model loaded from {filepath}")