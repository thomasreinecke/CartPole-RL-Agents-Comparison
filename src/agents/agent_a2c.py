# src/agents/agent_a2c.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from .base_agent import BaseAgent

# Defines the neural network architecture for the Actor and Critic.
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # Shared layers for processing state features.
        self.body = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        # Actor head predicts the policy (action probabilities).
        self.actor_head = nn.Linear(128, action_size)
        # Critic head predicts the state-value.
        self.critic_head = nn.Linear(128, 1)
        # Apply custom weight initialization to the network.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Applies orthogonal initialization to linear layers for training stability.
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        # Defines the forward pass through the network.
        features = self.body(state)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value

# The A2C agent that learns to solve the environment.
class A2CAgent(BaseAgent):
    def __init__(self, state_size, action_space, **kwargs):
        super().__init__(state_size, action_space)

        # Agent hyperparameters, allowing overrides from kwargs.
        learning_rate = kwargs.get('learning_rate', 2.5e-4)
        self.n_steps = kwargs.get('n_steps', 20)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambda_gae = kwargs.get('lambda_gae', 0.95)
        self.entropy_coefficient = kwargs.get('entropy_coefficient', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.critic_loss_coefficient = kwargs.get('critic_loss_coeff', 0.5)

        # Set the computation device (e.g., MPS for Apple Silicon, or CPU).
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"A2C Agent is using device: {self.device}")

        # Instantiate the network and optimizer.
        self.ac_network = ActorCriticNetwork(self.state_size, self.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Temporary storage for experiences collected between learning updates.
        self.trajectory = []
        # Store the latest losses for logging purposes.
        self.latest_actor_loss = 0.0
        self.latest_critic_loss = 0.0

    def choose_action(self, state):
        # Converts state to a tensor and gets action probabilities from the network.
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, _ = self.ac_network(state_tensor)
        
        # Creates a categorical distribution and samples an action.
        dist = Categorical(logits=action_logits)
        return dist.sample().item()

    def step(self, state, action, reward, next_state, done):
        # Appends a single step of experience to the trajectory buffer.
        self.trajectory.append((state, action, reward, next_state, done))
        # If the buffer is full, trigger a learning update.
        if len(self.trajectory) >= self.n_steps:
            self._learn()

    def finish_episode(self):
        # Triggers a learning update with any remaining data at the end of an episode.
        if self.trajectory:
            self._learn()
            
    def start_episode(self):
        """Hook for main.py to clear memory at the start of an episode."""
        self.trajectory.clear()

    def _learn(self):
        # Internal method to perform a learning update.
        # Unpacks the trajectory data into separate lists.
        states, actions, rewards, next_states, dones = zip(*self.trajectory)
        
        # Converts lists of data into PyTorch tensors.
        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states_t = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Gets value estimates from the critic for current and next states.
        with torch.no_grad():
            _, values = self.ac_network(states_t)
            _, next_values = self.ac_network(next_states_t)
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)

        # Calculates Generalized Advantage Estimation (GAE) backwards in time.
        advantages = torch.zeros_like(rewards_t)
        last_gae_lambda = 0
        for t in reversed(range(len(rewards_t))):
            non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_values[t] * non_terminal - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_gae * non_terminal * last_gae_lambda
        
        # Calculates the returns (targets for the critic).
        returns = advantages + values
        
        # Normalizes advantages for training stability.
        std_adv = advantages.std()
        if std_adv > 1e-8:
            advantages = (advantages - advantages.mean()) / std_adv

        # Gets new predictions from the network for loss calculation.
        logits, values_pred = self.ac_network(states_t)
        values_pred = values_pred.squeeze(-1)
        dist = Categorical(logits=logits)
        
        # Calculates the individual loss components.
        actor_loss = -(dist.log_prob(actions_t) * advantages.detach()).mean()
        critic_loss = nn.functional.mse_loss(values_pred, returns)
        entropy_loss = dist.entropy().mean()

        # Combines the losses into a single total loss value.
        total_loss = actor_loss + (self.critic_loss_coefficient * critic_loss) - (self.entropy_coefficient * entropy_loss)
        
        # Performs the optimization step.
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Clears the buffer for the next batch of experiences.
        self.trajectory.clear()
        
        # Stores the latest loss values for external logging.
        self.latest_actor_loss = actor_loss.item()
        self.latest_critic_loss = critic_loss.item()

    def get_status(self):
        # Returns a dictionary of the agent's current learning metrics.
        return {"Actor Loss": self.latest_actor_loss, "Critic Loss": self.latest_critic_loss}

    def save_model(self, filepath, verbose=True):
        # Saves the network's parameters to a file.
        os.makedirs(filepath.parent, exist_ok=True)
        torch.save(self.ac_network.state_dict(), filepath)
        if verbose:
            print(f"A2C model saved to {filepath}")

    def load_model(self, filepath):
        # Loads the network's parameters from a file.
        self.ac_network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"A2C model loaded from {filepath}")