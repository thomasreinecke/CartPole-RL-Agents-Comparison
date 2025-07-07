# src/agents/agent_ppo.py

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
        # Apply custom weight initialization.
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

# The PPO agent, which uses a clipped objective for stable policy updates.
class PPOAgent(BaseAgent):
    def __init__(self, state_size, action_space, **kwargs):
        super().__init__(state_size, action_space)

        # Agent hyperparameters, allowing overrides from kwargs.
        self.n_steps = kwargs.get('n_steps', 256)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.gamma = kwargs.get('gamma', 0.99)
        self.lambda_gae = kwargs.get('lambda_gae', 0.95)
        learning_rate = kwargs.get('learning_rate', 2.5e-4)
        self.entropy_coefficient = kwargs.get('entropy_coefficient', 0.01)
        self.critic_loss_coefficient = kwargs.get('critic_loss_coeff', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)

        # Set the computation device (e.g., MPS for Apple Silicon, or CPU).
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"PPO Agent is using device: {self.device}")

        # Instantiate the network and optimizer.
        self.ac_network = ActorCriticNetwork(self.state_size, self.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=learning_rate)
        
        # Temporary storage for experiences collected during an episode.
        self.trajectory = []
        # Store the latest losses for logging purposes.
        self.latest_actor_loss = 0.0
        self.latest_critic_loss = 0.0

    def start_episode(self):
        """Hook for main.py to clear memory at the start of an episode."""
        self.trajectory.clear()

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

    def finish_episode(self):
        # Triggers a learning update with the collected data at the end of an episode.
        if not self.trajectory:
            return
        self._learn()

    def _learn(self):
        # Internal method to perform a PPO learning update.
        # Unpacks the trajectory data into separate lists.
        states, actions, rewards, next_states, dones = zip(*self.trajectory)
        
        # Converts lists of data into PyTorch tensors.
        states_t = torch.from_numpy(np.array(states)).float().to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        
        # Gets value estimates for calculating advantages.
        with torch.no_grad():
            _, old_values = self.ac_network(states_t)
            _, next_values = self.ac_network(torch.from_numpy(np.array(next_states)).float().to(self.device))
            old_values = old_values.squeeze(-1)
            next_values = next_values.squeeze(-1)

        # Calculates Generalized Advantage Estimation (GAE).
        advantages = torch.zeros(len(rewards), device=self.device)
        last_gae_lambda = 0
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + self.gamma * next_values[t] * non_terminal - old_values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_gae * non_terminal * last_gae_lambda
        
        # Calculates the returns (targets for the critic).
        returns = advantages + old_values
        
        # Normalizes advantages for training stability.
        std_adv = advantages.std()
        if std_adv > 1e-8:
            advantages = (advantages - advantages.mean()) / std_adv
        
        # Gets the log probabilities of actions from the old policy.
        with torch.no_grad():
            old_logits, _ = self.ac_network(states_t)
            old_dist = Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions_t)

        # Performs multiple epochs of updates on the collected data.
        for _ in range(self.n_epochs):
            logits, values_pred = self.ac_network(states_t)
            values_pred = values_pred.squeeze(-1)
            dist = Categorical(logits=logits)
            
            # Calculates the ratio of new to old policy probabilities.
            new_log_probs = dist.log_prob(actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
            # Calculates the clipped surrogate objective for the actor loss.
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Calculates the clipped value loss for the critic.
            loss_unclipped = (values_pred - returns).pow(2)
            values_pred_clipped = old_values + torch.clamp(values_pred - old_values, -self.clip_epsilon, self.clip_epsilon)
            loss_clipped = (values_pred_clipped - returns).pow(2)
            critic_loss = torch.max(loss_unclipped, loss_clipped).mean()
            
            # Calculates the entropy bonus to encourage exploration.
            entropy_loss = dist.entropy().mean()

            # Combines the losses into a single total loss value.
            total_loss = actor_loss + (self.critic_loss_coefficient * critic_loss) - (self.entropy_coefficient * entropy_loss)
            
            # Performs the optimization step.
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
            self.optimizer.step()

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
            print(f"PPO model saved to {filepath}")

    def load_model(self, filepath):
        # Loads the network's parameters from a file.
        self.ac_network.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"PPO model loaded from {filepath}")