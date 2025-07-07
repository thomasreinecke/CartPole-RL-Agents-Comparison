# src/agents/agent_dueling_ddqn.py

import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
from .base_agent import BaseAgent

# Defines the Dueling Q-Network architecture.
class DuelingQNetwork(nn.Module):
    """
    A Dueling Network architecture that separates the estimation of
    state values (V(s)) and action advantages (A(s,a)).
    """
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # A shared layer to learn common state features.
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )
        
        # The "Value Stream" learns the value of the state, V(s).
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single value.
        )
        
        # The "Advantage Stream" learns the advantage of each action, A(s,a).
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size) # Outputs a value for each action.
        )

    def forward(self, state):
        """Combines the value and advantage streams to get final Q-values."""
        # Process the state through the shared feature layer.
        features = self.feature_layer(state)
        # Calculate the state-value and action-advantages.
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Recombine V(s) and A(s,a) to get the final Q-values.
        # Subtracting the mean of advantages ensures identifiability and stability.
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# Defines a fixed-size buffer to store experience tuples.
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Adds a new experience to the buffer.
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # Samples a random batch of experiences from the buffer.
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # Returns the current size of the buffer.
        return len(self.buffer)

# The Dueling Double DQN agent class.
class DuelingDDQNAgent(BaseAgent):
    def __init__(self, state_size, action_space, **kwargs):
        super().__init__(state_size, action_space)
        
        # Agent hyperparameters, allowing overrides from kwargs.
        replay_buffer_capacity = kwargs.get('replay_buffer_capacity', 10000)
        self.batch_size = kwargs.get('batch_size', 128)
        self.update_every = kwargs.get('update_every', 4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.999)
        self.min_epsilon = kwargs.get('min_epsilon', 0.01)
        learning_rate = kwargs.get('learning_rate', 5e-4)
        self.tau = kwargs.get('tau', 1e-3)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)

        # Set the computation device (e.g., MPS for Apple Silicon, or CPU).
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"DuelingDDQNAgent is using device: {self.device}")

        # Instantiate the DuelingQNetwork for both policy and target networks.
        self.policy_net = DuelingQNetwork(self.state_size, self.action_space.n).to(self.device)
        self.target_net = DuelingQNetwork(self.state_size, self.action_space.n).to(self.device)
        
        # Initialize the target network with the same weights as the policy network.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Set target network to evaluation mode.

        # Instantiate the replay buffer, optimizer, and loss function.
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Counter to track steps for periodic updates.
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Stores an experience and triggers learning if it's time.
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Learn every 'update_every' time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.replay_buffer) > self.batch_size:
            self._learn()

        # Decay epsilon at the end of each episode.
        if done:
            self._decay_epsilon()

    def choose_action(self, state):
        # Selects an action using the epsilon-greedy policy.
        if random.random() < self.epsilon:
            return self.action_space.sample() # Explore: take a random action.
        else:
            # Exploit: take the best known action.
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def _learn(self):
        # Internal method to update the policy network.
        # Sample a batch of transitions from the replay buffer.
        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert data to PyTorch tensors.
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().unsqueeze(1).to(self.device)

        # Get Q-values for the current state-action pairs from the policy network.
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Calculate the target Q-values using the Double DQN logic.
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            next_q_values[dones] = 0.0
        
        # Calculate the Bellman target.
        target_q_values = rewards + (self.gamma * next_q_values)
        
        # Compute the loss.
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Perform the optimization step.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._soft_update_target_network()

    def _soft_update_target_network(self):
        # Performs a soft update of the target network's weights.
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def _decay_epsilon(self):
        # Decreases the exploration rate.
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def get_status(self):
        # Returns the agent's current epsilon value for logging.
        return {"Epsilon": self.epsilon}
        
    def save_model(self, filepath, verbose=True):
        # Saves the policy network's parameters to a file.
        os.makedirs(filepath.parent, exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        if verbose:
            print(f"Dueling DDQN model saved to {filepath}")

    def load_model(self, filepath):
        # Loads parameters into both the policy and target networks.
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Dueling DDQN model loaded from {filepath}")