# src/agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """An abstract base class for all reinforcement learning agents."""

    def __init__(self, state_size, action_space, **kwargs):
        self.state_size = state_size
        self.action_space = action_space

    @abstractmethod
    def choose_action(self, state):
        """Given the current state, choose an action."""
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Process a single step of interaction from the environment."""
        pass
        
    # --- NEW METHOD ---
    @abstractmethod
    def get_status(self):
        """
        Returns a dictionary of agent-specific status information for logging.
        This allows the main loop to be agnostic about what is being logged.
        
        Example: {"Epsilon": 0.5, "Actor Loss": 0.123}
        
        Returns:
            dict: A dictionary of status metrics.
        """
        pass

    @abstractmethod
    def save_model(self, filepath, verbose=True):
        """
        Saves the agent's learned model/parameters to a file.
        
        Args:
            filepath (str or Path): The path to save the model to.
            verbose (bool): If True, prints a confirmation message.
        """
        pass

    @abstractmethod
    def load_model(self, filepath):
        """Loads the agent's learned model/parameters."""
        pass