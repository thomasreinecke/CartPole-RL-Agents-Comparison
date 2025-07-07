# src/main.py

# Standard library and third-party imports.
import gymnasium as gym
import time
import collections
import argparse
from pathlib import Path
import numpy as np
import sqlite3
import json

# --- Agent Imports ---
# Imports all agent classes from the centralized 'agents' package.
from agents import (
    QLearningAgent, DQNAgent, DQNTargetAgent, DDQNAgent, DuelingDDQNAgent,
    ReinforceAgent, A2CAgent, PPOAgent
)
# Imports the UI wrapper for visual feedback during demonstrations.
from environment_wrapper import RenderInfoWrapper


# --- Path Configuration ---
# Creates robust, OS-agnostic paths for models and the database.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = PROJECT_ROOT / "performance.db"
# Ensures the directory for saving models exists.
MODELS_DIR.mkdir(exist_ok=True)


# --- AGENT CONFIGURATION ---
# A centralized dictionary that defines all available agents.
# This makes the framework easily extensible: add a new agent here to use it.
AGENT_CONFIG = {
    "qlearning": {"class": QLearningAgent, "file_ext": "npy"},
    "dqn": {"class": DQNAgent, "file_ext": "pth"},
    "dqn_target": {"class": DQNTargetAgent, "file_ext": "pth"},
    "ddqn": {"class": DDQNAgent, "file_ext": "pth"},
    "dueling_ddqn": {"class": DuelingDDQNAgent, "file_ext": "pth"},
    "reinforce": {"class": ReinforceAgent, "file_ext": "pth"},
    "a2c": {"class": A2CAgent, "file_ext": "pth"},
    "ppo": {"class": PPOAgent, "file_ext": "pth"}
}


# --- TRAINING CONFIGURATION ---
# The maximum number of episodes to run the training for.
MAX_EPISODES = 10000
# The target score that defines a "perfect" run and the solve condition.
TARGET_SCORE = 1000
# The window for logging and checking the solve condition.
LOG_AND_SOLVE_WINDOW = 10


# --- Performance Logger Class ---
class PerformanceLogger:
    """Handles all SQLite database operations for logging training performance."""
    def __init__(self, db_path):
        # Establishes a connection to the SQLite database file.
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        # Ensures the required tables exist upon initialization.
        self._create_tables()

    def _create_tables(self):
        """Creates the database tables if they don't already exist."""
        # The 'runtime' table tracks high-level information about each training run.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS runtime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT
            )
        ''')
        # The 'training' table logs performance metrics at each logging interval.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS training (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                agent TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                duration REAL NOT NULL,
                average_score REAL NOT NULL,
                metrics_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runtime(id)
            )
        ''')
        self.conn.commit()

    def start_training_session(self, agent_name):
        """Deletes old data for the agent and starts a new session."""
        print(f"Clearing previous DB records for agent: {agent_name}")
        # Clears old data to ensure a fresh start for the new training run.
        self.cursor.execute("DELETE FROM training WHERE agent = ?", (agent_name,))
        # Creates a new record for the current run.
        self.cursor.execute("INSERT INTO runtime (agent, status) VALUES (?, ?)", (agent_name, 'running'))
        self.conn.commit()
        # Returns the unique ID for the new run.
        return self.cursor.lastrowid

    def log_training_step(self, run_id, agent_name, epoch, duration, avg_score, metrics_dict):
        """Logs a single training interval to the database."""
        # Converts the agent-specific metrics dictionary to a JSON string for storage.
        metrics_json = json.dumps(metrics_dict)
        # Inserts a new row with the performance data for the current interval.
        self.cursor.execute('''
            INSERT INTO training (run_id, agent, epoch, duration, average_score, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (run_id, agent_name, epoch, duration, avg_score, metrics_json))
        self.conn.commit()

    def end_training_session(self, run_id, status):
        """Updates the runtime record with an end time and status."""
        self.cursor.execute(
            "UPDATE runtime SET end_time = CURRENT_TIMESTAMP, status = ? WHERE id = ?",
            (status, run_id)
        )
        self.conn.commit()

    def close(self):
        # Closes the database connection cleanly.
        self.conn.close()


def get_model_path(agent_name):
    """Generates the model file path based on the agent's configuration."""
    ext = AGENT_CONFIG[agent_name]["file_ext"]
    return MODELS_DIR / f"cartpole_{agent_name}.{ext}"


def train(agent_name):
    """A generic training loop with a maximum episode limit."""
    print(f"--- Starting Training for {agent_name.upper()} ---")
    
    # Initialize the database logger.
    logger = PerformanceLogger(DB_PATH)
    run_id = logger.start_training_session(agent_name)
    
    # Create the gym environment with a maximum step limit per episode.
    env = gym.make("CartPole-v1", max_episode_steps=TARGET_SCORE)
    
    # Instantiate the correct agent class based on the configuration.
    model_path = get_model_path(agent_name)
    agent_class = AGENT_CONFIG[agent_name]["class"]
    state_size = env.observation_space.shape[0]
    agent = agent_class(state_size=state_size, action_space=env.action_space)

    # Setup data structures for tracking scores and time.
    scores = collections.deque(maxlen=LOG_AND_SOLVE_WINDOW)
    training_start_time = time.time()
    
    # Set a default status in case the training is interrupted.
    training_status = 'max_episodes_reached'

    try:
        # Main training loop, runs for a maximum number of episodes.
        for episode in range(1, MAX_EPISODES + 1):
            state, info = env.reset()
            steps = 0

            # Optional hook for agents that need to reset their state at the start of an episode.
            if hasattr(agent, 'start_episode'):
                agent.start_episode()

            # Inner loop for a single episode.
            while True:
                steps += 1
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # The agent processes the experience from the step.
                agent.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            
            # Optional hook for agents that perform updates at the end of an episode.
            if hasattr(agent, 'finish_episode'):
                agent.finish_episode()

            # Save the model silently after every episode.
            agent.save_model(model_path, verbose=False)
            scores.append(steps)
            
            # Log progress to the console and database at regular intervals.
            if episode % LOG_AND_SOLVE_WINDOW == 0:
                total_duration = time.time() - training_start_time
                average_score = np.mean(scores)
                agent_status = agent.get_status()
                status_str = " | ".join([f"{k}: {v:.4f}" for k, v in agent_status.items()])
                
                print(
                    f"Ep {episode:<5}/{MAX_EPISODES} | "
                    f"Duration: {total_duration:7.2f}s | "
                    f"Avg Score (last {LOG_AND_SOLVE_WINDOW}): {average_score:6.2f} | "
                    f"{status_str}"
                )
                
                logger.log_training_step(run_id, agent_name, episode, total_duration, average_score, agent_status)
                
                # Check if the environment has been solved.
                if average_score >= TARGET_SCORE:
                    print(f"\n--- Environment SOLVED in {episode} episodes! ---")
                    training_status = 'solved'
                    break
        else: 
            # This block executes if the for loop completes without breaking.
            print(f"\n--- Training finished after reaching MAX_EPISODES ({MAX_EPISODES}) ---")

    except KeyboardInterrupt:
        # Handles user interruption (Ctrl+C).
        print("\n--- Training interrupted by user. ---")
        training_status = 'interrupted'
    finally:
        # Cleanup block that always runs, regardless of how the loop ended.
        print("\n--- Final Save ---")
        agent.save_model(model_path, verbose=True)
        logger.end_training_session(run_id, training_status)
        logger.close()
        env.close()


def run_trained(agent_name):
    """A generic function to run and visualize any trained agent."""
    print(f"--- Running Trained {agent_name.upper()} Agent ---")
    
    # Creates the environment with rendering enabled.
    env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=TARGET_SCORE)
    env = RenderInfoWrapper(env)

    # Loads the appropriate agent and its saved model.
    model_path = get_model_path(agent_name)
    agent_class = AGENT_CONFIG[agent_name]["class"]
    state_size = env.observation_space.shape[0]
    agent = agent_class(state_size=state_size, action_space=env.action_space)

    try:
        agent.load_model(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model not found at {model_path}. Please train the agent first using 'make train AGENT={agent_name}'.")
        return

    # Turn off exploration for evaluation to see the learned policy's true performance.
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0

    try:
        # Loop to run episodes indefinitely until the user closes the window.
        while True:
            state, info = env.reset()
            while True:
                env.render()
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                if terminated or truncated:
                    time.sleep(0.5)
                    break
    except (KeyboardInterrupt, Exception):
        print(f"\nWindow closed or error occurred. Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    # Sets up command-line argument parsing to select the mode and agent.
    parser = argparse.ArgumentParser(description="Train or Run a Reinforcement Learning Agent on CartPole.")
    parser.add_argument("mode", choices=["train", "run"], help="Mode of operation.")
    parser.add_argument("--agent", choices=list(AGENT_CONFIG.keys()), default="qlearning", help="The agent to use.")
    args = parser.parse_args()

    # Executes the appropriate function based on command-line arguments.
    if args.agent not in AGENT_CONFIG:
        print(f"Error: Agent '{args.agent}' not found in AGENT_CONFIG.")
    elif args.mode == "train":
        train(args.agent)
    elif args.mode == "run":
        run_trained(args.agent)