
# CartPole-RL-Agents-Comparison

This repository contains a Python-based framework for a comparative analysis of 8 different model-free Reinforcement Learning (RL) agents. All agents are trained and evaluated on the classic `CartPole-v1` environment from the Gymnasium library.

The primary goal of this project is to provide a unified testing ground to observe the practical performance, stability, and learning dynamics of various algorithms, from foundational methods to state-of-the-art approaches.

## Model-Free Reinforcement Learning

All agents implemented in this project are **model-free**. This means they learn a policy or a value function directly from interaction with the environment through trial-and-error rewards. They do not try to learn the underlying "physics" or "rules" of the CartPole environment itself (e.g., they don't learn to predict the next state).

## Implemented Agents

The agents are organized into the two primary families of model-free RL:

#### Value-Based Agents
These agents learn to estimate the value of being in a certain state or taking a certain action.
- **`QLearningAgent`**: The foundational tabular method using a discretized state space.
- **`DQNAgent`**: A basic Deep Q-Network that uses a neural network to approximate Q-values.
- **`DQNTargetAgent`**: An enhancement to DQN that introduces a separate target-net to stabilize learning.
- **`DDQNAgent`**: An evolution of DQN that uses the Double Q-Learning algorithm to reduce value overestimation.
- **`DuelingDDQNAgent`**: A further enhancement that uses a Dueling Network architecture for more efficient value estimation.

#### Policy-Gradient Agents
These agents learn a policy directly, mapping states to action probabilities.
- **`ReinforceAgent`**: The most fundamental policy-gradient algorithm, which learns from complete episodes.
- **`A2CAgent`**: An Advantage Actor-Critic agent that introduces a critic to reduce variance and stabilize learning.
- **`PPOAgent`**: A state-of-the-art Proximal Policy Optimization agent that uses a clipped objective for highly stable and efficient training.

---

## Getting Started

This project uses `make` to simplify setup and execution.

### 1. Environment Setup

To get started, create the local Python virtual environment and install all the required packages from `requirements.txt`.

From the project's root directory, run:
```bash
make install
```
This command will create a `.venv` directory and install libraries like PyTorch, Gymnasium, and Plotly.

### 2. Training an Agent

You can train any of the implemented agents using the `make train` command. The training progress will be logged to the console and all performance data will be saved to the `performance.db` SQLite database file. Models are saved after every episode to the `models/` directory.

To train a specific agent, you must pass the `AGENT` variable to the command.

**Usage Examples:**
```bash
make train AGENT=qlearning
make train AGENT=dqn 
make train AGENT=dqn_target
make train AGENT=ddqn
make train AGENT=dueling_ddqn
make train AGENT=reinforce
make train AGENT=a2c
make train AGENT=ppo
```

### 3. Running a Trained Agent

After an agent has been trained, you can run a visual demonstration of its learned policy.

**Usage Examples:**
```bash
make run AGENT=qlearning
make run AGENT=dqn 
make run AGENT=dqn_target
make run AGENT=ddqn
make run AGENT=dueling_ddqn
make run AGENT=reinforce
make run AGENT=a2c
make run AGENT=ppo
```

This will open a Gymnasium window showing the agent attempting to solve the CartPole environment.

### 4. Evaluating Performance

This project includes a Jupyter Notebook, `evaluation.ipynb`, for visualizing the training data stored in `performance.db`.

**How to use:**
1.  Open the `evaluation.ipynb` file in a compatible editor like VS Code or a standalone Jupyter Lab/Notebook instance.
2.  **Select the Kernel:** In the top-right corner of the notebook, make sure you select the Python interpreter from your project's `.venv` to ensure all libraries are available.
3.  **Choose an Agent:** Modify the `AGENT_TO_ANALYZE` variable at the top of the notebook to select which agent(s) you want to plot.
4.  **Run the Cells:** Execute the cells to load the data from `performance.db` and generate the performance charts.# CartPole-RL-Agents-Comparison
