# --- Makefile (Complete and Generalized) ---

# ==============================================================================
#  Variables
# ==============================================================================

# Virtual environment directory
VENV = .venv

# Python interpreter from the virtual environment
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# The target file that represents a successful installation.
# If this file exists and requirements.txt hasn't changed, the install step is skipped.
INSTALL_TARGET = $(VENV)/bin/activate

# --- CONFIGURATION ---
# The default agent to use if none is specified via the command line.
# Example: `make train` will use 'dqn'. `make train AGENT=qlearning` will use 'qlearning'.
AGENT ?= reinforce


# ==============================================================================
#  Standard Commands (install, clean, test)
# ==============================================================================

# Use .PHONY to declare targets that are not files.
# This ensures 'make' will always run the command for these targets.
.PHONY: all install clean test train run

# The default target that runs when you just type 'make'.
all: install

# The 'install' command is now just a user-friendly alias for our real target.
install: $(INSTALL_TARGET)

# This is the core installation logic. This rule only runs if $(INSTALL_TARGET) does not exist
# or if requirements.txt is newer than the target file.
$(INSTALL_TARGET): requirements.txt
	@echo "--- Environment not found or requirements changed. Installing... ---"
	@rm -rf $(VENV)
	@echo "1/4: Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "2/4: Installing certifi for SSL workaround..."
	@$(PIP) install certifi --trusted-host pypi.org --trusted-host files.pythonhosted.org > /dev/null
	@CERT_PATH="$$( $(PYTHON) -m certifi )"; \
	echo "3/4: Upgrading pip..."; \
	$(PIP) install --upgrade pip --cert="$$CERT_PATH" --trusted-host pypi.org --trusted-host files.pythonhosted.org > /dev/null; \
	echo "4/4: Installing project requirements..."; \
	$(PIP) install -r requirements.txt --cert="$$CERT_PATH" --trusted-host pypi.org --trusted-host files.pythonhosted.org
	@echo "\n--- Installation complete! ---"
	# We 'touch' the file to update its timestamp, marking the installation as complete.
	@touch $(INSTALL_TARGET)

# Deletes the virtual environment, Python caches, and saved models.
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV) models/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleanup complete."

# Runs tests using pytest (assuming tests are in a 'tests/' directory).
test: $(INSTALL_TARGET)
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/


# ==============================================================================
#  Project-Specific Commands (train, run)
# ==============================================================================

# Train the specified agent.
# Usage:
#   make train              (trains the default agent, '$(AGENT)')
#   make train AGENT=dqn    (explicitly trains the DQN agent)
#   make train AGENT=qlearning (explicitly trains the Q-Learning agent)
train: $(INSTALL_TARGET)
	@echo "--- Starting training for agent: $(AGENT) ---"
	$(PYTHON) src/main.py train --agent $(AGENT)

# Run the pre-trained agent with visualization.
# Usage:
#   make run                (runs the default agent, '$(AGENT)')
#   make run AGENT=dqn      (explicitly runs the DQN agent)
#   make run AGENT=qlearning   (explicitly runs the Q-Learning agent)
run: $(INSTALL_TARGET)
	@echo "--- Running trained agent: $(AGENT) ---"
	$(PYTHON) src/main.py run --agent $(AGENT)