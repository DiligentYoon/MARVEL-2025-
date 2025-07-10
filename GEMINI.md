# Gemini Analysis Summary for MARVEL Project

This document summarizes the key architectural insights and workflow of the MARVEL project, as determined through collaborative analysis. It serves as a quick context refresher for future sessions.

## 1. Core Architecture: CTDE

The project is built on a **Centralized Training, Decentralized Execution (CTDE)** paradigm.

-   **Centralized Training**: A single, master process (`driver.py`) aggregates experiences from all agents and updates one central set of neural networks (`PolicyNet`, `QNet1`, `QNet2`). This allows for stable and efficient learning.
-   **Decentralized Execution**: During simulation, each agent acts autonomously based on its own local observations. `RLRunner` (Ray actors) manage the simulation and collect experiences.

## 2. Key Concepts & Mechanisms

### Parameter Sharing
All agents in a simulation share the exact same neural network instances (`PolicyNet`).

-   **Evidence**: `driver.py` initializes the networks. The `PolicyNet` weights are then distributed to all `RLRunner` instances, which in turn pass the same network reference to every `Agent` they manage. `QNet` instances are used centrally in `driver.py` for training.
-   **Benefit**: This approach is highly scalable and promotes generalization, as the experience of every agent contributes to the learning of a single, robust policy.

### Continuous Action Space
The project utilizes a continuous action space for agent control.

-   **Mechanism**: Instead of discrete waypoints and headings, the `agent.py:get_action` function outputs `velocity_scale` and `yaw_rate_scale`. These values are then scaled by `MAX_VELOCITY` and `MAX_YAW_RATE` (defined in `parameter.py`) to determine the agent's movement.

### Overlap Reward
A novel reward mechanism is introduced to encourage diverse exploration among agents.

-   **Mechanism**: The `agent.py:calculate_overlap_reward` function computes a reward based on the unique sensed area of each agent, penalizing overlapping sensing regions. This promotes agents exploring different parts of the environment.

### Training Algorithms (`TRAIN_ALGO`)
The `TRAIN_ALGO` parameter in `parameter.py` dictates the specific training algorithm and the structure of observations used during training.

-   **`TRAIN_ALGO = 0`**: Standard SAC (Soft Actor-Critic).
-   **`TRAIN_ALGO = 1`**: Multi-Agent Actor-Critic (MAAC), incorporating information about other agents' states.
-   **`TRAIN_ALGO = 2`**: Ground Truth (GT) observations, using full environment information for training.
-   **`TRAIN_ALGO = 3`**: Combines MAAC and Ground Truth observations.

## 3. Module Breakdown & Workflow

The overall workflow follows the CTDE model, orchestrated by the `driver`.

### Workflow Diagram

```mermaid
graph TD;
    %% Node Declarations
    nodeDriverLoop("Main Training Loop");
    nodeReplayBuffer{"Experience Buffer"};
    nodeNetworks["Policy & Q-Networks"];
    nodeTrainingFunc["Training Function (SAC)"];
    nodeRLRunner("RLRunner (Ray Actor)");
    nodeMultiAgentWorker("MultiAgentWorker");
    nodeAgentInstance("Agent");
    nodeEnv("Environment (Env)");
    nodeParameter("Parameter Configuration");

    %% Subgraph Grouping
    subgraph Driver [driver.py]
        nodeDriverLoop;
        nodeReplayBuffer;
        nodeNetworks;
        nodeTrainingFunc;
    end;

    subgraph RemoteExecution [Ray Actors]
        nodeRLRunner;
        nodeMultiAgentWorker;
    end;

    subgraph AgentSimulation [Simulation Components]
        nodeAgentInstance;
        nodeEnv;
    end;

    nodeParameter;

    %% Links
    nodeDriverLoop -- "Initializes & Manages" --> nodeNetworks;
    nodeDriverLoop -- "Manages" --> nodeReplayBuffer;
    nodeDriverLoop -- "Distributes Weights to" --> nodeRLRunner;
    nodeDriverLoop -- "Collects Experiences from" --> nodeRLRunner;
    nodeDriverLoop -- "Performs" --> nodeTrainingFunc;
    nodeTrainingFunc -- "Updates" --> nodeNetworks;
    nodeReplayBuffer -- "Feeds" --> nodeTrainingFunc;

    nodeRLRunner -- "Manages" --> nodeMultiAgentWorker;
    nodeRLRunner -- "Receives Weights from" --> nodeDriverLoop;
    nodeRLRunner -- "Sends Experiences to" --> nodeDriverLoop;

    nodeMultiAgentWorker -- "Manages Multiple" --> nodeAgentInstance;
    nodeMultiAgentWorker -- "Interacts with" --> nodeEnv;
    nodeMultiAgentWorker -- "Collects Experiences" --> nodeRLRunner;

    nodeAgentInstance -- "Uses" --> nodeNetworks;
    nodeAgentInstance -- "Acts on" --> nodeEnv;
    nodeAgentInstance -- "Calculates" --> nodeAgentInstance; %% Self-loop for overlap reward

    nodeEnv -- "Provides Observations & Rewards" --> nodeMultiAgentWorker;

    nodeParameter -- "Configures" --> nodeDriverLoop;
    nodeParameter -- "Configures" --> nodeRLRunner;
    nodeParameter -- "Configures" --> nodeMultiAgentWorker;
    nodeParameter -- "Configures" --> nodeAgentInstance;
    nodeParameter -- "Configures" --> nodeEnv;

    %% Styles
    style nodeNetworks fill:#f9f,stroke:#333,stroke-width:2px;
    style nodeRLRunner fill:#ccf,stroke:#333,stroke-width:2px;
    style nodeMultiAgentWorker fill:#cfc,stroke:#333,stroke-width:2px;
    style nodeAgentInstance fill:#ffc,stroke:#333,stroke-width:2px;
    style nodeEnv fill:#cff,stroke:#333,stroke-width:2px;
    style nodeParameter fill:#fcc,stroke:#333,stroke-width:2px;
```

### Execution & Training Flow

1.  **Configuration (`parameter.py`)**: All simulation and training parameters are defined in `parameter.py`.
2.  **Distribution (`driver.py`)**: The `driver` initializes `PolicyNet`, `QNet1`, and `QNet2`. It then distributes the current `PolicyNet` weights to all `RLRunner` actors.
3.  **Execution (`RLRunner` -> `MultiAgentWorker` -> `Agent`)**: Each `RLRunner` executes a simulation episode. Inside, a `MultiAgentWorker` manages the environment and multiple `Agent` instances. Each `Agent` uses the shared `PolicyNet` to decide its continuous actions (velocity and yaw rate) based on its local observation. Agents also calculate an `overlap_reward` to encourage diverse exploration.
4.  **Collection (`RLRunner` -> `driver.py`)**: The `RLRunner` collects the `(observation, action, reward, next_observation, done)` trajectories from all its agents. The structure of `observation` and `next_observation` can vary based on the `TRAIN_ALGO` parameter, potentially including ground truth or multi-agent information. These experiences are then sent back to the `driver`.
5.  **Aggregation (`driver.py`)**: The `driver` stores these experiences in a central `experience_buffer` (replay buffer).
6.  **Training (`driver.py`)**: The `driver` samples a mini-batch from the `experience_buffer` and updates the central `PolicyNet`, `QNet1`, `QNet2`, and `log_alpha` using the Soft Actor-Critic (SAC) algorithm. The cycle then repeats.
7.  **Logging & Checkpointing (`driver.py`)**: Training metrics are logged to TensorBoard and WandB. Model checkpoints are periodically saved.