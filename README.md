# sim — teaching a car to drive itself, with no ML framework

A 2D driving simulator where a car learns to complete a track by trial and error.
Two reinforcement learning agents are implemented **from scratch** — tabular
Q-learning and a deep Q-network — with no PyTorch, no TensorFlow, no Gym. The
only dependencies are `numpy` and `pygame`.

Trained weights are committed, so you can watch a trained car drive immediately
without training anything yourself.

```bash
pip install -r requirements.txt
python run_trained.py
```

## Why this exists

Reinforcement learning is easy to *use* and hard to *understand*. Calling
`stable_baselines3.DQN(...)` teaches you an API. Writing the replay buffer, the
target network, and the backward pass by hand teaches you what the algorithm
actually does — and, more usefully, what it does when it fails.

So everything here is hand-rolled, including the neural network.

## The two agents

**`agent.py` — tabular Q-learning.** A dictionary keyed by `(state, action)`,
epsilon-greedy action selection with decay from 1.0 to 0.01, ties broken at
random rather than by array order. Continuous sensor readings are bucketed into
discrete states, which is the whole reason the second agent exists: the table
grows too fast to generalise.

**`agent2.py` — deep Q-network, written in NumPy.** A 6→64→64→5 fully connected
network with ReLU activations:

- **He initialization** (`randn * sqrt(2/n)`) — the correct scaling for ReLU,
  and the difference between a network that learns and one that saturates.
- **Experience replay**, 10,000 transitions, uniformly sampled in minibatches to
  break the correlation between consecutive frames.
- **A target network**, copied from the online network every 1,000 steps, so the
  regression target stops moving while the estimator chases it.
- Forward and backward passes written out explicitly — the gradients are visible
  in the source rather than hidden behind autograd.

## The environment

The car has five distance sensors, cast as rays at −90°, −45°, 0°, +45° and +90°
from its heading, plus its own speed — a six-dimensional state. Five actions:
accelerate, brake, steer left, steer right, do nothing. Motion is a simple
kinematic model with acceleration, friction and angular steering.

Two tracks ship in `maps.py` (a circuit, and the same circuit with the finish
line moved to the far end, which stops the agent from memorising one direction).

## Reward shaping is the actual problem

The learning algorithm is the easy half. Getting a car to drive well is mostly a
question of what you pay it for, and every reward term here can be toggled and
tuned live from the in-app UI (`ui_manager.py`) without editing code or
restarting:

| Term | Default | What it's for |
|---|---:|---|
| `goal_reward` | +5000 | Reaching the finish line |
| `wall_penalty` | −5000 | Crashing |
| `sensor_penalty` | −5 | Being too close to a wall — teaches margin, not just survival |
| `idle_penalty` | −5 | Sitting still, which otherwise scores better than crashing |
| `speed_reward_factor` | ×0.3 | Going fast |
| `distance_reward_factor` | ×0.1 | Closing on the finish line |
| `steering_penalty` | −0.1 | Discourages the wobble that pure speed reward produces |

The idle penalty is the instructive one. Without it, a car that never moves
never crashes, and a wall penalty of −5000 makes standing still an excellent
strategy. Sitting in a local optimum doing nothing is the most common way this
kind of agent "learns".

Presets live in `training_config.json` and can be saved and reloaded from the UI.

## Seeing what the agent thinks

`heatmap.py` renders the Q-table as a red-green gradient over sensor distance
(x) against steering action (y) — green where the agent expects reward, red
where it expects to crash. Watching that surface form during training is far
more informative than a reward curve: you can see the moment the agent works out
that a wall dead ahead means turn, and you can see which regions of the state
space it has simply never visited.

## Running it

| Command | What it does |
|---|---|
| `python game.py` | Train from scratch, with live visualisation |
| `python run_trained.py` | Load committed weights and watch it drive |

Both open a menu to pick the agent (Q-learning or DQN) and the track. Training
has a fast mode that skips rendering when you want episodes rather than a show.

`q_table.pkl` and `dqn_model.pkl` are the trained parameters. Delete them to
start clean.

## Layout

```
agent.py       tabular Q-learning
agent2.py      DQN — replay buffer, target network, NumPy backprop
car.py         kinematics and sensor ray casting
game.py        training loop, rendering, main entry point
run_trained.py inference-only runner
heatmap.py     Q-value visualiser
maps.py        track definitions
config.py      hyperparameters and reward weights, load/save
ui_manager.py  in-app sliders and toggles
```

## Limitations

Honest about what this is: the tracks are simple, the physics is kinematic
rather than dynamic (no tyre slip, no weight transfer), and the DQN's NumPy
backward pass is written for clarity over speed — a framework would train this
in a fraction of the time. It is a learning exercise, and it is meant to be read.

## License

MIT — see [LICENSE](LICENSE).
