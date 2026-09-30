# LatencyAwareRL (CTRLE)

A research framework for **latency-aware reinforcement learning** built on semi-Markov decision processes (sMDPs).

Instead of the classic fixed-timestep RL loop, environments here are **discrete-event simulations**: the agent does not act every fixed `dt`. Instead, each action includes a **commitment horizon** — how long the current control should be held before the next decision — while the world evolves continuously in between (with random disturbances, control expiries, timeouts, etc. firing as events). Every `step()` therefore returns `tau`, the continuous time actually elapsed since the last decision, and the learning side discounts accordingly with `gamma ** tau` (sMDP-style PPO/GAE).

The environments themselves are written in fast, header-only **C++23** (OpenMP-parallelized, exposed to Python via **pybind11**), while training, visualization, and the lab GUI are pure **Python** (PyTorch + PyQt/pyqtgraph).

## Highlights

- **sMDP / discrete-event core** — priority-queue event simulation (`Decision`, `ControlExpiry`, `Disturbance`, `Timeout`, ...), a simulation `Clock`, and a `StepResult` carrying the elapsed time `tau`
- **Latency as part of the policy** — the agent outputs a control *and* a commitment horizon (mapped through `tanh` into `[h_min, h_max]`), so it learns *when* to think, not just *what* to do
- **C++20 concepts-checked environments** — `Environment` and `MultiAgentEnvironment` concepts enforce a uniform `reset / step / observe` interface (validated via `static_assert`)
- **Vectorized, OpenMP-parallel envs** — a single `VecEnv` steps thousands of environments with the GIL released
- **PPO with sMDP discounting** — GAE using `gamma^tau` per step, supporting both single- and multi-agent (per-agent rollout splitting)
- **Built-in lab GUI** — Qt application with a *Train* tab (launch training, live return/entropy plots, log console) and a *Replay* tab (load checkpoints and watch episodes rendered at adjustable speed)
- **Pluggable env registry** — new environments are discovered automatically via `envs/*/_register.py` files

## Environments

### RobotNav (`robot_env`)
Single-agent point-mass navigation in a 5×5 world:

- reach the goal while avoiding randomly placed obstacles and staying inside the walls
- physics: thrust/drag integration with fixed micro-steps between events
- stochastic **disturbances** (exponentially distributed inter-arrival) kick the robot between decisions
- action = 2D thrust + 1 commitment horizon (held in `[0.2, 2.0]` s); terminates on goal / crash / wall, truncates on the 30 s time budget

### CarRacing (`racing_env`)
Two-agent multi-agent racing on a straight track with randomly spawned rocks:

- progress-based reward, crash on rocks or track boundaries, car–car bumping with momentum exchange
- asynchronous decisions: each car has its own decision events, so `current_agents()` tells you *which* agent acts in each parallel env at each step
- action = longitudinal/lateral thrust + commitment horizon (held in `[0.15, 1.2]` s)

## Repository layout

```
CTRLE/
├── CMakeLists.txt              # pybind11 module build (robot_env, racing_env)
├── include/
│   ├── core/                   # Clock, Event queue + concepts, Environment concepts, Runner (VecEnv)
│   ├── robot_env/              # RobotNavEnv + OpenMP parallel env
│   └── car_env/                # CarRacingEnv + OpenMP parallel env
└── python/
    ├── run.py                  # entry point for the lab GUI
    ├── agents/ppo.py           # PPO agent (sMDP-aware GAE on the training side)
    ├── envs/
    │   ├── robot_env/          # bindings, training CLI, replay renderer, registry hook
    │   └── racing_env/         # bindings, training CLI, replay renderer, registry hook
    └── interface/              # Qt application, train/replay tabs, env registry
```

## Requirements

- C++23 compiler (GCC 13+ / Clang 17+)
- CMake ≥ 3.16
- [pybind11](https://github.com/pybind/pybind11), Eigen3, OpenMP
- Python ≥ 3.9 with `numpy`, `torch`, `pyqtgraph` (Qt bindings)

## Building

```bash
cmake -B build
cmake --build build -j
```

This compiles the `robot_env` and `racing_env` pybind11 modules and places the `.so`/`.pyd` files directly into `python/`, where the training scripts and GUI import them from. Note that the modules are compiled with `-march=native`, so build on the machine you plan to run on.

## Usage

All Python tooling lives in the `python/` directory and expects the compiled env modules to be importable from there.

### GUI

```bash
cd python
python run.py
```

- **Train tab**: pick a model + environment, set the number of parallel envs, updates, rollout length, seed, learning rate and entropy coefficient, then *Start training* — the training process runs in the background while mean return and entropy are plotted live.
- **Replay tab**: choose an environment, load checkpoint file(s) (for multi-agent envs, checkpoints named `..._0.pt`, `..._1.pt` are picked up together), and watch episodes at an adjustable playback speed.

### CLI training

```bash
cd python

# RobotNav
python -m envs.robot_env.main --envs 1024 --episodes 700 --steps 128 --seed 42

# CarRacing (2 agents, each with its own PPO policy)
python -m envs.racing_env.main --envs 256 --episodes 150 --steps 128 --seed 42
```

Common flags: `--envs` (parallel environments), `--episodes` (updates), `--steps` (rollout length), `--epochs`, `--minibatch`, `--clip`, `--entr`, `--vf`, `--gamma`, `--gae_lambda`, `--lr`, `--seed`. Checkpoints are written to `python/models/<env>/` (create the directory first); the best model by mean return is kept alongside periodic snapshots.

## Adding a new environment

1. Implement the env in `include/` satisfying the `Environment` (or `MultiAgentEnvironment`) concept and expose it through a pybind11 module in `CMakeLists.txt`.
2. In `python/envs/<your_env>/`, add the bindings import, a `main.py` training script, a replay renderer (subclass of `BaseRenderer`), and a `_register.py` calling `register(EnvSpec(...))`.
3. The GUI and registry pick it up automatically — no other wiring needed.

## Notes

- The environment core is **stateless**: all mutable simulation state lives in a `State` struct owned by the caller, which is what makes cheap OpenMP-parallel vectorization possible.
- Randomness is a fast per-state splitmix64-style RNG stream, seeded per parallel env (`seed + i`), so runs are reproducible.
- `python/envs/*/*.pyi` stub files are provided for type checking of the compiled modules.

