# CTRLE (Continuous Time RL Engine)

RL framework built on theory of sMDP.

The environment is time-continuous, in means of implementation: instead of fixed step, there's a queue of events, and the runner goes through records, integrating relevant variables along the passed time. In PPO, discount is computed as `gamma ** tau`.

The environments themselves are written in fast, header-only OpenMP-parallelized **C++23**, exposed to Python via pybind11. Training, visualization, and the GUI are pure **Python**.

## Environments

### RobotNav (`robot_env`)
Single-agent physical navigation in a square L^2 world:

- reach the goal while avoiding randomly placed obstacles and staying inside the walls
- physics: thrust/drag integration with fixed micro-steps between events
- stochastic disturbances (exponentially distributed inter-arrival) kick the robot between decisions
- action = thrust (x) x thrust (y) x commitment horizon (held in `[0.2, 2.0]` s); terminates on goal / crash / wall, truncates on the 30`s time budget

### CarRacing (`racing_env`)
Two-agent racing on a straight track with randomly spawned rocks:

- progress-based reward, crash on rocks or track boundaries, car–car bumping with momentum exchange
- asynchronous decisions: each car has its own decision events, so `current_agents()` tells you *which* agent acts in each parallel env at each step
- action = longitudinal thrust x lateral thrust x commitment horizon (held in `[0.15, 1.2]` s)

## Requirements

- C++23 compiler (GCC 13+ / Clang 17+)
- CMake ≥ 3.16
- pybind11, Eigen3, OpenMP
- Python ≥ 3.9

## Building

First, install `cmake`, a C++ compiler, `pybind11` and `eigen` with your OS package manager.

```bash
# Arch
sudo pacman -S cmake pybind11 eigen
# Debian/Ubuntu
sudo apt install cmake pybind11-dev libeigen3-dev
# Fedora
sudo dnf install cmake pybind11-devel eigen3-devel
```

Then create a virtual environment and install the Python dependencies from `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Finally, build the C++ modules (with the venv still active):

```bash
cmake -B build -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

This compiles the `robot_env` and `racing_env` pybind11 modules and places the `.so`/`.pyd` files directly into `python/`, where the training scripts and GUI import them from. Note that the modules are compiled with `-march=native`, so build on the machine you plan to run on. If you change the Python version or recreate the venv, delete `build/` and rebuild.

## Usage

All Python tooling lives in the `python/` directory and expects the compiled env modules to be importable from there.

### GUI

```bash
cd python
python run.py
```

- **Train tab**: pick a model + environment, set the number of parallel envs, updates, rollout length, seed, learning rate and entropy coefficient, then *Start training* — the training process runs in the background while mean return and entropy are plotted live.
- **Replay tab**: choose an environment, load checkpoint file(s) (for multi-agent envs, checkpoints are named `..._0.pt`, `..._1.pt`), and watch episodes at an adjustable playback speed.

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

1. Implement the env in `include/` satisfying the `Environment` or `MultiAgentEnvironment` concept and expose it through a pybind11 module in `CMakeLists.txt`.
2. In `python/envs/<your_env>/`, add the bindings import, a `main.py` training script, a replay renderer (subclass of `BaseRenderer`), and a `_register.py` calling `register(EnvSpec(...))`.
3. The registry picks it up automatically.

## Notes

- The environment core is stateless: all mutable simulation state lives in a `State` struct owned by the caller, which is what makes cheap OpenMP-parallel vectorization possible.
- Randomness is a fast per-state splitmix64-style RNG stream, seeded per parallel env (`seed + _env_idx`), so runs are reproducible.
- `python/envs/*/*.pyi` stub files are provided for type checking of the compiled modules.

