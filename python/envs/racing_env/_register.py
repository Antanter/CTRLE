
import numpy as np
import racing_env

import agents.ppo as ppo
from interface.registry import register, EnvSpec
from envs.racing_env.replay import CarRenderer


N_AGENTS = 2

def _current_agent(env) -> int:
    return int(np.asarray(env.current_agents()).reshape(-1)[0])


def _status(rew: float, terminated: bool) -> str:
    if not terminated:
        return "TIMEOUT"
    return "FINISH" if rew > 0 else "CRASH"


register(EnvSpec(
    code="racing_env",
    name="CarRacing",
    make_single=lambda seed: racing_env.RacingEnv(seed),
    make_vec=lambda seed, n: racing_env.VecRacingEnv(seed, n),
    make_agent=lambda env, idx: ppo.Agent(env.act_dim(), env.obs_dim(), env),
    make_renderer=lambda: CarRenderer(N_AGENTS),
    n_agents=N_AGENTS,
    current_agent=_current_agent,
    default_ckpt="ppo_racer_agent0.pt",
    status_fn=_status,
))