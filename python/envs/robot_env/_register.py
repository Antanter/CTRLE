
import robot_env

import agents.ppo as ppo
from interface.registry import register, EnvSpec
from envs.robot_env.replay import RobotRenderer

register(EnvSpec(
    code="robot_env",
    name="RobotNav",
    make_single=lambda seed: robot_env.RobotEnv(seed),
    make_vec=lambda seed, n: robot_env.VecRobotEnv(seed, n),
    make_agent=lambda env, idx: ppo.Agent(env.act_dim(), env.obs_dim(), env),
    make_renderer=lambda: RobotRenderer(),
    default_ckpt="ppo_smdp_agent.pt",
))