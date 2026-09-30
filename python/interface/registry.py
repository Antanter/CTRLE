
import importlib
import pathlib
from dataclasses import dataclass
from typing import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _default_status(rew: float, terminated: bool) -> str:
    if not terminated:
        return "TIMEOUT"
    return "GOAL" if rew > 0 else "CRASH"


@dataclass
class EnvSpec:
    code: str
    name: str
    make_single: Callable
    make_vec: Callable
    make_agent: Callable
    make_renderer: Callable
    n_agents: int = 1

    current_agent: Callable = lambda _ : 0
    default_ckpt: str = "agent.pt"
    status_fn: Callable = _default_status


REGISTRY: dict[str, EnvSpec] = {}


def register(spec: EnvSpec) -> None:
    REGISTRY[spec.code] = spec


def discover(root: str = "envs") -> None:
    for p in (ROOT / root).rglob("_register.py"):
        rel = p.relative_to(ROOT).with_suffix("")
        importlib.import_module(".".join(rel.parts))