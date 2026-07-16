from functools import partial
import sys
import os
from envs.mpeenv import PyMARLMPEEnv
from envs.multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["mpe"] = partial(env_fn, env=PyMARLMPEEnv)

try:
    from smac.env import StarCraft2Env
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
except ImportError:
    pass

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
