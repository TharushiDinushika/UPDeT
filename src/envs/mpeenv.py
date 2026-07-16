import numpy as np
from envs.multiagentenv import MultiAgentEnv
from envs.mpe.environment import MultiAgentEnv as MPEEnv
import envs.mpe.scenarios as scenarios

class PyMARLMPEEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        env_args = kwargs
        
        # Ensure we have a scenario object
        scenario_name = env_args.get("scenario_name", "simple_tag_adapt")
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        
        # We need an object that has attributes expected by make_world
        class Args:
            pass
        world_args = Args()
        for k, v in env_args.items():
            setattr(world_args, k, v)
        
        world = scenario.make_world(world_args)
        self.env = MPEEnv(world, reset_callback=scenario.reset_world, 
                          reward_callback=scenario.reward, 
                          observation_callback=scenario.observation, 
                          info_callback=scenario.info,
                          discrete_action=True)
                          
        self.n_agents = self.env.n
        self.episode_limit = getattr(world_args, "episode_length", 25)
        self._obs = None
        self._state = None
        self.reset()
        
    def step(self, actions):
        """ Returns reward, terminated, info """
        mpe_actions = []
        for i in range(self.n_agents):
            act = np.zeros(self.get_total_actions())
            act[actions[i]] = 1.0
            mpe_actions.append(act)
            
        obs_n, reward_n, done_n, info_n = self.env.step(mpe_actions)
        self._obs = obs_n
        
        reward = np.sum([r[0] for r in reward_n])
        terminated = all(done_n)
        info = info_n[0] if info_n else {}
        
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self._obs[0])

    def get_state(self):
        """ For MPE, state can just be concatenation of all obs """
        return np.concatenate(self._obs)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return 5  # no_op, right, left, up, down

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self.env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
