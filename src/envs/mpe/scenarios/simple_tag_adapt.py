import numpy as np
from envs.mpe.core import World, Agent, Landmark, Action
from envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.world_length = getattr(args, "episode_length", 25)
        num_good_agents = args.num_good_agents
        num_adversaries = args.num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0 # No blocks in ADAPT setup
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            
            # ADAPT rule-based prey
            if not agent.adversary:
                agent.action_callback = self.prey_policy
                
        # add landmarks (none)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        
        # Custom step logic to update global state after physics
        original_step = world.step
        def custom_step():
            original_step()
            # update caught_matrix here!
            for adv_idx, adv in enumerate(self.adversaries(world)):
                for ag_idx, ag in enumerate(self.good_agents(world)):
                    if self.is_collision(ag, adv):
                        if not self.caught_matrix[adv_idx, ag_idx]:
                            self.caught_matrix[adv_idx, ag_idx] = True
                            self.newly_caught[adv_idx, ag_idx] = True
            if np.all(np.any(self.caught_matrix, axis=0)):
                self.game_winning_condition_met = True
                
        world.step = custom_step
            
        # Initial states
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        
        # ADAPT specific state tracking
        num_adversaries = len(self.adversaries(world))
        num_good_agents = len(self.good_agents(world))
        self.caught_matrix = np.zeros((num_adversaries, num_good_agents), dtype=bool)
        self.newly_caught = np.zeros((num_adversaries, num_good_agents), dtype=bool)
        self.game_winning_condition_met = False
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def prey_policy(self, agent, world):
        # Rule-based policy for prey: 
        # If inside [-1, 1], choose a random action.
        # If outside, move towards the center (inside) as quickly as possible.
        action = Action()
        action.u = np.zeros(world.dim_p)
        action.c = np.zeros(world.dim_c)
        
        pos = agent.state.p_pos
        
        # Check if out of bounds
        if pos[0] < -1.0 or pos[0] > 1.0 or pos[1] < -1.0 or pos[1] > 1.0:
            # Find the axis and direction that is most out of bounds
            if abs(pos[0]) > abs(pos[1]):
                if pos[0] > 0:
                    dir = 1 # move left (-x)
                else:
                    dir = 2 # move right (+x)
            else:
                if pos[1] > 0:
                    dir = 3 # move down (-y)
                else:
                    dir = 4 # move up (+y)
        else:
            dir = np.random.randint(0, 5) # 0: stay, 1: left, 2: right, 3: down, 4: up
            
        if dir == 1: action.u[0] -= 1.0
        elif dir == 2: action.u[0] += 1.0
        elif dir == 3: action.u[1] -= 1.0
        elif dir == 4: action.u[1] += 1.0
            
        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        action.u *= sensitivity
        
        return action

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Only predators are trained using RL, so we only return reward for adversaries
        if not agent.adversary:
            return 0.0
            
        rew = 0.0
        adversaries = self.adversaries(world)
        adv_idx = adversaries.index(agent)
        
        # Check newly caught for this predator
        for ag_idx in range(self.newly_caught.shape[1]):
            if self.newly_caught[adv_idx, ag_idx]:
                rew += 10.0 # Reward for first-time capture
                self.newly_caught[adv_idx, ag_idx] = False # Clear flag after rewarding
                    
        # Check if game-winning condition is met
        if self.game_winning_condition_met:
            rew += 20.0
            
        return rew

    def info(self, agent, world):
        return {'is_success': self.game_winning_condition_met}

    def observation(self, agent, world):
        # Observation includes pos, vel of itself
        # and relative pos, vel of others, plus captured indicator for prey
        # We interleave these per-entity so that the attention module can reshape it to (n_entities, token_dim)
        
        # Self features (dim = 6, padded with 0 to match token_dim)
        self_feats = np.concatenate([agent.state.p_vel, agent.state.p_pos, np.zeros(2)])
        
        # Entity features
        entity_feats = []
        
        # Landmarks (padded to 6 dims to match other agents)
        for entity in world.landmarks:
            if not entity.boundary:
                rel_pos = entity.state.p_pos - agent.state.p_pos
                pad = np.zeros(4) # pad vel (2), captured (1), and is_prey (1)
                entity_feats.append(np.concatenate([rel_pos, pad]))
                
        agents = self.good_agents(world)
        
        # Other agents (dim = 6: rel_pos(2) + vel(2) + captured(1) + is_prey(1))
        for other in world.agents:
            if other is agent:
                continue
            
            rel_pos = other.state.p_pos - agent.state.p_pos
            vel = other.state.p_vel
            
            if not other.adversary:
                ag_idx = agents.index(other)
                is_captured = 1.0 if np.any(self.caught_matrix[:, ag_idx]) else 0.0
                captured = np.array([is_captured])
                is_prey = np.array([1.0])
            else:
                captured = np.array([0.0])
                is_prey = np.array([0.0])
                
            entity_feats.append(np.concatenate([rel_pos, vel, captured, is_prey]))
            
        if len(entity_feats) > 0:
            return np.concatenate([self_feats] + entity_feats)
        else:
            return self_feats
