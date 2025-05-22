import gymnasium as gym
from gymnasium.spaces import Discrete

def make_taxi_env(render_mode=None):
    env = gym.make("Taxi-v3", render_mode=render_mode)
    assert isinstance(env.observation_space, Discrete)
    assert isinstance(env.action_space, Discrete)
    return env
