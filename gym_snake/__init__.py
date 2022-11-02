from gym_snake.version import VERSION as __version__
from gym.envs.registration import register

__author__ = "Ming Yu"

register(
    id="Snake-v0",
    entry_point="gym_snake.environment:SnakeEnv",
)
