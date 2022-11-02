import gym
import gym_snake

from trainer import *
from agent import *
from utils import *

ENV_TITLE = 'Snake-v0'
TARGET = 400
TARGET_SIZE = 100
REPLAY_SIZE = 2000
BATCH_SIZE = 64
CYCLE = 10

env = gym.make(ENV_TITLE, render_mode='human',
               width=20, height=20, block_size=40)
agent = PPOAgent(env, 6, 4, True)
trainer = Trainer(TARGET, TARGET_SIZE, REPLAY_SIZE, BATCH_SIZE)
result = trainer.train_on_policy(agent)
#result = trainer.train_off_policy(agent, CYCLE)
display_result(result, [20, 100])
