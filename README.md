# gym-with-pytorch
## Quick Start
``` python
import gym

from trainer import Trainer
from agent import PPOAgent
from utils import display_result

ENV_TITLE = 'CartPole-v1'
TARGET = 400
TARGET_SIZE = 100
REPLAY_SIZE = 2000
BATCH_SIZE = 64

env = gym.make(ENV_TITLE)
agent = PPOAgent(env, obs_size=env.observation_space.shape[0], action_size=env.action_space.n, discrete=True)
trainer = Trainer(TARGET, TARGET_SIZE, REPLAY_SIZE, BATCH_SIZE)
result = trainer.train_on_policy(agent)
display_result(result, window_sizes=[20, 100])
```

### Result of PPO agent on CartPole-v1

Achieved 100-episode average rewards above 400 after 930 episodes played.

![result](/resource/result.png)
