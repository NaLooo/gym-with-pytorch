import numpy as np

from utils import ReplayBuffer


class Trainer:
    def __init__(
        self,
        target,
        target_size,
        replay_size=10000,
        batch_size=64
    ) -> None:

        self.target = target
        self.target_size = target_size
        self.replay_buffer = ReplayBuffer(replay_size, batch_size)

    def train_on_policy(self, agent):
        result = []
        iter = 0

        while True:
            iter += 1
            reward, exp = agent.play_episode()
            agent.update_net(exp.batch())
            result.append(reward)
            print('Episode {}: {}'.format(iter, reward))
            if np.mean(result[-self.target_size:]) > self.target:
                break

        return result

    def train_off_policy(self, agent, cycle=10):
        result = []
        iter = 0

        while True:
            iter += 1
            reward, exp = agent.play_episode()
            self.replay_buffer.extend(exp)
            if self.replay_buffer.enough_exp:
                for _ in range(cycle):
                    agent.update_net(self.replay_buffer.sample(), True)
            result.append(reward)
            print('Episode {}: {}'.format(iter, reward))
            if np.mean(result[-self.target_size:]) > self.target:
                break

        return result
