from shutil import ExecError
import matplotlib.pyplot as plt
import numpy as np

from collections import *
from nn import *

# 'advantage',
Experience = namedtuple('Experience', field_names=[
                        'state', 'action', 'reward', 'trajectory_reward', 'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, size=10000, batch_size=64) -> None:
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size

    @property
    def enough_exp(self):
        return len(self.buffer) >= self.buffer.maxlen * 0.3

    def extend(self, exp):
        for i in range(len(exp)):
            self.buffer.append(Experience(
                exp.state[i], exp.action[i], exp.reward[i], exp.trajectory_reward[i], exp.done[i], exp.new_state[i]))

    def sample(self):
        indices = np.random.choice(
            len(self.buffer), self.batch_size, replace=False)
        batch = zip(*[self.buffer[idx] for idx in indices])
        return batch


class Experiences:
    def __init__(self) -> None:
        self.state = []
        self.action = []
        self.reward = []
        self.trajectory_reward = []
        self.done = []
        self.new_state = []

    def __len__(self):
        return len(self.state)

    def append(self, s, a, r, d, n):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(d)
        self.new_state.append(n)

    def calc_advantage(self, gamma, lmbda):
        self.trajectory_reward = calc_advantage(gamma, lmbda, self.reward)

    def zip(self):
        pass

    def batch(self):
        return (self.state, self.action, self.reward, self.trajectory_reward, self.done, self.new_state)


def display_result(result, window_sizes, show_unscaled=False):
    if show_unscaled:
        plt.plot(result, label='unscaled')
    for i in window_sizes:
        if i > len(result):
            continue
        x = np.arange(i, len(result))
        plt.plot(x, moving_average(result, i), label=str(i)+' episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend(loc='upper left')
    plt.show()


def moving_average(arr, size):
    res = []
    for i in range(len(arr)-size):
        res.append(np.mean(arr[i:i+size]))
    return res


def calc_advantage(gamma, lmbda, arr):
    advantage_list = []
    advantage = 0.0
    for item in arr[::-1]:
        advantage = gamma * lmbda * advantage + item
        advantage_list.append(advantage)
    advantage_list.reverse()
    return advantage_list
