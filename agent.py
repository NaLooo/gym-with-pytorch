import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from random import randint, random
from utils import *
from nn import *


class ActorCriticAgent:
    def __init__(
        self,
        env,
        obs_size,
        action_size,
        discrete,
        actor_hidden_size=[128],
        critic_hidden_size=[128],
        epochs=10,
        gamma=0.98,
        lmbda=0.95,
        eps=0.2,
        e_greedy=False,
        start_epsilon=1.0,
        epsilon_step=0.01,
    ) -> None:

        self.env = env
        self.obs_size = obs_size
        self.action_size = action_size
        self.discrete = discrete
        self.epochs = epochs
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.e_greedy = e_greedy
        self.epsilon = start_epsilon
        self.epsilon_step = epsilon_step
        self.actor = self.get_actor(obs_size, actor_hidden_size, action_size)
        self.critic = self.get_critic(obs_size, critic_hidden_size)
        self.optimizers = [
            Adam(self.actor.parameters(), 1e-4),
            Adam(self.critic.parameters(), 3e-4)
        ]

    def get_actor(self, obs_size, actor_hidden_size, action_size):
        if self.discrete:
            return PolicyNetDiscrete(obs_size, actor_hidden_size, action_size)
        else:
            return PolicyNetContinuos(obs_size, actor_hidden_size, action_size)

    def get_critic(self, obs_size, critic_hidden_size):
        return ValueNet(obs_size, critic_hidden_size, 1)

    def play_episode(self):
        episode_reward = 0.0
        episode_exp = Experiences()
        state = self.env.reset()

        while True:
            action = self.action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode_exp.append(state, action, reward, done, next_state)
            if done:
                break
            episode_reward += reward
            state = next_state
        if self.e_greedy:
            self.update_epsilon()
        episode_exp.calc_advantage(self.gamma, self.lmbda)
        return episode_reward, episode_exp

    def action(self, state):
        prob = random()
        if self.discrete:
            if self.e_greedy and prob <= self.epsilon:
                action = randint(0, self.action_size-1)
            else:
                p = self.actor(torch.tensor(state)).detach().numpy()
                action = np.random.choice(self.action_size, 1, p=p)[0]
        else:
            if self.e_greedy and prob <= self.epsilon:
                action = (random()-0.5) * self.action_size
            else:
                mu, sigma = self.actor(torch.tensor(state, dtype=torch.float))
                action = torch.distributions.Normal(mu, sigma).sample()
                action = action.clamp(-self.action_size,
                                      self.action_size).detach().numpy()
        return action

    def update_net(self, batch, off_policy=False):
        states, actions, rewards, trajectory_reward, done_mask, next_states = batch
        states = torch.tensor(np.array(states), dtype=torch.float)
        if self.discrete:
            actions = torch.tensor(actions, dtype=int).view(-1, 1)
        else:
            actions = torch.tensor(actions, dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(rewards).view(-1, 1)
        trajectory_reward = torch.tensor(trajectory_reward).view(-1, 1)
        done_mask = torch.tensor(done_mask, dtype=int).view(-1, 1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        self.train(states, actions, rewards,
                   trajectory_reward, done_mask, next_states, off_policy)

    def train(self, states, actions, rewards, trajectory_reward, done_mask, next_states, off_policy=False):
        with torch.no_grad():
            td_target = rewards + self.gamma * \
                self.critic(next_states) * (1-done_mask)
        for _ in range(self.epochs):
            td_delta = td_target - self.critic(states)
            log_probs = torch.log(self.actor(states).gather(1, actions))

            actor_loss = torch.mean(-log_probs * td_delta.detach())
            critic_loss = torch.mean(F.mse_loss(
                self.critic(states), td_target.detach()))

            self.gradient_descent(actor_loss, critic_loss)

    def gradient_descent(self, *loss_list):
        for opt in self.optimizers:
            opt.zero_grad()
        for loss in loss_list:
            loss.backward()
        for opt in self.optimizers:
            opt.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_step:
            self.epsilon -= self.epsilon_step
        if self.epsilon < 0.01:
            self.epsilon = 0.01


class PPOAgent(ActorCriticAgent):
    def __init__(self, env, obs_size, action_size, discrete, actor_hidden_size=[128], critic_hidden_size=[128], epochs=10, gamma=0.98, lmbda=0.95, eps=0.2, e_greedy=False, start_epsilon=1, epsilon_step=0.01) -> None:
        super().__init__(env, obs_size, action_size, discrete, actor_hidden_size,
                         critic_hidden_size, epochs, gamma, lmbda, eps, e_greedy, start_epsilon, epsilon_step)

    def train(self, states, actions, rewards, trajectory_reward, done_mask, next_states, off_policy=False):
        with torch.no_grad():
            td_target = (rewards + self.gamma *
                         self.critic(next_states) * (1-done_mask)).float()
            td_delta = td_target - self.critic(states)
            if off_policy:
                td_target = trajectory_reward
                advantage = trajectory_reward
            else:
                advantage = torch.tensor(calc_advantage(
                    self.gamma, self.lmbda, td_delta.detach().numpy()), dtype=torch.float)
            if self.discrete:
                old_log_probs = torch.log(self.actor(
                    states).gather(1, actions))
            else:
                mu, sigma = self.actor(states)
                old_log_probs = torch.distributions.Normal(
                    mu, sigma).log_prob(actions)

        for _ in range(self.epochs):
            if self.discrete:
                log_probs = torch.log(self.actor(states).gather(1, actions))
            else:
                mu, sigma = self.actor(states)
                log_probs = torch.distributions.Normal(
                    mu, sigma).log_prob(actions).float()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(
                self.critic(states), td_target))

            self.gradient_descent(actor_loss, critic_loss)
