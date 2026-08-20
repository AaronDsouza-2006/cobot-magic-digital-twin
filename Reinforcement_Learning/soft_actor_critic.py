import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import random
import torch.nn.functional as F
import mujoco
import mujoco.viewer
from collections import deque
from cobot_env import CobotEnv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter(f"run/{int(time.time())}")


class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.critic1= nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state, action):
        inp = torch.cat([state, action], dim=-1)
        Q1 = self.critic1(inp)
        Q2 = self.critic2(inp)
        return Q1, Q2

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_limit):
        super().__init__()
        self.feature_head = nn.Sequential(
                    nn.Linear(state_dim, hidden_dims[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[0], hidden_dims[1]),
                    nn.ReLU(),
                )
        self.mean_head = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[1], action_dim)
        self.register_buffer(
            "action_limit",
            torch.tensor(action_limit, dtype=torch.float32)
        )

    def forward(self, state):
        # (B, 23)
        features = self.feature_head(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        sample = dist.rsample()
        entropy = dist.entropy().mean() # average entropy
        writer.add_scalar("charts/policy_entropy", entropy)
        action = torch.tanh(sample) 

        log_prob = dist.log_prob(sample)
        log_prob -= torch.log(1-action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = action * self.action_limit
    
        return action, log_prob

    def sample(self, state):
        features = self.feature_head(state)
        mean = self.mean_head(features)
        action = torch.tanh(mean)
        action = action * self.action_limit
        return action

class SoftActorCritic:
    def __init__(self, CobotCubeEnv):
        self.env = CobotCubeEnv

        state_dim = 7+7+3+3+3
        action_dim = 7
        hidden_dim = (256, 256)

        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim, 
                                  self.env.max_delta)

        self.target_entropy = - action_dim
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for param in self.target_critic.parameters():
            param.requires_grad = False

        initial_alpha = 0.2
        
        self.log_alpha = torch.tensor(np.log(initial_alpha), dtype=torch.float32, 
                                      requires_grad=True)

        self.replay_buffer = ReplayBuffer(500_000)

        self.gamma = 0.998
        self.batch_size = 256

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

    @torch.no_grad()
    def get_target(self, next_states, rewards, dones):
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)
        next_actions, log_prob = self.actor(next_states)

        target_Q1, target_Q2 = self.target_critic(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target = rewards + self.gamma * (1-dones) * (
            target_Q - self.log_alpha.exp() * log_prob
        )
        return target

    def update_critic(self, targets, states, actions):
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)

        Q1, Q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(Q1, targets) + F.mse_loss(Q2, targets)   

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        states = torch.as_tensor(states, dtype=torch.float32)

        new_actions, log_prob = self.actor(states)
        Q1, Q2 = self.critic(states, new_actions)
        Q = torch.min(Q1, Q2)
        H = self.log_alpha.exp() * log_prob

        actor_loss = (H - Q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()


    def soft_target_update(self, tau=0.005):
        for target_param, critic_param in zip(self.target_critic.parameters(), 
                                            self.critic.parameters()):
            target_param.data.copy_(
                tau * critic_param.data +
                (1-tau) * target_param.data
            )

    def train(self, n_iterations, resume=False, print_alpha=False):
        self.actor.train()
        self.critic.train()
        state, _ = self.env.reset()
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        episode_reward = 0
        for step in range(n_iterations):
            if step < n_iterations//10 and not resume:
                action = self.env.action_space.sample()
            else:
                action, _ = self.actor(state)
                action = action.squeeze(0).detach().numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            # print('action',action)
            # print('ctrl',self.env.data.ctrl)
            # print('qpo',self.env.data.qpos[-8:])
            episode_reward += reward
            done = terminated or truncated
            self.replay_buffer.push(state.squeeze(0).numpy(), action, reward, next_state, done)

            if len(self.replay_buffer) >= self.batch_size:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
                
                targets = self.get_target(next_states, rewards, dones)
                self.update_critic(targets, states, actions)
                self.update_actor(states)
                self.soft_target_update()

            if done:    
                print(f"Step {step}: {episode_reward:.2f}")
                episode_reward = 0
                state, _ = self.env.reset()
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                
            else:
                state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

            if print_alpha and step % 5000 == 0:
                print(torch.exp(self.log_alpha).item())
                print("log_prob:", _.mean().item())



    def train_with_plot(self, n_iterations, resume=False, print_alpha=False):
        self.actor.train()
        self.critic.train()

        state, _ = self.env.reset()
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        episode_reward = 0
        episode_rewards = []

        plt.ion()
        fig, ax = plt.subplots()

        for step in range(n_iterations):
            if step < n_iterations // 10 and not resume:
                action = self.env.action_space.sample()
            else:
                action, _ = self.actor(state)
                action = action.squeeze(0).detach().numpy()

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward += reward
            episode_rewards.append(reward)

            done = terminated or truncated

            self.replay_buffer.push(
                state.squeeze(0).numpy(),
                action,
                reward,
                next_state,
                done
            )

            if len(self.replay_buffer) >= self.batch_size:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                targets = self.get_target(next_states, rewards, dones)
                self.update_critic(targets, states, actions)
                self.update_actor(states)
                self.soft_target_update()

            if done:
                print(f"Step {step}: {episode_reward:.2f}")

                # Plot rewards from this episode
                ax.clear()
                ax.plot(episode_rewards)
                ax.axhline(0, linestyle="--")
                ax.set_title(f"Episode ending at step {step} | Return: {episode_reward:.2f}")
                ax.set_xlabel("Environment Step")
                ax.set_ylabel("Reward")
                ax.grid(True)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

                episode_reward = 0
                episode_rewards = []

                state, _ = self.env.reset()
                state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

            else:
                state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

            if print_alpha and step % 5000 == 0:
                print(torch.exp(self.log_alpha).item())

        plt.ioff()
        plt.show()


    @torch.no_grad()
    def eval(self, CobotCubeEnv):
        self.actor.eval()
        eval_env = CobotCubeEnv
        state, _ = eval_env.reset()
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while(not done):
            action = self.actor.sample(state)
            action = action.squeeze(0).detach().numpy()
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
            time.sleep(0.01)
        eval_env.close()