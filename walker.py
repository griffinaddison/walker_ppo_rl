from dm_control import suite,viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

## Gotta do this EXTRA stuff cause mac is EXTRA
import matplotlib  
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import scipy.signal

from tqdm import tqdm


from models import *

import time as time
import os



class RolloutBuffer:
    def __init__(self, action_dim):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def data(self):
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32)
        }

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []




def compute_returns_advantages(rewards, values, dones, gamma, lam):
    """ Compute advantages with simple formla (not GAE) """

    ## Compute returns via discounted cumsum
    returns = np.zeros(len(rewards))
    current_return = 0
    for i in reversed(range(len(rewards))):
        current_return = rewards[i] + gamma * current_return * (1 - dones[i])
        returns[i] = current_return

    ## Compute advantages using GAE
    advantages = np.zeros_like(returns)
    gae = 0
    for i in reversed(range(len(rewards))):
        next_value = values[i+1] if i+1 < len(rewards) else 0 
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * gae * (1 - dones[i])
        advantages[i] = gae

    return returns, advantages

def update_actor(actor_critic, actor_optimizer, buffer, advantages, clip_param, epochs=80):

    data = buffer.data()

    for step in (range(epochs)):

        states = torch.as_tensor(data['states'], dtype=torch.float32)
        actions = torch.as_tensor(data['actions'], dtype=torch.float32)
        log_probs = torch.as_tensor(data['log_probs'], dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)

        ## Calculate clipped-surrogate loss
        # Get log probs from new actor
        _, new_log_probs = actor_critic.pi(states, actions) # Pytorch is able to batch these
        # Calculate the ratio of the new and old probabilities
        ratios = torch.exp(new_log_probs - log_probs)
        # Clipping method
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -(torch.min(surr1, surr2)).mean()  # Take the negative min of surr1 and surr2 for maximization

        ## TODO: could add entropy loss to encourage exploration

        ## TODO: could end optimization early based on KL-divergence target

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()


def update_critic(actor_critic, critic_optimizer, buffer, returns, epochs=80):

    data = buffer.data()

    for _ in (range(epochs)):

        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        states = torch.as_tensor(data['states'], dtype=torch.float32)

        ## Gradient descent
        critic_optimizer.zero_grad()
        value_loss = F.mse_loss(actor_critic.v(states), returns_t)
        value_loss.backward()
        critic_optimizer.step()


def visualize_policy(env, actor, num_episodes=1):
    """ Visualize the actor's policy in the environment. """
    # print("Available keys in observation:", time_step.observation.keys())
    def policy(time_step):
        # Ensure observation is 
        obs=time_step.observation
        obs=np.array(obs['orientations'].tolist()+[obs['height']]+obs['velocity'].tolist())        # obs = time_step.observation['observations']
        obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Add batch dimension
        pi, _ = actor(obs_tensor)
        action = pi.sample()
        
        if np.any(np.isnan(action)):
            print("NaN detected in action output")
        
        return action

    for _ in range(num_episodes):
        viewer.launch(env, policy)


def ppo_train(env, actor_critic, episodes, steps_per_episode, max_ep_len=1000, gamma=0.99, lam=0.97, 
              clip_param=0.2, actor_lr=1e-3, critic_lr=1e-3, actor_update_epochs=80, critic_update_epochs=80):

    model_dir = f"models/PPO-{int(time.time())}"
    os.makedirs(model_dir, exist_ok=True)
    
    actor_optimizer = optim.Adam(actor_critic.pi.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(actor_critic.v.parameters(), lr=critic_lr)

    # get action dim
    action_dim = env.action_spec().shape[0]
    buffer = RolloutBuffer(action_dim)

    episode_returns = []
    episode_lengths = []

    t = env.reset()
    x = t.observation
    x = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

    for episode in tqdm(range(episodes), desc="Training PPO", unit="episodes"):

        traj_returns, traj_lens = [], []
        traj_return, traj_len = 0, 0
        buffer.clear()

        ## For as much as 4000 steps per episode 
        for step in range(steps_per_episode):
            with torch.no_grad():
                u, v, log_prob_u = actor_critic.step(torch.as_tensor(x, dtype=torch.float32))
                r = env.step(u)
                _xp = r.observation
                done = r.last()
                xp=np.array(_xp['orientations'].tolist()+[_xp['height']]+_xp['velocity'].tolist())

                traj_return += r.reward
                traj_len += 1

                buffer.store(x, u, r.reward.item(), r.last(), log_prob_u, v.item())

                ## Update state
                x=xp

                timeout = traj_len == max_ep_len
                if done or timeout:
                    traj_returns.append(traj_return)
                    # print(f"\n traj done at step: {step}, traj_ret: {traj_return}, traj_len: {traj_len}\n")
                    traj_return = 0

                    t = env.reset()
                    x = t.observation
                    x = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
                    traj_return, traj_len = 0, 0

        episode_return = np.mean(traj_returns) if traj_returns else 0
        episode_returns.append(episode_return)


        ## Step 4 & 5: Compute rewards-to-go and advantages
        returns, advantages = compute_returns_advantages(buffer.rewards, \
                                                             buffer.values, \
                                                             buffer.dones, \
                                                             gamma, 
                                                             lam)
        ## Step 6 & 7: Update policy and value networks
        update_actor(actor_critic,
                     actor_optimizer,
                     buffer,
                     advantages, 
                     clip_param,
                     epochs=actor_update_epochs)
        update_critic(actor_critic,
                      critic_optimizer,
                      buffer,
                      returns,
                      epochs=critic_update_epochs)

        print(f"\n episode #{episode} done: episode_return: {episode_return:.2f}\n")
        if episode % 50 == 0:
            torch.save(actor_critic.state_dict(), f'{model_dir}/actor_critic_{episode}.pth')
            print(f'\nModel saved at episode {episode}')

    return actor_critic, episode_returns, episode_lengths




# Setup the environment
seed = 82401
torch.manual_seed(seed)
np.random.seed(seed)

r0 = np.random.RandomState(42)
env = suite.load("walker", "walk", task_kwargs={'random': False})

# Parameters
U = env.action_spec(); action_dim = U.shape[0]
X = env.observation_spec(); state_dim = 14 + 1 + 9
hidden_dim_actor = 64  # You can adjust this
hidden_dim_critic = 64
gamma = 0.95
lam = 0.97
clip_param = 0.2
actor_lr = 3e-4
critic_lr = 1e-3
actor_update_epochs = 80 
critic_update_epochs = 80 
episodes = 750 
steps_per_episode = 15000
episodes = 10 
steps_per_episode = 2000
max_ep_len = 1000


ac_kwargs = {}
actor_critic = MLPActorCritic(state_dim, action_dim, **ac_kwargs)

# Run the training loop
actor_critic, episode_returns, episode_lengths = ppo_train(env, 
                                                           actor_critic, 
                                                           episodes,
                                                           steps_per_episode, 
                                                           max_ep_len,
                                                           gamma,
                                                           lam,
                                                           clip_param, 
                                                           actor_lr, 
                                                           critic_lr, 
                                                           actor_update_epochs, 
                                                           critic_update_epochs)

print("Training completed!")

# Plotting
plt.ion()
plt.figure(figsize=(10, 5))
plt.plot(episode_returns, label='Episode Return')
plt.xlabel('Episodes')
plt.ylabel('Total Return')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show(block=True)





