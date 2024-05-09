from dm_control import suite,viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

## Gotta do this EXTRA stuff cause mac is EXTRA
import matplotlib  
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import scipy.signal

from tqdm import tqdm


from models import *

class RolloutBuffer:
    def __init__(self, action_dim):
        # self.states = torch.empty((0, 24), dtype=torch.float32)
        # self.actions = torch.empty((0, action_dim), dtype=torch.float32)
        # self.rewards = torch.empty((0, 1), dtype=torch.float32)
        # self.dones = torch.empty((0, 1), dtype=torch.float32)
        # self.log_probs = torch.empty((0, 1), dtype=torch.float32)
        # self.values = torch.empty((0, 1), dtype=torch.float32)

        # self.states = np.array([], dtype=np.float32) 
        # self.actions = np.array([], dtype=np.float32)
        # self.rewards = np.array([], dtype=np.float32)
        # self.dones = np.array([], dtype=np.float32)
        # self.log_probs = np.array([], dtype=np.float32)
        # self.values = np.array([], dtype=np.float32)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


    def store(self, state, action, reward, done, log_prob, value):
        # Ensure each new data point is a row vector of the appropriate dimensions
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Reshape to 1x24 if not already
        # # action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        # reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        # done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
        # log_prob = torch.tensor([log_prob], dtype=torch.float32).unsqueeze(0)
        # value = torch.tensor([value], dtype=torch.float32).unsqueeze(0)
        #
        # # Concatenate along the first dimension (adding rows)
        # self.states = torch.cat((self.states, state), dim=0)
        # self.actions = torch.cat((self.actions, action), dim=0)
        # self.rewards = torch.cat((self.rewards, reward), dim=0)
        # self.dones = torch.cat((self.dones, done), dim=0)
        # self.log_probs = torch.cat((self.log_probs, log_prob), dim=0)
        # self.values = torch.cat((self.values, value), dim=0)
        #
        # self.states = torch.cat((self.states, torch.tensor(state, dtype=torch.float32)), dim=0)
        # self.actions = torch.cat((self.actions, torch.tensor(action, dtype=torch.float32)), dim=0)
        # self.rewards = torch.cat((self.rewards, torch.tensor([reward], dtype=torch.float32)), dim=0)
        # self.dones = torch.cat((self.dones, torch.tensor([done], dtype=torch.float32)), dim=0)
        # print("\n log_prob:", log_prob)
        # self.log_probs = torch.cat((self.log_probs, torch.tensor([log_prob], dtype=torch.float32)), dim=0)
        # self.values = torch.cat((self.values, torch.tensor([value], dtype=torch.float32)), dim=0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

        # self.states = np.append(self.states, state)
        # self.actions = np.append(self.actions, action)
        # self.rewards = np.append(self.rewards, reward)
        # self.dones = np.append(self.dones, done)
        # self.log_probs = np.append(self.log_probs, log_prob)
        # self.values = np.append(self.values, value)


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
        # self.states = torch.tensor([], dtype=torch.float32)
        # self.actions = torch.tensor([], dtype=torch.float32)
        # self.rewards = torch.tensor([], dtype=torch.float32)
        # self.dones = torch.tensor([], dtype=torch.float32)
        # self.log_probs = torch.tensor([], dtype=torch.float32)
        # self.values = torch.tensor([], dtype=torch.float32)

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # self.states = np.array([], dtype=np.float32) 
        # self.actions = np.array([], dtype=np.float32)
        # self.rewards = np.array([], dtype=np.float32)
        # self.dones = np.array([], dtype=np.float32)
        # self.log_probs = np.array([], dtype=np.float32)
        # self.values = np.array([], dtype=np.float32)


    # def get_shuffled_data(self):
    #    
    #     # # Retrieve data as dictionary of tensors
    #     # data = self.data()
    #     #
    #     # # Determine the length of the data
    #     # num_entries = self.states.size(0)
    #     # print("\n num_entries:", num_entries)
    #     # print("\n self.states.size():", self.states.size())
    #     #
    #     # # Create a permutation of indices to shuffle the data
    #     # shuffled_indices = torch.randperm(num_entries)
    #
    #     # Shuffle each tensor in the dictionary using the shuffled indices
    #
    #
    #
    #     shuffled_data = {key: val[np.randperm(len(val))] for key, val in self.data()}
    #
    #     return shuffled_data




## U = ACTOR = POLICY
# class Actor(nn.Module):
#     def __init__(self,xdim,udim,
#                  hdim=32):
#         super().__init__()
#         self.xdim,self.udim = xdim, udim
#
#         # self.activation = activation
#         ### TODO
#
#         ## Define layers
#         # self.fc1 = nn.Linear(xdim, hdim)
#         # self.fc2 = nn.Linear(hdim, hdim)
#         # self.output = nn.Linear(hdim, udim)
#
#
#         # # Initialize as orthogonal
#         # nn.init.orthogonal_(self.fc1.weight, gain=0.01)
#         # nn.init.orthogonal_(self.fc2.weight, gain=0.01)
#         # nn.init.orthogonal_(self.output.weight, gain=0.01)
#
#
#
#         self.mu_net = nn.Sequential(
#             nn.Linear(xdim, hdim), # Hidden layer 1 (linear)
#             nn.Tanh(),             # Hidden layer 1 (activation)
#             nn.Linear(hdim, hdim), # Hidden layer 2 (linear)
#             nn.Tanh(),             # Hidden layer 2 (activation)
#             nn.Linear(hdim, udim)  # Output layer
#         )
#
#         log_std = -0.5 * np.ones(udim, dtype=np.float32)
#         self.log_std_layer = nn.Parameter(torch.as_tensor(log_std)) # 0.5 so well-formed
#
        ### END TODO

    # def forward(self,x):
    #     ### TODO
    #
    #     # x = F.tanh(self.fc1(x))
    #     # x = F.tanh(self.fc2(x))
    #     #
    #     # mean = self.output(x)
    #
    #     mean = self.mu_net(x)
    #
    #     std = torch.exp(self.log_std_layer)
    #
    #     
    #     pi = Normal(mean, std)
    #     action = pi.sample()
    #     log_prob = pi.log_prob(action).sum(axis=-1) # log prob of each component of the action
    #     ### END TODO
    #     # print("\n type(action):", type(action))
    #     # print("\n type(log_prob):", type(log_prob))
    #     return action, log_prob

# class Critic(nn.Module):
#     def __init__(self, xdim, hdim=32):
#         super(Critic, self).__init__()
#
#
#         ## Input of state
#         # self.fc1 = nn.Linear(state_dim, hidden_dim)
#         # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         # self.output = nn.Linear(hidden_dim, 1)
#         ## Output of 1 (it outputs value given state)
#
#
#         # ## Initialize as orthogonal
#         # nn.init.orthogonal_(self.fc1.weight, gain=0.01)
#         # nn.init.orthogonal_(self.fc2.weight, gain=0.01)
#         # nn.init.orthogonal_(self.output.weight, gain=0.01)
#
#         self.v_net = nn.Sequential(
#             nn.Linear(xdim, hdim),
#             nn.Tanh(),
#             nn.Linear(hdim, hdim),
#             nn.Tanh(),
#             nn.Linear(hdim, 1)
#         )
#
#     def forward(self, x):
#         # x = F.tanh(self.fc1(state))
#         # x = F.tanh(self.fc2(x))
#         # value = self.output(x)
#
#         return torch.squeeze(self.v_net(x), -1)




# def rollout(e,actor,T=1000, render=False):
#     """
#     e: environment
#     actor: controller
#     T: time-steps
#     """
#
#     traj=[]
#     t=e.reset()
#     x=t.observation
#     x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
#     for i in range(T):
#         with torch.no_grad():
#             ## Sample action from pytorch nn
#             u, log_prob_u = actor(torch.from_numpy(x).float().unsqueeze(0))
#         u = u.numpy()
#         log_prob_u = log_prob_u.numpy()
#         ## Step environment using this action
#         r = e.step(u.numpy())
#         x=r.observation
#         xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
#
#         t=dict(xp=xp,r=r.reward,u=u, logp_u=log_prob_u, d=r.last())
#         traj.append(t)
#         x=xp
#         if r.last():
#             # print("\n done at iteration:", i)
#             break
#     # print("\n finished rolling out, traj length:", len(traj))
#     return traj




# def discount_cumsum(x, discount):
#     """
#     from spinningup's ppo:
#
#     magic from rllab for computing discounted cumulative sums of vectors.
#
#
#     input: 
#         vector x, 
#         [x0, 
#          x1, 
#          x2]
#
#     output:
#         [x0 + discount * x1 + discount^2 * x2,  
#          x1 + discount * x2,
#          x2]
#     """
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def compute_returns_advantages(rewards, values, dones, gamma):
    """ Compute advantages with simple formla (not GAE) """
    #
    # lam = 0.97
    #
    # advantages = torch.zeros_like(rewards)
    # gae = 0
    # for t in reversed(range(len(rewards))):
    #     if t == len(rewards) - 1 or dones[t]:
    #         next_value = 0
    #     else:
    #         next_value = values[t+1]
    #     ## TODO: check if value should be negative
    #     delta = rewards[t] + gamma * next_value - values[t]
    #     gae = delta + gamma * lam * gae * (1 - dones[t])
    #     advantages[t] = gae
    #
    # # num_steps = len(rewards)
    # # returns = torch.zeros_like(rewards)
    # # _return = 0
    # # for t in reversed(range(len(rewards))):
    # #     _return = rewards[t] + gamma * _return * (1 - dones[t])  
    # #     returns[t] = _return
    #
    # returns = values.detach() + advantages
    #
    # # deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    # # advantages = discount_cumsum(deltas.detach().numpy(), gamma * lam)
    # # returns = discount_cumsum(rewards.detach().numpy(), gamma)[:-1]
    # # 
    # # advantages = np.copy(advantages)
    # # returns = np.copy(returns) # copy to avoid negative stride error
    # #
    # # advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1)
    # # returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    #
    # # print("\n type retutns:", type(returns)) 
    # ## Normalize advantages
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
    #
    # ## Normalize returns
    # # returns = (returns - returns.mean()) / (returns.std() + 1e-8)


    # Compute the returns-to-go for each state
    returns_to_go = np.zeros(len(rewards))
    current_return = 0
    for i in reversed(range(len(rewards))):
        current_return = rewards[i] + gamma * current_return * (1 - dones[i])
        returns_to_go[i] = current_return

    # Compute the advantage using GAE
    advantages = np.zeros_like(returns_to_go)
    gae = 0
    lam = 0.97
    for i in reversed(range(len(rewards))):
        next_value = values[i+1] if i+1 < len(rewards) else 0 
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * gae * (1 - dones[i])
        advantages[i] = gae





    return returns_to_go, advantages

def update_actor(actor_critic, actor_optimizer, buffer, advantages, clip_param, epochs=80):

    for _ in (range(epochs)):


        # print("\n 1")
        data = buffer.data()

        states = torch.as_tensor(data['states'], dtype=torch.float32)
        # print("\n type(buffer.actions):", type(data.actions))
        # print("\n type(buffer.actions[0]):", type(data.actions[0]))
        actions = torch.as_tensor(data['actions'], dtype=torch.float32)
        # print("\n type(buffer.log_probs):", type(data.log_probs))
        log_probs = torch.as_tensor(data['log_probs'], dtype=torch.float32)
        advantages = torch.as_tensor(advantages, dtype=torch.float32)


        actor_optimizer.zero_grad()

        # Get log probs from new actor
        _, new_log_probs = actor_critic.pi(states, actions) # Pytorch is able to batch these
        
        # Calculate the ratio of the new and old probabilities
        ratios = torch.exp(new_log_probs - log_probs)
        
        # print("\n 2")
        # Clipping method
        # print("\n type(ratios):", type(ratios))
        # print("\n type(advantages):", type(advantages))
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -(torch.min(surr1, surr2)).mean()  # Take the negative min of surr1 and surr2 for maximization


        # ## add entropy bonus
        # entropy = -new_log_probs * torch.exp(new_log_probs)
        # actor_loss += 0.01 * entropy.mean()
        
        # print("\n 3")
        target_kl = 0.01
        kl = (log_probs - new_log_probs).mean().item()
        if kl > 1.5 * target_kl:
            break

        # print("\n 4")
        ## Optimize the actor/policy based on its previous loss
        actor_loss.backward()
        actor_optimizer.step()
        # print("\n 5")


def update_critic(actor_critic, critic_optimizer, buffer, returns, epochs=80):

    data = buffer.data()
    # print("\n type(data['states']):", type(data['states']))
    num_data = data['states'].shape[0]

    # for _ in tqdm(range(epochs), desc="Critic Update", unit="epochs"):
    for _ in (range(epochs)):

        critic_optimizer.zero_grad()


        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        states = torch.as_tensor(data['states'], dtype=torch.float32)
        value_preds = actor_critic.v(states)


        # Calculate loss
        # print("\n type(value_preds) should be tensor:", type(value_preds))
        # print("\n type(returns_t) should be tensor:", type(returns_t))
        # value_loss = F.mse_loss(value_preds, returns)
        value_loss = ((value_preds - returns_t) ** 2).mean()

        # Optimize the critic
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
        

        # action = action.detach().numpy().squeeze()  # Remove batch dimension
        
        # Debug: Check if action contains any None values
        if np.any(np.isnan(action)):
            print("NaN detected in action output")
        
        return action

    for _ in range(num_episodes):
        viewer.launch(env, policy)


def ppo_train(env, actor_critic, episodes, steps_per_episode, gamma=0.99, lam=0.97, 
              clip_param=0.2, actor_lr=1e-3, critic_lr=1e-3, actor_update_epochs=80, critic_update_epochs=80):
    
    actor_optimizer = optim.Adam(actor_critic.pi.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(actor_critic.v.parameters(), lr=critic_lr)

    # get action dim
    action_dim = env.action_spec().shape[0]
    buffer = RolloutBuffer(action_dim)

    episode_returns = []
    episode_lengths = []
    episode_max_advantages = []

    all_returns = []


    for episode in tqdm(range(episodes), desc="Training PPO", unit="episodes"):

        ep_ret = 0
        ep_len = 0

        # # if episode % 10000 == 0:
        # if episode == episodes - 1:
        #     print(f"Visualizing policy at episode {episode}")
        #     visualize_policy(env, actor_critic.pi)

        # trajectory_returns = torch.tensor([], dtype=torch.float32)
        # trajectory_lengths = torch.tensor([], dtype=torch.float32)



        for step in tqdm(range(steps_per_episode), desc="Steps per Episode", unit="steps"):
            # traj = []
            buffer.clear()
            t = env.reset()
            x = t.observation
            x = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
            ep_ret, ep_len = 0, 0
            while True:
                with torch.no_grad():
                    # u, log_prob_u = actor(torch.from_numpy(x).float().unsqueeze(0))
                    u, v, log_prob_u = actor_critic.step(torch.as_tensor(x, dtype=torch.float32))
                # u = u.numpy()
                # log_prob_u = log_prob_u.numpy()
                r = env.step(u)
                x = r.observation
                xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

                # v = actor_critic.v(torch.tensor(xp, dtype=torch.float32).unsqueeze(0))
                # print("\n type(u):", type(u))
                buffer.store(xp, u, r.reward.item(), r.last(), log_prob_u, v.item())

                # t=dict(xp=xp,r=r.reward,u=u, logp_u=log_prob_u, d=r.last())
                # traj.append(t)

                ep_ret += r.reward
                ep_len += 1

                ## Update state
                x=xp


                if r.last() or ep_len == steps_per_episode:
                    # print("\n done at step:", ep_len)
                    all_returns.append(ep_ret)


                    ## Step 4 & 5: Compute rewards-to-go and advantages

                    returns, advantages = compute_returns_advantages(buffer.rewards, \
                                                                     buffer.values, \
                                                                     buffer.dones, \
                                                                     gamma)


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



                    break

        average_return = np.mean(all_returns) if all_returns else 0
        episode_returns.append(average_return)
        episode_max_advantages.append(advantages.max().item())
        if episode % 10 == 0:
            print(f"\n Average return for episode {episode}: {average_return:.2f}")
         


        # ## Step 3: Rollout current policy 
        # for traj_idx in tqdm(range(traj_per_episode), desc="Rollout", unit="trajectories"):
        #
        #     trajectory = rollout(env, actor)
        #     trajectory_lengths = torch.cat((trajectory_lengths, torch.tensor([len(trajectory)])))
        #
        #     for step in trajectory:
        #         # Ask critic to estimate the value of this state
        #         v = critic(torch.tensor(step['xp'], dtype=torch.float32).unsqueeze(0)).item()
        #         # Then store this step in the data buffer
        #         buffer.store(step['xp'], step['u'], step['r'], step['d'], step['logp_u'], v)
        #         # if(step['d']):
        #         #     print("\n done at step", step['d'])
        #         ep_ret += step['r']
        #         ep_len += 1
        #
        #     # trajectory_returns = torch.cat((trajectory_returns, returns.mean().reshape(1)))
        # episode_lengths.append(trajectory_lengths.mean())
        #
        # all_returns.append(ep_ret)
        #
        #
        # ## Step 4 & 5: Compute rewards-to-go and advantages
        # returns, advantages = compute_returns_advantages(buffer.rewards, \
        #                                                  buffer.values, \
        #                                                  buffer.dones, \
        #                                                  gamma)
        # 
        # # advantages = advantages * 1e8
        # episode_returns.append(returns.mean().item())
        # episode_max_advantages.append(advantages.max().item())
        #
        #
        # ## Step 6 & 7 Update policy and value networks
        # update_actor(actor, 
        #               actor_optimizer, 
        #               buffer,
        #               advantages, 
        #               clip_param,
        #               epochs=actor_update_epochs)
        #
        # update_critic(critic, 
        #              critic_optimizer, 
        #              buffer,
        #              returns,
        #              epochs=critic_update_epochs)
        #
        # # env.reset()
        # buffer.clear()
        #
        # # print("\n trajectory_returns.mean():", trajectory_returns.mean())
        # # print("\n trajectory_returns:", trajectory_returns)

        # episode_returns.append(trajectory_returns.mean())
        # print("\n episode:", episode, "\n episode_return:", episode_returns[-1], "\n episode_advantage.max:", episode_max_advantages[-1], "\n ep_ret: ", ep_ret, "\n ep_len: ", ep_len)


            
        # # Do it repeatedly until a certain total number of time steps
        # while total_steps < steps_per_episode:
        #     # Rollout one trajectory, for at most the remaining steps of this episode
        #     trajectory = rollout(env, actor, steps_per_episode - total_steps)
        #     total_steps += len(trajectory)
        #     num_traj_per_episode += 1
        #
        #     for step in trajectory:
        #         # Ask critic to estimate the value of this state
        #         v = critic(torch.tensor(step['xp'], dtype=torch.float32).unsqueeze(0)).item()
        #         # Then store this step in the data buffer
        #         buffer.store(step['xp'], step['u'], step['r'], step['d'], step['logp_u'], v)
        #         # episode_return += step['r']
        #         episode_return += step['r']
        #         episode_length += 1
        #
        #
        #     if len(buffer.states) >= batch_size:
        #
        #         ## Step 4 & 5: Compute rewards-to-go and advantages
        #         returns, advantages = compute_returns_advantages(buffer.rewards, buffer.values, buffer.dones, gamma)
        #         # print("\n returns.mean():", returns.mean(), " advantages.mean():", advantages.mean())
        #         env.reset()
        #
        #         ## Step 6 & 7 Update policy and value networks
        #         update_actor(actor, 
        #                       actor_optimizer, 
        #                       buffer,
        #                       advantages, 
        #                       clip_param,
        #                       epochs=actor_update_epochs)
        #
        #         update_critic(critic, 
        #                      critic_optimizer, 
        #                      buffer,
        #                      returns,
        #                      epochs=critic_update_epochs)
        #
        #         # if total_steps % 10000 == 0:
        #         #     ## Visualize new policy
        #         #     policy = lambda x: actor(torch.tensor(x, dtype=torch.float32).
        #         #                 unsqueeze(0))[0].detach().numpy()
        #         #     viewer.launch(env, policy=policy)
        #
        #         buffer.clear()
        #
        # # episode_rewards.append(episode_return)
        # # episode_rewards.append(returns.mean())
        # episode_returns.append(episode_return)
        # episode_lengths.append(episode_length)
        # 
        # # moving_avg_rewards.append(np.mean(episode_rewards[-10:]))
        # # print("\n num_traj_per_episode: ", num_traj_per_episode,\
        # #       " returns.mean():", returns.mean(), \
        # #       " moving_avg_reward:", moving_avg_rewards[-1])






    return actor_critic, episode_returns, episode_lengths






# """
# Setup walker environment
# """
# r0 = np.random.RandomState(42)
# e = suite.load('walker', 'walk',
#                  task_kwargs={'random': r0})
# U=e.action_spec();udim=U.shape[0];
# X=e.observation_spec();xdim=14+1+9;
#
#
# # """
# # Visualize a random controller
# def u(dt):
#     return np.random.uniform(low=U.minimum,
#                              high=U.maximum,
#                                 size=U.shape)
# # viewer.launch(e,policy=u)
# # """
#
#
# # Example rollout using a network
# uth=uth_t(xdim,udim)
# traj=rollout(e,uth)
# # print("\n len(traj):", len(traj))

# Setup the environment

r0 = np.random.RandomState(42)
env = suite.load("walker", "walk", task_kwargs={'random': False})
## this was wrong, below is correct way of instantiation env


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
episodes = 100 
steps_per_episode = 1000

# Initialize actor and critic networks
# actor = Actor(xdim=state_dim, udim=action_dim, hdim=hidden_dim_actor)
# critic = Critic(state_dim, hdim=hidden_dim_critic)

ac_kwargs = {}
actor_critic = MLPActorCritic(state_dim, action_dim, **ac_kwargs)

# Run the training loop
actor_critic, episode_returns, episode_lengths = ppo_train(env, 
                                                           actor_critic, 
                                                           episodes,
                                                           steps_per_episode, 
                                                           gamma,
                                                           lam,
                                                           clip_param, 
                                                           actor_lr, 
                                                           critic_lr, 
                                                           actor_update_epochs, 
                                                           critic_update_epochs)

print("Training completed!")


# visualize_policy(env, actor_critic.pi)

# Plotting
plt.ion()
plt.figure(figsize=(10, 5))
plt.plot(episode_returns, label='Episode Return')
# plt.plot(episode_lengths, label='Episode Length')
# plt.plot(moving_avg_rewards, label='Moving Average (10 episodes)', linestyle='--')
plt.xlabel('Episodes')
# plt.ylabel('Total Return')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show(block=True)

# dummy data to make sure plt is working
# plt.figure(2)
# plt.plot([1, 2, 3, 4])
# plt.show()





