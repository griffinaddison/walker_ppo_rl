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

from tqdm import tqdm

class RolloutBuffer:
    def __init__(self, action_dim):
        self.states = torch.empty((0, 24), dtype=torch.float32)
        self.actions = torch.empty((0, action_dim), dtype=torch.float32)
        self.rewards = torch.empty((0, 1), dtype=torch.float32)
        self.dones = torch.empty((0, 1), dtype=torch.float32)
        self.log_probs = torch.empty((0, 1), dtype=torch.float32)
        self.values = torch.empty((0, 1), dtype=torch.float32)

    def store(self, state, action, reward, done, log_prob, value):
        # Ensure each new data point is a row vector of the appropriate dimensions
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Reshape to 1x24 if not already
        # action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
        log_prob = torch.tensor([log_prob], dtype=torch.float32).unsqueeze(0)
        value = torch.tensor([value], dtype=torch.float32).unsqueeze(0)

        # Concatenate along the first dimension (adding rows)
        self.states = torch.cat((self.states, state), dim=0)
        self.actions = torch.cat((self.actions, action), dim=0)
        self.rewards = torch.cat((self.rewards, reward), dim=0)
        self.dones = torch.cat((self.dones, done), dim=0)
        self.log_probs = torch.cat((self.log_probs, log_prob), dim=0)
        self.values = torch.cat((self.values, value), dim=0)

        # self.states = torch.cat((self.states, torch.tensor(state, dtype=torch.float32)), dim=0)
        # self.actions = torch.cat((self.actions, torch.tensor(action, dtype=torch.float32)), dim=0)
        # self.rewards = torch.cat((self.rewards, torch.tensor([reward], dtype=torch.float32)), dim=0)
        # self.dones = torch.cat((self.dones, torch.tensor([done], dtype=torch.float32)), dim=0)
        # print("\n log_prob:", log_prob)
        # self.log_probs = torch.cat((self.log_probs, torch.tensor([log_prob], dtype=torch.float32)), dim=0)
        # self.values = torch.cat((self.values, torch.tensor([value], dtype=torch.float32)), dim=0)

    def data(self):
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "values": self.values,
        }

    def clear(self):
        self.states = torch.tensor([], dtype=torch.float32)
        self.actions = torch.tensor([], dtype=torch.float32)
        self.rewards = torch.tensor([], dtype=torch.float32)
        self.dones = torch.tensor([], dtype=torch.float32)
        self.log_probs = torch.tensor([], dtype=torch.float32)
        self.values = torch.tensor([], dtype=torch.float32)


    def get_shuffled_data(self):
       
        # # Retrieve data as dictionary of tensors
        # data = self.data()
        #
        # # Determine the length of the data
        # num_entries = self.states.size(0)
        # print("\n num_entries:", num_entries)
        # print("\n self.states.size():", self.states.size())
        #
        # # Create a permutation of indices to shuffle the data
        # shuffled_indices = torch.randperm(num_entries)

        # Shuffle each tensor in the dictionary using the shuffled indices



        shuffled_data = {key: val[torch.randperm(val.size(0))] for key, val in self.data().items()}

        return shuffled_data




## U = ACTOR = POLICY
class Actor(nn.Module):
    def __init__(s,xdim,udim,
                 hdim=32,fixed_var=True):
        super().__init__()
        s.xdim,s.udim = xdim, udim
        ## Bool that disables learning of variance
        s.fixed_var=fixed_var

        ### TODO

        ## Define layers
        s.fc1 = nn.Linear(xdim, hdim)
        s.fc2 = nn.Linear(hdim, hdim)
        s.output = nn.Linear(hdim, udim)

        if not s.fixed_var:
            # If variance not fixed, learn it as a separate parameter
            s.log_std_layer = nn.Parameter(torch.zeros(udim) - 0.5) # 0.5 so well-formed

        ### END TODO

    def forward(s,x):
        ### TODO

        x = F.relu(s.fc1(x))
        x = F.relu(s.fc2(x))

        mean = s.output(x)

        if s.fixed_var:
            std = torch.exp(torch.zeros_like(mean))
        else:
            std = torch.exp(s.log_std_layer)

        
        pi = Normal(mean, std)
        action = pi.sample()
        log_probs = pi.log_prob(action) # log prob of each component of the action
        log_prob = log_probs.sum()
        ### END TODO
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super(Critic, self).__init__()

        ## Input of state
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        ## Output of 1 (it outputs value given state)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.output(x)
        return value




def rollout(e,actor,T=1000, render=False):
    """
    e: environment
    actor: controller
    T: time-steps
    """

    traj=[]
    t=e.reset()
    x=t.observation
    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
    for _ in range(T):
        with torch.no_grad():
            ## Sample action from pytorch nn
            u, log_prob_u = actor(torch.from_numpy(x).float().unsqueeze(0))
        ## Step environment using this action
        r = e.step(u.numpy())
        x=r.observation
        xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

        t=dict(xp=xp,r=r.reward,u=u, logp_u=log_prob_u, d=r.last())
        traj.append(t)
        x=xp
        if r.last():
            break
    return traj





def compute_returns_advantages(rewards, values, dones, gamma):
    """ Compute advantages with simple formla (not GAE) """


    # num_steps = len(rewards)
    # returns = torch.zeros(num_steps)
    # advantages = torch.zeros(num_steps)
    #
    # # Start with the last reward, where the next value is 0 since it's the end of the episode
    # next_value = 0
    # 
    # # Reverse loop through rewards to accumulate sums
    # for t in reversed(range(num_steps)):
    #     if dones[t]:
    #         next_value = 0  # If the current step is terminal, next value should be reset
    #
    #     # The return at time t is the reward at time t plus discount * next value
    #     returns[t] = rewards[t] + gamma * next_value
    #     next_value = returns[t]  # Update next_value to the return at time t
    #
    #     # The advantage is the return at time t minus the value estimate at time t
    #     advantages[t] = returns[t] - values[t]
    #
    # return returns, advantages
    lam = 0.95
    gamma = 0.99

    num_steps = len(rewards)
    returns = torch.zeros(num_steps)
    advantages = torch.zeros(num_steps)
    next_value = 0
    gae = 0  # Generalized advantage estimation

    for t in reversed(range(num_steps)):
        # If the state is terminal, next_value and gae reset
        if dones[t]:
            next_value = 0
            gae = 0
        # Delta is the TD residual
        delta = rewards[t] + gamma * next_value - values[t]
        # Update gae
        gae = delta + gamma * lam * gae
        # Store the calculated advantage
        advantages[t] = gae
        # The return is just the value plus the advantage
        returns[t] = advantages[t] + values[t]
        # Update next_value to be the current value estimate
        next_value = values[t]

    return returns, advantages

def update_actor(actor, actor_optimizer, buffer, advantages, clip_param, update_epochs, batch_size):
    for epoch in range(update_epochs):
        # Shuffle the data at the beginning of each epoch
        data = buffer.get_shuffled_data()
        
        for batch_idx in range(0, len(data['states']), batch_size):
            # Extract batches correctly handling the last batch which might be smaller than batch_size
            batch_states = data['states'][batch_idx:batch_idx+batch_size]
            batch_actions = data['actions'][batch_idx:batch_idx+batch_size]
            batch_old_log_probs = data['log_probs'][batch_idx:batch_idx+batch_size]
            batch_advantages = advantages[batch_idx:batch_idx+batch_size]
            
            # Get log probs from new actor
            _, new_log_probs = actor(batch_states) # Pytorch is able to batch these
            
            # Calculate the ratio of the new and old probabilities
            ratios = torch.exp(new_log_probs - batch_old_log_probs)
            
            # Clipping method
            # print("\n ratios.size():", ratios.size())
            # print("\n batch_advantages:", batch_advantages)
            
            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # Take the negative min of surr1 and surr2 for maximization
            
            ## Optimize the actor/policy based on its previous loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

def update_critic(critic, critic_optimizer, buffer, returns, update_epochs, batch_size):
    # Fetch data once to avoid repeated processing
    buffer_data = buffer.data()
    states = buffer_data['states']
    num_data = states.size(0)

    for _ in range(update_epochs):
        # Shuffle indices to ensure random batches
        indices = torch.randperm(num_data)

        ## For each batch
        for start_idx in range(0, num_data, batch_size):
            end_idx = min(start_idx + batch_size, num_data)  # Ensure we do not go out of bounds
            batch_indices = indices[start_idx:end_idx]

            batch_states = states[batch_indices]
            batch_returns = returns[batch_indices]

            # Predict the value for each state in the batch
            value_preds = critic(batch_states).squeeze(-1)  # Squeeze the last dimension if necessary

            # Check if the batch is empty (important in edge cases)
            if batch_states.size(0) == 0:
                continue

            # Calculate loss
            value_loss = F.mse_loss(value_preds, batch_returns)

            # Optimize the critic
            critic_optimizer.zero_grad()
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
        action, _ = actor(obs_tensor)
        action = action.detach().numpy().squeeze()  # Remove batch dimension
        
        # Debug: Check if action contains any None values
        if np.any(np.isnan(action)):
            print("NaN detected in action output")
        
        return action

    for _ in range(num_episodes):
        viewer.launch(env, policy)


        # time_step = env.reset()
        # while not time_step.last():
        #     action, _ = actor(torch.tensor(time_step.observation['observations'], dtype=torch.float32).unsqueeze(0))
        #     time_step = env.step(action.detach().numpy().squeeze())
        #     env.render()  # Make sure your environment supports rendering

def ppo_train(env, actor, critic, episodes, steps_per_episode, gamma=0.99, 
              clip_param=0.2, actor_lr=1e-3, critic_lr=1e-3, update_epochs=10, 
              batch_size=64):
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

    # get action dim
    action_dim = env.action_spec().shape[0]
    buffer = RolloutBuffer(action_dim)


    episode_rewards = []
    moving_avg_rewards = []

    
    for episode in tqdm(range(episodes)):

        
        # if episode % 10 == 0:
        #     print("\n episode:", episode)
        #     print("\n episode_length:", len(episode_rewards))
        #     # visualize the policy
        #     policy = lambda x: actor(torch.tensor(x, dtype=torch.float32).
        #                 unsqueeze(0))[0].detach().numpy()
        #     viewer.launch(env, policy=policy)

        if episode % 50000 == 0 and episode > 0:
            print(f"Visualizing policy at episode {episode}")
            visualize_policy(env, actor)

        episode_return = 0
        ## Step 3: Rollout current policy 

        total_steps = 0
        num_traj_per_episode = 0

        # Do it repeatedly until a certain total number of time steps
        while total_steps < steps_per_episode:
            # Rollout one trajectory, for at most the remaining steps of this episode
            trajectory = rollout(env, actor, steps_per_episode - total_steps)
            total_steps += len(trajectory)
            num_traj_per_episode += 1

            for step in trajectory:
                # Ask critic to estimate the value of this state
                v = critic(torch.tensor(step['xp'], dtype=torch.float32).unsqueeze(0)).item()
                # Then store this step in the data buffer
                buffer.store(step['xp'], step['u'], step['r'], step['d'], step['logp_u'], v)
                episode_return += step['r']


            if len(buffer.states) >= batch_size:

                ## Step 4 & 5: Compute rewards-to-go and advantages
                returns, advantages = compute_returns_advantages(buffer.rewards, buffer.values, buffer.dones, gamma)
                ## Step 6 & 7 Update policy and value networks
                update_actor(actor, 
                              actor_optimizer, 
                              buffer,
                              advantages, 
                              clip_param, 
                              update_epochs, 
                              batch_size)

                update_critic(critic, 
                             critic_optimizer, 
                             buffer,
                             returns,
                             update_epochs,
                             batch_size)

                # if total_steps % 10000 == 0:
                #     ## Visualize new policy
                #     policy = lambda x: actor(torch.tensor(x, dtype=torch.float32).
                #                 unsqueeze(0))[0].detach().numpy()
                #     viewer.launch(env, policy=policy)

                buffer.clear()

        episode_rewards.append(episode_return)
        
        moving_avg_rewards.append(np.mean(episode_rewards[-10:]))
        print("\n num_traj_per_episode: ", num_traj_per_episode,\
              " episode_reward:", episode_return, \
              " moving_avg_reward:", moving_avg_rewards[-1])





    return actor, critic, episode_rewards, moving_avg_rewards






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
env = suite.load(domain_name="walker", task_name="walk")

# Parameters
state_dim = 14+1+9  # Adapt based on your specific environment
action_dim = env.action_spec().shape[0]
hidden_dim = 64  # You can adjust this
gamma = 0.99
clip_param = 0.2
actor_lr = 1e-0
critic_lr = 1e-0
update_epochs = 10
batch_size = 64
episodes = 100 
steps_per_episode = 2048

# Initialize actor and critic networks
actor = Actor(xdim=state_dim, udim=action_dim, hdim=hidden_dim, fixed_var=False)
critic = Critic(state_dim, hidden_dim)

# Initialize optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# Run the training loop
actor, critic, episode_rewards, moving_avg_rewards = ppo_train(env, actor, critic, episodes, steps_per_episode, gamma, 
                          clip_param, actor_lr, critic_lr, update_epochs, batch_size)

print("Training completed!")


visualize_policy(env, actor)

# Plotting
plt.ion()
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, label='Episode Return')
# plt.plot(moving_avg_rewards, label='Moving Average (10 episodes)', linestyle='--')
plt.xlabel('Episodes')
plt.ylabel('Total Return')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show(block=True)

# dummy data to make sure plt is working
plt.figure(2)
plt.plot([1, 2, 3, 4])
plt.show()





