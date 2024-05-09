# CODE IS MODIFIED FROM OPENAI SPINNINGUP ppo.ppy IMPLEMENTATION

import numpy as np
from dm_control import suite, viewer
from models import MLPActorCritic
import torch
from torch.optim import Adam
from models import *
from tqdm import tqdm
from matplotlib import pyplot as plt

def process_trajectories(traj):
    # Preallocate lists to collect the data
    obs_list = []
    act_list = []
    adv_list = []
    logp_old_list = []
    ret_list = []

    # Iterate through each step in the trajectory
    for step in traj:
        obs_list.append(step['obs'])  # Assuming each obs is already a 24-dimensional vector
        act_list.append(step['action'])
        adv_list.append(step['adv'])
        logp_old_list.append(step['logp'])
        ret_list.append(step['ret'])

    # Convert lists to numpy arrays
    obs = np.array(obs_list, dtype=np.float32)  # Shape will naturally be [n, 24] if each obs is 24-dim
    act = np.array(act_list, dtype=np.float32)  # Shape [n,], assuming actions are scalar or adjust dtype accordingly
    adv = np.array(adv_list, dtype=np.float32)  # Shape [n,]
    logp_old = np.array(logp_old_list, dtype=np.float32)  # Shape [n,]
    ret = np.array(ret_list, dtype=np.float32)  # Shape [n,]

    return obs, act, adv, logp_old, ret

def ppo(actor_critic=MLPActorCritic, ac_kwargs=dict(), 
        local_steps_per_epoch=1000, epochs=500, gamma=0.95, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lambda_=0.97,
        target_kl=0.01):

    # Setup walker environment
    r0 = np.random.RandomState(42)
    env = suite.load('walker', 'walk', task_kwargs={'random': False})
    U = env.action_spec(); udim = U.shape[0]
    X = env.observation_spec(); xdim = 14 + 1 + 9  # Adjusted to match observation space

    ac = actor_critic(xdim, udim, **ac_kwargs)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        act_t = torch.as_tensor(act, dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        logp_old_t = torch.as_tensor(logp_old, dtype=torch.float32)

        pi, logp = ac.pi(obs_t, act_t)
        ratio = torch.exp(logp - logp_old_t)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t
        loss_pi = -(torch.min(ratio * adv_t, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old_t - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(obs, ret):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        return ((ac.v(obs_t) - ret_t)**2).mean()


    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update(traj):
        obs, act, adv, logp_old, ret = process_trajectories(traj)

        pi_l_old, pi_info_old = compute_loss_pi(obs, act, adv, logp_old)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(obs, ret).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(obs, act, adv, logp_old)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(obs, ret)
            loss_v.backward()
            vf_optimizer.step()

    ave_returns = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(epochs), desc="Epochs"):
        all_returns = []

        for step in tqdm(range(local_steps_per_epoch), desc="Steps per Epoch", leave=False):
            traj = []
            obs = env.reset().observation
            x = np.concatenate([obs['orientations'], [obs['height']], obs['velocity']])
            ep_ret, ep_len = 0,0
            while True: 
                with torch.no_grad():
                    a, v, logp = ac.step(torch.as_tensor(x, dtype=torch.float32))
                r = env.step(a)
                # Process new observation
                next_x = np.concatenate([r.observation['orientations'], [r.observation['height']], r.observation['velocity']])
                
                # Store the transition
                traj.append({
                    'obs': x,
                    'action': a,
                    'reward': r.reward,
                    'done': r.last(),
                    'logp': logp,
                    'value': v
                })
                ep_ret += r.reward
                ep_len += 1
        
                # Update observation
                x = next_x

                # Check if the episode has ended
                if r.last() or ep_len >= local_steps_per_epoch:
                    all_returns.append(ep_ret)

                    # Compute the returns-to-go for each state
                    returns_to_go = np.zeros(len(traj))
                    current_return = 0
                    for i in reversed(range(len(traj))):
                        current_return = traj[i]['reward'] + gamma * current_return * (1 - traj[i]['done'])
                        returns_to_go[i] = current_return

                    # Compute the advantage using GAE
                    advantages = np.zeros_like(returns_to_go)
                    gae = 0
                    for i in reversed(range(len(traj))):
                        delta = traj[i]['reward'] + gamma * (traj[i+1]['value'] if i+1 < len(traj) else 0) * (1 - traj[i]['done']) - traj[i]['value']
                        gae = delta + gamma * lambda_ * gae * (1 - traj[i]['done'])
                        advantages[i] = gae

                    # Add computed values to traj for later use
                    for i, data in enumerate(traj):
                        data['ret'] = returns_to_go[i]
                        data['adv'] = advantages[i]

                    break

        # After finishing all steps in the epoch, calculate the average return
        average_return = np.mean(all_returns) if all_returns else 0
        ave_returns.append(average_return)
        if epoch % 10 == 0:
            print(f"\nAverage return for epoch {epoch}: {average_return:.2f}")
        
        # Perform PPO update!
        update(traj)

        if epoch % 100 == 0:
            torch.save(ac.state_dict(), f'actor_critic_{epoch}.pth')
            print(f'\nModel saved at epoch {epoch}/500!')

        

    
    print("\nTraining completed!")
    return ave_returns, ac.state_dict()


ppo(actor_critic=MLPActorCritic, ac_kwargs=dict(), 
        local_steps_per_epoch=1000, epochs=500, gamma=0.95, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lambda_=0.97,
        target_kl=0.01)
