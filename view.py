from dm_control import suite, viewer
import numpy as np
from models import *
import mujoco
import torch as th

r0 = np.random.RandomState(42)
env = suite.load('walker', 'walk', task_kwargs={'random': False})
U = env.action_spec(); udim = U.shape[0]
X = env.observation_spec(); xdim = 14 + 1 + 9 

model = MLPActorCritic(xdim, udim)
state_dict = torch.load('actor_critic_400.pth')
model.load_state_dict(state_dict)

def u(dt):
    x = dt.observation
    x = np.concatenate([x['orientations'], [x['height']], x['velocity']])
    a, v, logp_a = model.step(torch.as_tensor(x, dtype=torch.float32))
    return a

viewer.launch(env, policy=u)
