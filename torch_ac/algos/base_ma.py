from abc import ABC, abstractmethod
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np

class BaseAlgoMA(ABC):
    """The base class for RL algorithms for multiple agents."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        self.n_agents = self.env.envs[0].n_agents
        
        # 128 x 16
        shape = (self.num_frames_per_proc, self.num_procs)
        # 128 x 10 x 2, 128 x 32
        new_shape = (self.num_frames_per_proc, self.num_procs, self.n_agents)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            # 10 x 2 x 2048 -> 20 x 2048
            self.memory = torch.zeros(new_shape[1]*new_shape[2], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*new_shape, self.acmodel.memory_size, device=self.device)
            # 400 x 10 x 2 x 2048, 400 x 20 x 2048 -> 8000 x 2048

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)

        self.actions = torch.zeros(*new_shape, device=self.device, dtype=torch.int)
        self.log_probs = torch.zeros(*new_shape, device=self.device)
        
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, self.n_agents, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, self.n_agents, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def reset_memory(self):

        new_shape = (self.num_frames_per_proc, self.num_procs, self.n_agents)
        if self.acmodel.recurrent:
            # 10 x 2 x 2048 -> 20 x 2048
            self.memory = torch.zeros(new_shape[1]*new_shape[2], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*new_shape, self.acmodel.memory_size, device=self.device)        

    def repeat_experience_per_agent(self, x):
        # 128 x 16
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, self.n_agents)
        # 128 x 16 x n
        return x

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            # 32x7x7x6, 16x2x7x7x6                   #16x2x7x7x6
            agent_obs, env_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                if self.acmodel.recurrent:
                    # 10 x 2 x 2048 , 20 x 2048       
                    # dist, value, memory = self.acmodel(agent_obs, env_obs, self.memory * self.mask.unsqueeze(1))
                    # tmp = torch.einsum('ijk,ij->ijk', self.memory, self.mask.unsqueeze(1).repeat(1, self.n_agents))
                    tmp = (self.mask.unsqueeze(1).repeat(1, self.n_agents)).reshape(-1).unsqueeze(1) # [20]
                    dist, value, memory = self.acmodel(agent_obs, env_obs, self.memory*tmp)
                else:
                    dist, value = self.acmodel(agent_obs, env_obs)
            action = dist.sample()
            action = action.reshape(self.num_procs, self.n_agents)
            
            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory.reshape(-1, self.n_agents, self.memory.shape[-1])
                self.memory = memory
            # import ipdb; ipdb.set_trace()
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.sum(torch.tensor(reward, device=self.device), dim=1)

            action = action.reshape(-1)

            # log of action probabilities corresponding to each selected action
            log_probs = dist.log_prob(action)
            log_probs = log_probs.reshape(self.num_procs, self.n_agents)
            self.log_probs[i] = log_probs

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones((self.num_procs, self.n_agents), device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    # self.log_return.append(self.log_episode_return[i].item())
                    # self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    # self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    self.log_return.append(torch.sum(self.log_episode_return[i]).item())
                    self.log_reshaped_return.append(torch.sum(self.log_episode_reshaped_return[i]).item())
                    self.log_num_frames.append(torch.sum(self.log_episode_num_frames[i]).item())
                    
            mask_tmp = self.mask.unsqueeze(1).repeat(1,self.n_agents).reshape(self.num_procs, self.n_agents)
            self.log_episode_return *= mask_tmp
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= mask_tmp

            # self.log_episode_return *= self.mask
            # self.log_episode_reshaped_return *= self.mask
            # self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        agent_obs, env_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                tmp = (self.mask.unsqueeze(1).repeat(1, self.n_agents)).reshape(-1).unsqueeze(1) # [20]
                _, next_value, _ = self.acmodel(agent_obs, env_obs, self.memory*tmp)
            else:
                _, next_value = self.acmodel(agent_obs, env_obs)


        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.
        # 2 x 7 x 7 x 6 - 2048 

        # 7 x7 x 6 - 4096
        # 7 x7 x 6
        exps = DictList()
        # 2048 x 2 x 7 x7 x6
        # import ipdb; ipdb.set_trace()
        obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        # exps.agent_obs = [self.obss[i][j][k]
        #             for k in range(self.n_agents)
        #             for j in range(self.num_procs)
        #             for i in range(self.num_frames_per_proc)]

        # exps.env_obs = [self.obss[i][j]
        #             for k in range(self.n_agents)
        #             for j in range(self.num_procs)
        #             for i in range(self.num_frames_per_proc)]

        # exp.env_obs = []
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D # 8000(10 x 2 x 400) x 2048
            exps.memory = self.memories.transpose(0, 1).reshape(-1, self.memories.shape[3])
            # T x P -> P x T -> (P * T) x 1
            tmp = self.masks.unsqueeze(-1).repeat(1, 1, self.n_agents)
            exps.mask = tmp.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T

        repeated_values = self.repeat_experience_per_agent(self.values)
        repeated_rewards = self.repeat_experience_per_agent(self.rewards)
        repeated_advantages = self.repeat_experience_per_agent(self.advantages)

        exps.action = self.actions.transpose(0, 1).reshape(-1)

        exps.value = repeated_values.transpose(0, 1).reshape(-1)
        exps.reward = repeated_rewards.transpose(0, 1).reshape(-1)
        exps.advantage = repeated_advantages.transpose(0, 1).reshape(-1)
        # exps.value = self.values.transpose(0, 1).reshape(-1)
        # exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        # exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        # 4096x7x7x6  , # 4096x2x7x7x6
        exps.agent_obs, exps.env_obs = self.preprocess_obss(obs, device=self.device)

        tmp = exps.env_obs.image.unsqueeze(-1)
        tmp = tmp.repeat(1, 1, 1, 1, 1, self.n_agents)
        tmp = tmp.permute(0, 5, 1, 2, 3, 4)
        tmp = tmp.flatten(start_dim=0, end_dim=1)
        exps.env_obs.image = tmp

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
