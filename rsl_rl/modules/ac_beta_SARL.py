#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, log_softmax
from torch.distributions import Beta
import math


def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self_state_mlp_layer = [128]
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.self_state_embedding_mlp = mlp(self_state_dim, self_state_mlp_layer)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim  # self_state_mlp_layer[-1]  #
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        assert not torch.isnan(state).any(), "NaN detected in input state before mlp1!"
        assert not torch.isinf(state).any(), "Inf detected in input state before mlp1!"
        
        # state += 1e-6
        # print(state)

        mlp1_output = self.mlp1(state.reshape((-1, size[2])))
        assert not torch.isnan(mlp1_output).any(), "NaN detected in mlp1 output!"

        mlp2_output = self.mlp2(mlp1_output)
        assert not torch.isnan(mlp2_output).any(), "NaN detected in mlp2 output!"

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.reshape(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).contiguous().reshape(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).reshape(size[0], size[1], 1).squeeze(dim=2)
        assert not torch.isnan(scores).any(), "NaN detected in attention output!"
        assert not torch.isinf(scores).any(), "Inf detected in attention output!"

        # masked softmax
        weights_softmax = softmax(scores, dim=1).unsqueeze(2)
        
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights_non_shifted = (scores_exp / (torch.sum(scores_exp, dim=1, keepdim=True) + 1e-8)).unsqueeze(2)

        max_values, _ = torch.max(scores, dim=-1, keepdim=True)
        scores = scores - max_values
        scores_exp = torch.exp(scores) * (scores != 0).float()
        assert not torch.isnan(scores_exp).any(), "NaN detected in scores_exp output!"
        assert not torch.isinf(scores_exp).any(), "Inf detected in scores_exp output!"
        weights = (scores_exp / (torch.sum(scores_exp, dim=1, keepdim=True) + 1e-8)).unsqueeze(2)

        # print(f"weights_softmax (torch) maximum is: {weights_softmax.max().item()}")
        # print(f"weights maximum is: {weights.max().item()}")
        # print(f"weights_non_shifted maximum is: {weights_non_shifted.max().item()}")
        # scores = scores - torch.max(scores, dim=-1)
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights_softmax[0, :, 0].data.cpu().numpy()
        assert not np.isnan(self.attention_weights).any(), "NaN detected in attention_weights!"

        # output feature is a linear combination of input features
        features = mlp2_output.reshape(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights_softmax, features), dim=1)
        assert not torch.isnan(weighted_feature).any(), "NaN detected in weighted_feature output!"

        # TODO: @vairaviv tried to bring robot state into latent space instead of directly feeding it to mlp
        # self_state = self.self_state_embedding_mlp(self_state)
        
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        assert not torch.isnan(joint_state).any(), "NaN detected in joint_state output!"

        value = self.mlp3(joint_state)
        assert not torch.isnan(value).any(), "NaN detected in mlp3 output!"

        return value


class ActorCriticBetaSARL(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_humans,
        robot_state_dim,
        human_state_dim=5,
        local_map_dim=3,
        cell_size=1.0,
        cell_num=4**2,
        with_global_state=True,
        mlp1_dims=[256, 256, 256],
        mlp2_dims=[256, 256, 256],
        mlp3_dims=[256, 256, 256],
        attention_dims=[256, 256, 256],
        beta_initial_logit=0.5,  # centered mean intially
        beta_initial_scale=5.0,  # sharper distribution initially
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actor_obs = robot_state_dim + human_state_dim + cell_num * local_map_dim # for three layers of information in the local map
        self.num_humans = num_humans
        actor_mlp3_dims = mlp3_dims.copy()
        actor_mlp3_dims[-1] = num_actions * 2
        self.actor = ValueNetwork(
            input_dim=self.num_actor_obs,
            self_state_dim=robot_state_dim,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            mlp3_dims=actor_mlp3_dims,
            attention_dims=attention_dims,
            with_global_state=with_global_state,
            cell_size=cell_size,
            cell_num=cell_num,
        )
        self.critic = ValueNetwork(
            input_dim=self.num_actor_obs,
            self_state_dim=robot_state_dim,
            mlp1_dims=mlp1_dims,
            mlp2_dims=mlp2_dims,
            mlp3_dims=mlp3_dims,
            attention_dims=attention_dims,
            with_global_state=with_global_state,
            cell_size=cell_size,
            cell_num=cell_num,
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.distribution = Beta(1, 1)
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        self.beta_initial_logit_shift = math.log(beta_initial_logit / (1.0 - beta_initial_logit))  # inverse sigmoid
        self.beta_initial_scale = beta_initial_scale
        self.output_dim = num_actions

        # disable args validation for speedup
        Beta.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def std(self):
        return self.distribution.stddev

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_beta_parameters(self, logits):
        """Get alpha and beta parameters from logits"""
        ratio = self.sigmoid(logits[..., : self.output_dim] + self.beta_initial_logit_shift)
        sum = (self.soft_plus(logits[..., self.output_dim :]) + 1) * self.beta_initial_scale

        # Compute alpha and beta
        alpha = ratio * sum
        beta = sum - alpha

        # Nummerical stability
        alpha += 1e-6
        beta += 1e-4
        return alpha, beta

    def update_distribution(self, observations):
        """Update the distribution of the policy"""
        if self.num_humans > observations.shape[0]:
            raise ValueError(
                f"""[ERROR]: Provided num_humans: {self.num_humans} in 'PPOBaseBetaSOADRLCfg' is greater than num_env deployed.
                            Please check the config file or increase the num_envs."""
            )
        #observations = observations.reshape(-1, self.num_humans, self.num_actor_obs)
        observations = observations.reshape(observations.shape[0], -1, self.num_actor_obs)
        if observations.shape[1] != self.num_humans:
            raise ValueError(
                f"""[ERROR]: Provided num_humans: {self.num_humans} in 'PPOBaseBetaSOADRLCfg' is not equal to the number of humans in the observation.
                            Please check the config file or the observation config, got {observations.shape[1]} humans in the observation."""
            )

        logits = self.actor(observations)
        alpha, beta = self.get_beta_parameters(logits)

        # Update distribution
        self.distribution = Beta(alpha, beta, validate_args=False)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # observations = observations.reshape(-1, self.num_humans, self.num_actor_obs)
        observations = observations.reshape(observations.shape[0], -1, self.num_actor_obs)
        logits = self.actor(observations)
        actions_mean = self.sigmoid(logits[:, : self.output_dim] + self.beta_initial_logit_shift)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        critic_observations = critic_observations.reshape(-1, self.num_humans, self.num_actor_obs)
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
