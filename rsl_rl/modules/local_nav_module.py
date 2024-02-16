import torch.nn as nn
import numpy as np
import torch
from rsl_rl.modules.submodules.cnn_modules import Pool1DConv, SimpleCNN2
from rsl_rl.modules.submodules.mlp_modules import MLP
from rsl_rl.modules import EmpiricalNormalization


class SimpleNavPolicy(nn.Module):
    def __init__(self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        history_shape: list[int],
        height_map_shape: list[int],
        proprioceptive_shape: int,
        activation: str,
        scan_cnn_channels: list[int],
        scan_cnn_fc_shape: list[int],
        scan_latent_size: list[int],
        history_channels: list[int],
        history_fc_shape: list[int],
        history_latent_size: list[int],
        history_kernel_size: list[int],
        output_mlp_size: list[int],
        **kwargs
    ):
        super().__init__()

        self.h_map_shape = torch.tensor(height_map_shape)
        self.h_map_size = torch.prod(self.h_map_shape).item()
        self.prop_size = proprioceptive_shape
        self.history_shape = torch.tensor(history_shape)
        self.history_obs_size = torch.prod(self.history_shape).item()

        self.activation_fn = get_activation(activation)

        self.num_obs = self.h_map_size + self.prop_size + self.history_obs_size
        assert self.num_obs == num_actor_obs, "num_obs and num_actor_obs should be the same"

        self.action_size = num_actions

        self.num_obs_normalizer = self.h_map_size + self.prop_size

        self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs_normalizer], until=1.0e8)

        grid_map_length = int(np.sqrt(self.h_map_shape[-1]))
        self.scan_encoder = SimpleCNN2([self.h_map_shape[0], grid_map_length, grid_map_length],
                                        scan_cnn_channels,
                                        scan_cnn_fc_shape,
                                        scan_latent_size,
                                        self.activation_fn)


        self.history_encoder = Pool1DConv(self.history_shape,
                                          history_channels,
                                          history_fc_shape,
                                          history_latent_size,
                                          history_kernel_size,
                                          activation_fn=self.activation_fn)

        self.action_head = MLP(
            output_mlp_size,
            self.activation_fn,
            self.prop_size+ scan_latent_size + history_latent_size,
            self.action_size,
            init_scale=1.0 / np.sqrt(2),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_nor = self.obs_normalizer(obs[:,:self.num_obs_normalizer])
        obs = torch.cat((obs_nor, obs[:,self.num_obs_normalizer:]), dim=1)

        prop, scan,  history = self.split_obs(obs)

        # scan
        scan_latent = self.scan_encoder(scan)

        # history
        history_latent = self.history_encoder(history)

        concat_feature = torch.cat([scan_latent, prop, history_latent], dim=1)
        output = self.action_head(concat_feature)
        return output

    def split_obs(self, obs):
        return torch.split(obs, [self.prop_size, self.h_map_size, self.history_obs_size], dim=1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU
    elif act_name == "selu":
        return nn.SELU
    elif act_name == "relu":
        return nn.ReLU
    elif act_name == "crelu":
        return nn.ReLU
    elif act_name == "lrelu":
        return nn.LeakyReLU
    elif act_name == "tanh":
        return nn.Tanh
    elif act_name == "sigmoid":
        return nn.Sigmoid
    elif act_name == "softsign":
        return nn.Softsign
    else:
        print("invalid activation function!")
        return None