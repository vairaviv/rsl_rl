#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
from __future__ import annotations

import torch
import torch.nn as nn
import torch.jit

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from torch.distributions import Beta
import math


class MLP(nn.Module):
    def __init__(self, input_size, shape, actionvation_fn, init_scale=2.0, add_activation_fn_at_end: bool = False):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        if len(shape) == 1 and not add_activation_fn_at_end:
            modules = [nn.Linear(input_size, shape[0])]
        else:
            modules = [nn.Linear(input_size, shape[0]), self.activation_fn]
        
        scale = [init_scale]

        for idx in range(len(shape) - 1):
            modules.append(nn.Linear(shape[idx], shape[idx + 1]))
            if idx < len(shape)-2 or add_activation_fn_at_end:
                modules.append(self.activation_fn)
                scale.append(init_scale)

        self.architecture = nn.Sequential(*modules)
        scale.append(init_scale)

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [shape[-1]]

    def forward(self, x):
        return self.architecture(x)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))

        ]


class CNN2D(nn.Module):
    def __init__(
        self, 
        input_shape, 
        channels, 
        mlp_layers, 
        activation_fn, 
        kernels, 
        strides, 
        num_history_time_steps,
        max_pooling=False, 
        groups=1,
        add_activation_fn_at_end=False,
    ):
        super().__init__()
        # input_shape = input_shape.permute(0, 3, 2, 1)
        self.max_pooling = max_pooling
        modules = [nn.Conv2d(input_shape[0], channels[0], kernels[0], strides[0], groups=groups), activation_fn]

        if groups > 1:
            self.groups = groups
            self.num_history = num_history_time_steps
            self.sem_channels = int(input_shape[0] / self.num_history)
        for idx in range(len(channels) - 1):
            modules.append(nn.Conv2d(channels[idx], channels[idx + 1], kernels[idx + 1], strides[idx + 1], groups=groups))
            modules.append(activation_fn)
            if max_pooling:
                modules.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # modules.append(nn.Conv2d(channels[-1], channels[-1], 2, 2))

        self.input_shape = input_shape
        self.conv_module = nn.Sequential(*modules)

        dummy_input = torch.zeros(1, input_shape[0], input_shape[1], input_shape[2])
        dummy_output = self.conv_module(dummy_input).view(1, -1)

        self.fc = MLP(dummy_output.shape[1], mlp_layers, activation_fn, 1.0 / float(math.sqrt(2)), add_activation_fn_at_end)

    def forward(self, x):
        try:
            if self.groups > 1 and self.groups == self.input_shape[0]/self.num_history:
                # reshape and bring it into order for CNN to group the channels by the time
                x = x.reshape(-1, self.num_history, self.sem_channels, self.input_shape[1], self.input_shape[2])
                x = x.permute(0, 2, 1, 3, 4)
                x = x.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            elif self.groups > 1:
                ValueError(
                    f"""[ERROR]: Tried to group input Channels in CNN2D to {self.groups}, \n
                    but num_history is {self.num_history}, \n
                    and input_channels are {self.input_shape[0]}"""
                )
        except AttributeError:
            x = x.reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = self.conv_module(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x


def calculate_cnn_output_size(
    input_dim: int,
    layer_channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
):
    """
    Calculate the output size of a CNN created with FlexibleBatchCircularlPadConv2d layers.

    Args:
        input_dim (int): Length of the input sequence.
        layer_channels (list[int]): List of output channels for each layer.
        kernel_sizes (list[int]): List of kernel sizes for each layer.
        strides (list[int]): List of strides for each layer.

    Returns:
        int: Final output length after all layers.
    """
    current_length = input_dim  # Start with the input length

    # Get all layer configurations
    input_output_pairs = zip([input_dim] + layer_channels[:-1], layer_channels, kernel_sizes, strides)

    for _, _, kernel_size, stride in input_output_pairs:
        padding = kernel_size // 2  # for Circular padding, which ensures input length remains same
        current_length = (current_length + 2 * padding - kernel_size) // stride + 1

    return current_length


class ActorCriticBeta2DCNN(nn.Module):
    """A Network structure for training a navigation policy.
    
    possible input dimensions are:
    target_dim --> input to MLP
    cpg_dim --> input to MLP
    lidar_extra_dim --> input to MLP
    lidar_dim --> input to CNN
    lidar_history_dim --> input to CNN
    
    # Network structure:
    #                                                    _________
    #                                  Input:           |         |
    #                                  target_pos       |  MLP2   |
    #                                  cpg_state        |_________| 
    #                                                        |
    #                                                        |
    #                                                        |
    #                                                        |
    #                                                        v                                     
    #                  _________             _________                _________                 params Beta:
    # Input           |         |           |         |              |         |                alpha_vx,
    # semantic map    |  CNN    | --------> |  MLP1   |------------> |  MLP3   | -----> Output: beta_vx, 
    #                 |_________|           |_________|              |_________|                alpha_vy, 
    #                                                                                           beta_vy
    #                                                                                           alpha_theta,
    #                                                                                           beta_theta
        
    for beta distribution, this is predicted:
    Symmetric (alpha = beta):
    - If alpha = beta = 1: Uniform distribution over [0, 1].
    - If alpha = beta > 1: Bell-shaped curve centered at x = 0.5.
    - If alpha = beta < 1: U-shaped curve with higher density near 0 and 1.

    Asymmetric (alpha ≠ beta):
    - If alpha > beta: Skewed towards 1.
    - If alpha < beta: Skewed towards 0.
        
    """

    is_recurrent = False

    def __init__(
            self,
            num_actor_obs: dict,
            num_critic_obs: dict,
            num_actions: int,
            proprio_dim: int,
            semantic_map_dim: list[int],
            proprio_layer_dim: list[int],
            semantic_cnn_channel_dim: list[int],
            semantic_cnn_kernel_sizes: list[int],
            semantic_cnn_strides: list[int],
            semantic_cnn_to_mlp_layer_dim: list[int],
            nav_layer_dim: list[int],
            parallel_to_mlp_layer_dim: list[int] = [512],
            activation: str = "elu",
            num_history_time_steps: int = 1,
            parallel_CNN_process: bool = False,
            max_pooling: bool = False,
            group_channels: int = 1,
            beta_initial_logit: float = 0.5,  # centered mean initially
            beta_initial_scale: float = 5.0,  # sharper distribution initially
            **kwargs,
    ):
        
        # check if any additional arguments were given to the module
        if kwargs:
            print(
                "ActorCriticBeta.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.proprio_dim = proprio_dim
        self.cnn_input_shape = semantic_map_dim

        activation_module = get_activation(activation)

        # TODO @vairaviv assumption made that the NN is symmetric and num_actor_obs == num_critic_obs,
        # if changed this needs to be adapted

        ##
        # define networks
        ##
        if parallel_CNN_process:
            self.parallel_CNN_process = parallel_CNN_process
            self.num_history_time_steps = num_history_time_steps
            self.actor_semantic_embedding_cnn = []
            self.critic_semantic_embedding_cnn = []
            for i in range(num_history_time_steps):
                self.actor_semantic_embedding_cnn.append(
                    CNN2D(
                        input_shape=self.cnn_input_shape,
                        channels=semantic_cnn_channel_dim,
                        kernels=semantic_cnn_kernel_sizes,
                        strides=semantic_cnn_strides,
                        activation_fn=activation_module,
                        mlp_layers=semantic_cnn_to_mlp_layer_dim,
                        max_pooling=max_pooling,
                    )
                )
                self.critic_semantic_embedding_cnn.append(
                    CNN2D(
                        input_shape=self.cnn_input_shape,
                        channels=semantic_cnn_channel_dim,
                        kernels=semantic_cnn_kernel_sizes,
                        strides=semantic_cnn_strides,
                        activation_fn=activation_module,
                        mlp_layers=semantic_cnn_to_mlp_layer_dim,
                        max_pooling=max_pooling
                    )
                )
                # setattr(
                #     self, 
                #     f"actor_semantic_embedding_cnn_{i}", 
                #     CNN2D(
                #         input_shape=self.cnn_input_shape,
                #         channels=semantic_cnn_channel_dim,
                #         kernels=semantic_cnn_kernel_sizes,
                #         strides=semantic_cnn_strides,
                #         activation_fn=activation_module,
                #         mlp_layers=semantic_cnn_to_mlp_layer_dim
                #     )
                # )
                # setattr(
                #     self, 
                #     f"critic_semantic_embedding_cnn_{i}", 
                #     CNN2D(
                #         input_shape=self.cnn_input_shape,
                #         channels=semantic_cnn_channel_dim,
                #         kernels=semantic_cnn_kernel_sizes,
                #         strides=semantic_cnn_strides,
                #         activation_fn=activation_module,
                #         mlp_layers=semantic_cnn_to_mlp_layer_dim
                #     )
                # )
            setattr(
                self,
                "parallel_cnn_to_mlp",
                MLP(
                    input_size=num_history_time_steps * semantic_cnn_to_mlp_layer_dim[-1],
                    shape=[parallel_to_mlp_layer_dim],
                    actionvation_fn=activation_module,
                    init_scale=1.0 / float(math.sqrt(2))
                )
            )
            last_semantic_mlp_dim = parallel_to_mlp_layer_dim[-1]
                
        else:
            self.num_history_time_steps = num_history_time_steps
            if num_history_time_steps > 1:
                # extend channel dimension if maps are concatenated
                # semantic_cnn_channel_dim = [num_history_time_steps * dim for dim in semantic_cnn_channel_dim]
                # semantic_cnn_to_mlp_layer_dim = [num_history_time_steps * dim for dim in semantic_cnn_to_mlp_layer_dim]
                self.cnn_input_shape[0] *= num_history_time_steps
                
            # CNN for semantic map embedding
            self.actor_semantic_embedding_cnn = CNN2D(
                input_shape=self.cnn_input_shape,
                channels=semantic_cnn_channel_dim,
                kernels=semantic_cnn_kernel_sizes,
                strides=semantic_cnn_strides,
                activation_fn=activation_module,
                mlp_layers=semantic_cnn_to_mlp_layer_dim,
                max_pooling=max_pooling,
                num_history_time_steps=num_history_time_steps,
                groups=group_channels,
                add_activation_fn_at_end=True,
            )
            self.critic_semantic_embedding_cnn = CNN2D(
                input_shape=self.cnn_input_shape,
                channels=semantic_cnn_channel_dim,
                kernels=semantic_cnn_kernel_sizes,
                strides=semantic_cnn_strides,
                activation_fn=activation_module,
                mlp_layers=semantic_cnn_to_mlp_layer_dim,
                max_pooling=max_pooling,
                num_history_time_steps=num_history_time_steps,
                groups=group_channels,
                add_activation_fn_at_end=True,
            )
            last_semantic_mlp_dim = semantic_cnn_to_mlp_layer_dim[-1]


        # Proprio embedding
        self.actor_proprio_embedding_mlp = MLP(
            input_size=self.proprio_dim,
            shape=proprio_layer_dim,
            actionvation_fn=activation_module,
            add_activation_fn_at_end=True,
        )
        self.critic_proprio_embedding_mlp = MLP(
            input_size=self.proprio_dim,
            shape=proprio_layer_dim,
            actionvation_fn=activation_module,
            add_activation_fn_at_end=True,
        )
        
        # MLP for Navigation, with output defined for actor and critic separate
        actor_out_layers = nav_layer_dim + [num_actions * 2] # for the distribution needed (alpha and beta for each action)
        critic_out_layers = nav_layer_dim + [1] # this is just for the value function, just one value should be outputed
        
        self.actor_nav_mlp = MLP(
            input_size=proprio_layer_dim[-1] + last_semantic_mlp_dim,
            shape=actor_out_layers,
            actionvation_fn=activation_module
        )
        self.critic_nav_mlp = MLP(
            input_size=proprio_layer_dim[-1] + last_semantic_mlp_dim,
            shape=critic_out_layers,
            actionvation_fn=activation_module
        )

        # TODO: join to one actor and one critic to print the structure
        # print(f"Actor net: {self.actor}")

        # print(f"Critic net: {self.critic}")

        # print(f"num actor params: {sum(p.numel() for p in self.actor.parameters())}")
        # print(f"num critic params: {sum(p.numel() for p in self.critic.parameters())}")
        print(f"total num params: {sum(p.numel() for p in self.parameters())}")

        # Action noise
        self.distribution = Beta(1, 1)
        self.soft_plus = torch.nn.Softplus(beta=1)
        self.sigmoid = nn.Sigmoid()
        self.beta_initial_logit_shift = math.log(beta_initial_logit / (1.0 - beta_initial_logit))  # inverse sigmoid
        self.beta_initial_scale = beta_initial_scale
        # @vairaviv check if it should be num_acitons or the whole output dim of the actor network?
        # should be the actual num of actions
        # self.output_dim = actor_out_layers[-1]
        self.output_dim = num_actions

        # disable args validation for speedup
        Beta.set_default_validate_args = False

 
    @staticmethod
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
    
    def actor_forward(self, x, masks=None, hidden_states=None):
        proprio_obs = x[:, :self.proprio_dim]
        
        if hasattr(self, "parallel_CNN_process"):
            cnn_obs = x[:, self.proprio_dim:].reshape(x.shape[0], self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
            sem_embedded_future = [torch.jit.fork(cnn, cnn_obs[:, i, :, :]) for i, cnn in enumerate(self.actor_semantic_embedding_cnn)]
            sem_embedded = torch.stack([torch.jit.wait(f) for f in sem_embedded_future])
        else:
            try:
                cnn_obs = x[:, self.proprio_dim:].reshape(x.shape[0], self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
            except RuntimeError:
                raise RuntimeError(f"""check if your configurations are correct, most likely mistake in observations config,
                                  expected input shape {self.cnn_input_shape} but got {x.shape}""")
            sem_embedded = self.actor_semantic_embedding_cnn(cnn_obs)
        
        proprio_embedded = self.actor_proprio_embedding_mlp(proprio_obs)

        input_to_nav_mlp = torch.cat((sem_embedded, proprio_embedded), dim=-1)

        return self.actor_nav_mlp(input_to_nav_mlp)
    
    def critic_forward(self, x, masks=None, hidden_states=None):
        proprio_obs = x[:, :self.proprio_dim]
        
        if hasattr(self, "parallel_CNN_process"):
            cnn_obs = x[:, self.proprio_dim:].reshape(x.shape[0], self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
            sem_embedded_future = [torch.jit.fork(cnn, cnn_obs[:, i, :, :]) for i, cnn in enumerate(self.critic_semantic_embedding_cnn)]
            sem_embedded = torch.stack([torch.jit.wait(f) for f in sem_embedded_future])
        else:
            cnn_obs = x[:, self.proprio_dim:].reshape(x.shape[0], self.cnn_input_shape[0], self.cnn_input_shape[1], self.cnn_input_shape[2])
            sem_embedded = self.critic_semantic_embedding_cnn(cnn_obs)
        
        proprio_embedded = self.critic_proprio_embedding_mlp(proprio_obs)

        input_to_nav_mlp = torch.cat((sem_embedded, proprio_embedded), dim=-1)

        return self.critic_nav_mlp(input_to_nav_mlp)
    
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

    def update_distribution(self, observations, masks=None, hidden_states=None):
        """Update the distribution of the policy"""
        logits = self.actor_forward(observations, masks, hidden_states)
        alpha, beta = self.get_beta_parameters(logits)

        # Update distribution
        self.distribution = Beta(alpha, beta, validate_args=False)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # forward pass actor
        logits = self.actor_forward(observations)
        actions_mean = self.sigmoid(logits[:, : self.output_dim] + self.beta_initial_logit_shift)
        return actions_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # forward pass critic
        value = self.critic_forward(critic_observations, masks, hidden_states)
        return value