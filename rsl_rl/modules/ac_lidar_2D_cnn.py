from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories
from torch.distributions import Beta
import math


def create_mlp(input_dim: int, layer_dims: list[int], activation: nn.Module, remove_last_nonlinearity: bool = False):
    layers = []  # List to store layers

    # Get all layer dimensions pairs
    input_output_pairs = zip([input_dim] + layer_dims[:-1], layer_dims)

    # Create Linear and ReLU layers for each pair
    for input_size, output_size in input_output_pairs:
        layers.append(nn.Linear(input_size, output_size))
        layers.append(activation)

    if remove_last_nonlinearity:
        layers.pop()  # Remove the last ReLU added in the loop
    # Construct the sequential model
    return nn.Sequential(*layers)

class CircularPadConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride):
        super().__init__()
        self.padding_size = kernel_size // 2
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=(1, stride))

    def circular_pad_1d(self, x, pad):
        """Apply circular padding to the last dimension of a tensor.
        (num_envs, history_length_dim, lidar_dim) --> the circular pattern of a lidar is padded
        depeneding on the kernel size, in order to keep the input output sizes the same.
        """
        if pad == 0:
            return x
        return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)
    
    def temporal_pad_1d(self, x, pad):
        "Apply padding to the temporal data in order to keep the dimensions."

        if pad == 0:
            return x
        else:
            padded_x = torch.cat(
                [
                    # Repeat the first slice `pad` times
                    x[..., :1, :].repeat(1, 1, pad, 1),
                    x,  # The original tensor
                    # Repeat the last slice `pad` times
                    x[..., -1:, :].repeat(1, 1, pad, 1)
                ],
                dim=-2  # Concatenate along the history dimension
            )
        return padded_x

    def forward(self, x):
        original_shape = x.shape
        # # Flatten all but the last two dimensions into one dimension for batch handling
        # x = x.reshape(-1, *original_shape[-2:])
        # Apply circular padding for lidar distances
        x = self.circular_pad_1d(x, self.padding_size)
        # Apply constant padding for temporal data
        x = self.temporal_pad_1d(x, self.padding_size)
        # Apply convolution
        x = self.conv(x)
        # Reshape to the original batch structure with new channel dimension
        new_shape = original_shape[:-3] + (x.shape[-3], x.shape[-2], x.shape[-1]) # num_envs, channels, history_dim, lidar_dim
        x = x.reshape(new_shape)
        return x


def create_2D_cnn(
    input_channels: int,
    layer_channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    activation: nn.Module,
):
    layers = []  # List to store layers

    # Get all layer dimensions pairs
    input_output_pairs = zip([input_channels] + layer_channels[:-1], layer_channels, kernel_sizes, strides)

    # Create custom Conv1d with circular padding and activation layers for each pair
    for input_channels, output_channels, kernel_size, stride in input_output_pairs:
        layers.append(CircularPadConv2d(input_channels, output_channels, kernel_size, stride))
        layers.append(activation)

    # Construct the sequential model
    return nn.Sequential(*layers)


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


class ActorCriticBetaLidar2DCNN(nn.Module):
    """A Network structure for training a navigation policy.
    
    possible input dimensions are:
    target_dim --> input to MLP
    cpg_dim --> input to MLP
    lidar_extra_dim --> input to MLP
    lidar_dim --> input to CNN
    lidar_history_dim --> input to CNN
    
    # Network structure:
    #                             _________                               _________
    #           Input:           |         |            Input:           |         |
    #           pos_history      |  MLP2   |            target_pos       |  MLP4   |
    #                            |_________|            cpg_state        |_________| 
    #                                 |                                       |
    #                                 |                                       |
    #                                 --------------------                    |
    #                                                     |                   |
    #                                                     v                   v                                     
    #                  _________             _________          _________          _________                 params Beta:
    # Input           |         |           |         |        |         |        |         |                alpha_vx,
    # lidar_dim x     |  CNN    | --------> |  MLP1   |------> |  MLP3   |------> |  MLP5   | -----> Output: beta_vx, 
    # history dim     |_________|           |_________|        |_________|        |_________|                alpha_vy, 
    #                                                                                                        beta_vy
    #                                                                                                        alpha_theta,
    #                                                                                                        beta_theta
        
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
            num_actor_obs: int,
            num_critic_obs: int,
            num_actions: int,
            target_dim: int,
            cpg_dim: int,
            lidar_dim: int,
            lidar_extra_dim: int,
            lidar_history_dim: int,
            target_cpg_layer_dim: list[int],
            lidar_cnn_channel_dim: list[int],
            lidar_cnn_kernel_sizes: list[int],
            lidar_cnn_strides: list[int],
            lidar_cnn_to_mlp_layer_dim: list[int],
            lidar_extra_mlp_layer_dim: list[int],
            lidar_merge_mlp_layer_dim: list[int],
            out_layer_dim: list[int],
            activation: str = "elu",
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

        # TODO @vairaviv assumption made that the NN is symmetric and num_actor_obs == num_critic_obs,
        # if changed this needs to be adapted
        if (
            num_actor_obs != target_dim + cpg_dim + (lidar_dim) * lidar_history_dim + lidar_extra_dim 
            or num_actor_obs != num_critic_obs
        ):
            raise ValueError(
                f"""num_actor_obs must be equal to target_dim + cpg_dim + (lidar_dim + lidar_extra_dim) * lidar_history_dim .
                num_actor_obs: {num_actor_obs}, 
                target_dim: {target_dim}, 
                cpg_dim: {cpg_dim},
                lidar_dim * lidar_history_dim : {(lidar_dim) * lidar_history_dim }, 
                lidar_extra_dim: {lidar_extra_dim}"""
            )
        
        activation_module = get_activation(activation)
        self.target_dim = target_dim
        self.cpg_dim = cpg_dim
        self.target_cpg_obs_dim = self.target_dim + self.cpg_dim
        self.lidar_dim = lidar_dim
        self.lidar_history_dim = lidar_history_dim
        self.lidar_extra_dim = lidar_extra_dim
        
        ##
        # define networks
        ##

        # CNN for lidar embedding
        self.actor_lidar_embedding_cnn = create_2D_cnn(
            1,
            lidar_cnn_channel_dim,
            lidar_cnn_kernel_sizes,
            lidar_cnn_strides,
            activation_module,
        )

        self.critic_lidar_embedding_cnn = create_2D_cnn(
            1,
            lidar_cnn_channel_dim,
            lidar_cnn_kernel_sizes,
            lidar_cnn_strides,
            activation_module,
        )

        # calculate CNN output size, 
        # TODO: should be the same as what? last lidar_cnn_channel_dim * (the lidar_cnn_channel_dim / lidar_cnn_strides)
        calculated_out_size = calculate_cnn_output_size(
            1, 
            lidar_cnn_channel_dim, 
            lidar_cnn_kernel_sizes, 
            lidar_cnn_strides
        )
        calculated_flattened_out_dim = calculated_out_size * lidar_cnn_channel_dim[-1]

        dummy_input = torch.zeros(1,1, lidar_history_dim, lidar_dim)
        dummy_out = self.actor_lidar_embedding_cnn(dummy_input)
        flattened_out_dim = dummy_out.shape[1] * dummy_out.shape[2] * dummy_out.shape[3] 

        if calculated_flattened_out_dim != flattened_out_dim:
            # raise ValueError(
            #     f"""The calculated CNN output size and the actual output size dont match.
            #     calculated flattened output dim: {calculated_flattened_out_dim},
            #     acutal flattened output dim: {flattened_out_dim}"""
            # )
            print(f"calculated CNN output size: {calculated_flattened_out_dim}, actual CNN output size: {flattened_out_dim}")
        
        # CNN to MLP conversion (MLP1)
        self.actor_lidar_embedding_to_mlp = create_mlp(
            flattened_out_dim, lidar_cnn_to_mlp_layer_dim, activation_module
        )
        self.critic_lidar_embedding_to_mlp = create_mlp(
            flattened_out_dim, lidar_cnn_to_mlp_layer_dim, activation_module
        )

        # lidar extra observation embeddings in MLP (MLP2)
        if lidar_extra_dim > 0 :
            self.actor_lidar_extra_mlp = create_mlp(lidar_extra_dim, lidar_extra_mlp_layer_dim, activation_module)
            self.critic_lidar_extra_mlp = create_mlp(lidar_extra_dim, lidar_extra_mlp_layer_dim, activation_module)
        else:
            lidar_extra_mlp_layer_dim = [0]


        # merge the lidar MLP with the lidar extra MLP (MLP3)

        self.actor_lidar_merged_mlp = create_mlp(
            lidar_cnn_to_mlp_layer_dim[-1] + lidar_extra_mlp_layer_dim[-1], lidar_merge_mlp_layer_dim, activation_module
        )
        self.critic_lidar_merged_mlp = create_mlp(
            lidar_cnn_to_mlp_layer_dim[-1] + lidar_extra_mlp_layer_dim[-1], lidar_merge_mlp_layer_dim, activation_module
        )


        # MLPs for regular observations like target position and cpg obs
        self.actor_target_cpg_mlp = create_mlp(self.target_cpg_obs_dim, target_cpg_layer_dim, activation_module)
        self.critic_target_cpg_mlp = create_mlp(self.target_cpg_obs_dim, target_cpg_layer_dim, activation_module)

        # MLP for Navigation, with output defined for actor and critic separate
        actor_out_layers = out_layer_dim + [num_actions * 2] # for the distribution needed (alpha and beta for each action)
        critic_out_layers = out_layer_dim + [1] # this is just for the value function, just one value should be outputed
        self.actor_out_mlp = create_mlp(
            lidar_merge_mlp_layer_dim[-1] + target_cpg_layer_dim[-1], actor_out_layers, activation_module
        )
        self.critic_out_mlp = create_mlp(
            lidar_merge_mlp_layer_dim[-1] + target_cpg_layer_dim[-1], critic_out_layers, activation_module
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
        target_cpg_obs = x[..., : self.target_cpg_obs_dim]
        lidar_obs = x[..., self.target_cpg_obs_dim : self.target_cpg_obs_dim + self.lidar_dim * self.lidar_history_dim]
        lidar_obs = lidar_obs.unsqueeze(1) # channel 1 for the 2D CNN
        if self.lidar_extra_dim > 0:
            # TODO: @vairaviv this assumes the observation vector only contains target, cpg, lidar+history 
            # and pose history information in this order and is concatenated accordingly
            lidar_extra_obs = x[..., -self.lidar_extra_dim:]

        # reshape the observation and embed the lidar data in CNN
        batch_shape = lidar_obs.shape[:-1]
        lidar_obs_reshaped = lidar_obs.reshape(*batch_shape, self.lidar_history_dim, self.lidar_dim)
        lidar_embedded_cnn = self.actor_lidar_embedding_cnn(lidar_obs_reshaped)
        
        # reshape the tensor from CNN to input to MLP 
        lidar_embedded_cnn = lidar_embedded_cnn.view(*batch_shape, -1)
        lidar_embedded_mlp = self.actor_lidar_embedding_to_mlp(lidar_embedded_cnn)

        # process extra lidar observation if available
        if self.lidar_extra_dim > 0:
            lidar_extra_embedded = self.actor_lidar_extra_mlp(lidar_extra_obs)
            # merge the lidar and lidar extra embeddings
            lidar_embedded_mlp = torch.cat((lidar_embedded_mlp.squeeze(1), lidar_extra_embedded), dim=-1)

        lidar_merged_embedded = self.actor_lidar_merged_mlp(lidar_embedded_mlp)

        target_cpg_embedded = self.actor_target_cpg_mlp(target_cpg_obs)
        # TODO: @vairaviv understand what this is used for in previous code
        if masks is not None:
            target_cpg_embedded = unpad_trajectories(target_cpg_embedded, masks)

        # navigation output 
        all_combined_embedded = torch.cat((lidar_merged_embedded, target_cpg_embedded), dim=-1)
        return self.actor_out_mlp(all_combined_embedded)
    
    def critic_forward(self, x, masks=None, hidden_states=None):
        target_cpg_obs = x[..., : self.target_cpg_obs_dim]
        lidar_obs = x[..., self.target_cpg_obs_dim : self.target_cpg_obs_dim + self.lidar_dim * self.lidar_history_dim]
        lidar_obs = lidar_obs.unsqueeze(1) # channel 1 for the 2D CNN
        if self.lidar_extra_dim > 0:
            # TODO: @vairaviv this assumes the observation vector only contains target, cpg, lidar+history 
            # and pose history information in this order and is concatenated accordingly
            lidar_extra_obs = x[..., -self.lidar_extra_dim:]

        # reshape the observation and embed the lidar data in CNN
        batch_shape = lidar_obs.shape[:-1]
        lidar_obs_reshaped = lidar_obs.reshape(*batch_shape, self.lidar_history_dim, self.lidar_dim)
        lidar_embedded_cnn = self.critic_lidar_embedding_cnn(lidar_obs_reshaped)
        
        # reshape the tensor from CNN to input to MLP 
        lidar_embedded_cnn = lidar_embedded_cnn.view(*batch_shape, -1)
        lidar_embedded_mlp = self.critic_lidar_embedding_to_mlp(lidar_embedded_cnn)

        # process extra lidar observation if available
        if self.lidar_extra_dim > 0:
            lidar_extra_embedded = self.critic_lidar_extra_mlp(lidar_extra_obs)
            # merge the lidar and lidar extra embeddings
            lidar_embedded_mlp = torch.cat((lidar_embedded_mlp.squeeze(1), lidar_extra_embedded), dim=-1)

        lidar_merged_embedded = self.critic_lidar_merged_mlp(lidar_embedded_mlp)

        target_cpg_embedded = self.critic_target_cpg_mlp(target_cpg_obs)
        # TODO: @vairaviv understand what this is used for in previous code
        if masks is not None:
            target_cpg_embedded = unpad_trajectories(target_cpg_embedded, masks)

        # navigation output 
        all_combined_embedded = torch.cat((lidar_merged_embedded,target_cpg_embedded), dim=-1)
        return self.critic_out_mlp(all_combined_embedded)

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
