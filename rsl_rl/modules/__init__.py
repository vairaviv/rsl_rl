#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic, ActorCriticSeparate
from .actor_critic_recurrent import ActorCriticRecurrent
from .actor_critic_beta import ActorCriticBeta
from .normalizer import EmpiricalNormalization
from .local_nav_module import SimpleNavPolicy

# imported into pascal's distribution
from .ac_beta_compress import ActorCriticBetaCompress, ActorCriticBetaCompressTemporal, ActorCriticBetaLidarTemporal
from .ac_lidar import ActorCriticBetaRecurrentLidar
from .ac_lidar_extra import ActorCriticBetaRecurrentLidarCnn
from .ac_lidar_height import ActorCriticBetaRecurrentLidarHeightCnn
from .ac_lidar_cnn import ActorCriticBetaLidarCNN

__all__ = [
    "ActorCritic",
    "ActorCriticRecurrent",
    "ActorCriticBeta",
    "EmpiricalNormalization",
    "SimpleNavPolicy",
    "ActorCriticSeparate",
    "ActorCriticBetaCompress",
    "ActorCriticBetaCompressTemporal",
    "ActorCriticBetaLidarTemporal",
    "ActorCriticBetaRecurrentLidar",
    "ActorCriticBetaRecurrentLidarCnn",
    "ActorCriticBetaRecurrentLidarHeightCnn",
    "ActorCriticBetaLidarCNN"
]

    # Network structure Actor Critic Lidar CNN with feature alignment for target pos + proprioception:
    #                             _________                               _________
    #           Input:           |         |            Input:           |         |
    #           pos_history      |  MLP2   |            target_pos       |  MLP4   |
    #                            |_________|            proprioception   |_________|
    #                                 |                                       |
    #                                 |                                       |
    #                                 --------------------                    |
    #                                                     |                   |
    #                                                     v                   v                                     
    #                  _________             _________         _________          _________
    # Input           |         |           |         |        |         |        |         |
    # lidar_dim x     |  CNN    | --------> |  MLP1   |------> |  MLP3   |------> |  MLP5   | -----> Output
    # history dim     |_________|           |_________|        |_________|        |_________|