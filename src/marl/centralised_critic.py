from typing import Optional

import numpy as np
from ray.rllib.algorithms import PPO
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
  PPOTF1Policy,
  PPOTF2Policy,
)
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.train._internal.session import _TrainingResult

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class CentralizedCriticModel(TorchModelV2, nn.Module):
  """Multi-agent model that implements a centralized VF."""

  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    TorchModelV2.__init__(
        self, obs_space, action_space, num_outputs, model_config, name
    )
    nn.Module.__init__(self)

    # Base of the model
    self.model = TorchFC(obs_space, action_space, num_outputs, model_config, name)

    # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
    input_size = 4 + 4 + 5  # obs (modified) + opp_obs + opp_act

    self.central_vf = nn.Sequential(
        SlimFC(input_size, 16, activation_fn = nn.Tanh),
        SlimFC(16, 1),
    )

  @override(ModelV2)
  def forward(self, input_dict, state, seq_lens):
    model_out, _ = self.model(input_dict, state, seq_lens)
    return model_out, []

  def central_value_function(self, obs, opponent_obs, opponent_actions):
    input_ = torch.cat(
        [
          obs,
          opponent_obs,
          torch.nn.functional.one_hot(opponent_actions.long(), 5).float(),  # 2 -> 5?
        ],
        1,
    )

    # print(f"input_ {input_}")
    # print(f"input_ {input_.shape}")

    central_vf = self.central_vf(input_)
    # print(f"central vf {central_vf.shape}")
    # print(f"central vf {central_vf}")
    reshaped = torch.reshape(central_vf, [-1])
    # print(f"reshaped {reshaped.shape}")
    # print(f"reshaped {reshaped}")

    return reshaped

  @override(ModelV2)
  def value_function(self):
    return self.model.value_function()  # not used


OPPONENT_OBS = "opponent_obs"
OPPONENT_ACTION = "opponent_action"


class CentralizedValueMixin:
  """Add method to evaluate the central value function from the model."""

  def __init__(self):
    self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(
    policy, sample_batch, other_agent_batches = None, episode = None
):
  if hasattr(policy, "compute_central_vf"):
    assert other_agent_batches is not None
    if policy.config["enable_connectors"]:
      [(_, _, opponent_batch)] = list(other_agent_batches.values())
    else:
      [(_, opponent_batch)] = list(other_agent_batches.values())

    # also record the opponent obs and actions in the trajectory
    sample_batch[OPPONENT_OBS] = opponent_batch[SampleBatch.CUR_OBS]
    sample_batch[OPPONENT_ACTION] = opponent_batch[SampleBatch.ACTIONS]

    # overwrite default VF prediction with the central VF
    sample_batch[SampleBatch.VF_PREDS] = (
      policy.compute_central_vf(
          convert_to_torch_tensor(
              sample_batch[SampleBatch.CUR_OBS], policy.device
          ),
          convert_to_torch_tensor(sample_batch[OPPONENT_OBS], policy.device),
          convert_to_torch_tensor(
              sample_batch[OPPONENT_ACTION], policy.device
          ),
      )
      .cpu()
      .detach()
      .numpy()
    )
  else:
    # Policy hasn't been initialized yet, use zeros.
    sample_batch[OPPONENT_OBS] = np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
    sample_batch[OPPONENT_ACTION] = np.zeros_like(sample_batch[SampleBatch.ACTIONS])
    sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
        sample_batch[SampleBatch.REWARDS], dtype = np.float32
    )

  completed = sample_batch[SampleBatch.TERMINATEDS][-1]
  if completed:
    last_r = 0.0
  else:
    last_r = sample_batch[SampleBatch.VF_PREDS][-1]

  train_batch = compute_advantages(
      sample_batch,
      last_r,
      policy.config["gamma"],
      policy.config["lambda"],
      use_gae = policy.config["use_gae"],
  )

  return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
  # Save original value function.
  vf_saved = model.value_function

  # print(f"seq len {train_batch.get(SampleBatch.SEQ_LENS)}")

  # print(f"obs1 {train_batch[SampleBatch.CUR_OBS]}")

  # Calculate loss with a custom value function.
  model.value_function = lambda: policy.model.central_value_function(
      train_batch[SampleBatch.CUR_OBS],
      train_batch[OPPONENT_OBS],
      train_batch[OPPONENT_ACTION],
  )
  policy._central_value_out = model.value_function()

  # print(f"val targets {train_batch[Postprocessing.VALUE_TARGETS]}")
  # print(f"model {model}")
  # print(f"dist class {dist_class}")
  # print(f"train batch {train_batch}")

  loss = base_policy.loss(model, dist_class, train_batch)

  # Restore original value function.
  model.value_function = vf_saved

  return loss


def central_vf_stats(policy, train_batch):
  # Report the explained variance of the central value function.
  return {
    "vf_explained_var": explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], policy._central_value_out
    )
  }


def get_ccppo_policy(base):
  class CCPPOTFPolicy(CentralizedValueMixin, base):
    def __init__(self, observation_space, action_space, config):
      base.__init__(self, observation_space, action_space, config)
      CentralizedValueMixin.__init__(self)

    @override(base)
    def loss(self, model, dist_class, train_batch):
      # Use super() to get to the base PPO policy.
      # This special loss function utilizes a shared
      # value function defined on self, and the loss function
      # defined on PPO policies.
      return loss_with_central_critic(
          self, super(), model, dist_class, train_batch
      )

    @override(base)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches = None, episode = None
    ):
      return centralized_critic_postprocessing(
          self, sample_batch, other_agent_batches, episode
      )

    @override(base)
    def stats_fn(self, train_batch: SampleBatch):
      stats = super().stats_fn(train_batch)
      stats.update(central_vf_stats(self, train_batch))
      return stats

  return CCPPOTFPolicy


CCPPOStaticGraphTFPolicy = get_ccppo_policy(PPOTF1Policy)
CCPPOEagerTFPolicy = get_ccppo_policy(PPOTF2Policy)


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
  def __init__(self, observation_space, action_space, config):
    PPOTorchPolicy.__init__(self, observation_space, action_space, config)
    CentralizedValueMixin.__init__(self)

  @override(PPOTorchPolicy)
  def loss(self, model, dist_class, train_batch):
    return loss_with_central_critic(self, super(), model, dist_class, train_batch)

  @override(PPOTorchPolicy)
  def postprocess_trajectory(
      self, sample_batch, other_agent_batches = None, episode = None
  ):
    return centralized_critic_postprocessing(
        self, sample_batch, other_agent_batches, episode
    )


class CentralizedCritic(PPO):
  @classmethod
  @override(PPO)
  def get_default_policy_class(cls, config):
    return CCPPOTorchPolicy
