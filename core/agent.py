import numpy as np
import torch
import torch.nn.functional as functional

from core import model
from core.replay_buffer import ReplayBuffer
from core.settings import settings


class Agent:
    def __init__(self, state_size, action_size, action_max):

        self.TAU = settings.TAU
        # Tau is a factor of how we are going to modulate the parameters of
        # our target value network. Rather than doing a hard copy, we are
        # going to de-tune the parameters. Same strategy is used in DDPG and
        # TD3.
        self.REWARD_SCALE = settings.REWARD_SCALE
        # Reward scaling is how we are going to account for the entropy in
        # the framework. We are going to scale the rewards and critic's loss
        # by some factor.
        self.LEARNING_RATE = settings.LEARNING_RATE
        self.GAMMA = settings.GAMMA
        self.MEMORY_SIZE = settings.MEMORY_SIZE
        self.BATCH_SIZE = settings.BATCH_SIZE
        self.INITIAL_LEARNING = settings.INITIAL_LEARNING

        self.actor = model.Actor(state_size, action_size, action_max)
        self.critic_1 = model.Critic(state_size, action_size)
        self.critic_2 = model.Critic(state_size, action_size)
        self.value = model.Value(state_size)
        self.value_target = model.Value(state_size)

        self.update_network_parameters(self.value, self.value_target, tau=1.0)
        self.memory = ReplayBuffer(self.MEMORY_SIZE)

        if settings.CHECKPOINT is not None:
            self.load(settings.CHECKPOINT)

    def act(self, state):
        """Return action for given state."""

        state = np.expand_dims(state, 0)
        state = torch.FloatTensor(state).to(settings.DEVICE)
        actions, _ = self.actor.sample_action(state, reparametrize=False)
        return actions.detach().cpu().numpy()[0]

    @staticmethod
    def update_network_parameters(local, target, tau):
        """Soft-update target network parameters.

        θ_target = τ * θ_local + (1 - τ) * θ_target
        """

        local_params = dict(local.named_parameters())
        target_params = dict(target.named_parameters())

        for name in local_params:
            local_params[name] = \
                tau * local_params[name].clone() + \
                (1-tau) * target_params[name].clone()

        target.load_state_dict(local_params)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """Update the model using experience batch.

        Written using these referential implementations:
            https://github.com/haarnoja/sac
            https://github.com/ku2482/soft-actor-critic.pytorch
            https://spinningup.openai.com/en/latest/algorithms/sac.html
        """

        if len(self.memory) < self.INITIAL_LEARNING:
            return

        batch = self.memory.sample(self.BATCH_SIZE)
        original_states, chosen_actions, rewards, next_states, dones = batch
        dones = dones.view(-1)
        rewards = rewards.view(-1)

        # Calculate values of current and next states using the value or
        # target_value network, respectively.
        original_state_value = self.value(original_states).view(-1)
        next_state_value = self.value_target(next_states).view(-1)
        # Set value to zero, where states are terminal (from definition of
        # the value function).
        next_state_value[dones] = 0.0

        # We calculate how the actions we would choose with current policy
        # are better than the ones chosen originally.
        new_actions, log_probs = self.actor.sample_action(
            original_states, reparametrize=False)
        log_probs = log_probs.view(-1)
        # Using two critic networks and then taking their minimum improves
        # the stability of learning.
        q1_new_policy = self.critic_1(original_states, new_actions)
        q2_new_policy = self.critic_2(original_states, new_actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # Calculate the losses and backpropagate.
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * functional.mse_loss(
            original_state_value, value_target)
        # By default, PyTorch will discard the graph calculations every time
        # it backpropagates. However, we have coupling between the losses for
        # the various deep neural networks. Therefore, we need to keep the
        # graph alive until we calculate the other losses.
        value_loss.backward(retain_graph=True)  # noqa
        self.value.optimizer.step()

        new_actions, log_probs = self.actor.sample_action(
            original_states, reparametrize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1(original_states, new_actions)
        q2_new_policy = self.critic_2(original_states, new_actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.REWARD_SCALE * rewards + self.GAMMA * next_state_value
        # Multiplied by the value of the states resulting from the actions the
        # agent took.
        # Scaling factor handles the inclusion of the entropy in our loss
        # function, therefore helps encourage exploration.
        q1_old_policy = self.critic_1(original_states, chosen_actions).view(-1)
        q2_old_policy = self.critic_2(original_states, chosen_actions).view(-1)
        critic_1_loss = 0.5 * functional.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * functional.mse_loss(q2_old_policy, q_hat)

        # We can discard the graph now -> retain_graph=False
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()  # noqa
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters(
            self.value, self.value_target, tau=self.TAU)

    def save(self, episode):
        path = settings.FOLDER / f"ep{episode}"
        path.mkdir(parents=True, exist_ok=True)
        state_dict = {
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_1_optimizer': self.critic_1.optimizer.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_2_optimizer': self.critic_2.optimizer.state_dict(),
            'value': self.value.state_dict(),
            'value_optimizer': self.value.optimizer.state_dict(),
            'value_target': self.value_target.state_dict(),
            'value_target_optimizer': self.value_target.optimizer.state_dict()}
        torch.save(state_dict, path / "model.pt")

    def load(self, path):
        if path.suffix != ".pt":
            path /= "model.pt"
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        try:
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_1.load_state_dict(checkpoint['critic_1'])
            self.critic_1.optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
            self.critic_2.load_state_dict(checkpoint['critic_2'])
            self.critic_2.optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
            self.value.load_state_dict(checkpoint['value'])
            self.value.optimizer.load_state_dict(checkpoint['value_optimizer'])
            self.value_target.load_state_dict(checkpoint['value_target'])
            self.value_target.optimizer.load_state_dict(checkpoint['value_target_optimizer'])
        except RuntimeError:
            raise SystemExit(
                "Error: Loaded model doesn't match given environment.")
