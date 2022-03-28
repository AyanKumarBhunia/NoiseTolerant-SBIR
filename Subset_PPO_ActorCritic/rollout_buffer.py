import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

    def add_buffer(self, batch, action, log_probs, reward):

        self.states.append(batch)
        self.actions.append(action)
        self.logprobs.append(log_probs)
        self.rewards.append(reward)
