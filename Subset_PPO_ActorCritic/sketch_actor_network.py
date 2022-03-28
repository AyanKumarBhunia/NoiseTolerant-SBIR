from torch.distributions import Categorical

import torch.nn as nn
from Networks import Stroke_Embedding_Network
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sketch_Actor_Network(nn.Module):
    def __init__(self, hp):
        super(Sketch_Actor_Network, self).__init__()

        self.hp = hp

        self.stroke_embedding_network = Stroke_Embedding_Network(hp)
        self.stroke_selector_fc = nn.Linear(128, 2)  # categorical

    def forward(self, batch):
        output_x, num_stroke_x = self.stroke_embedding_network(batch)

        stroke_select_dist = 0  # declaration (to prevent ide warnings)

        stroke_output = self.stroke_selector_fc(
            output_x)  # (N, L, 128) --> (N, L, 2)
        stroke_output = F.softmax(stroke_output, dim=1)
        stroke_select_dist = Categorical(stroke_output)

        return stroke_select_dist, output_x, num_stroke_x
