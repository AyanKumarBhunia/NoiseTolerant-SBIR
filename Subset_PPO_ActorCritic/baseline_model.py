from torch.autograd import Variable
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Baseline_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Baseline_Model, self).__init__()
        self.sample_embedding_network = eval(hp.backbone_name + '_Network(hp)')

    def forward(self, input):
        return self.sample_embedding_network(input)









