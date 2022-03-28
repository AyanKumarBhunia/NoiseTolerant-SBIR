import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Stroke_Embedding_Network(nn.Module):
    def __init__(self, hp):
        super(Stroke_Embedding_Network, self).__init__()

        hp.data_encoding_type = '3point'
        if hp.data_encoding_type == '3point':
            inp_dim = 3
        elif hp.data_encoding_type == '5point':
            inp_dim = 5
        hp.hidden_size = 128
        hp.stroke_LSTM_num_layers = 1

        self.LSTM_stroke = nn.GRU(inp_dim, 128, num_layers=hp.stroke_LSTM_num_layers,
                                  batch_first=True, bidirectional=True)

        self.embedding_1 = nn.Linear(hp.hidden_size*2*hp.stroke_LSTM_num_layers, hp.hidden_size)

        self.LSTM_global = nn.GRU(hp.hidden_size, hp.hidden_size, num_layers=hp.stroke_LSTM_num_layers,
                                   batch_first=True, bidirectional=True)

        self.embedding_2 = nn.Linear(2*hp.hidden_size, hp.hidden_size)
        self.layernorm = nn.LayerNorm(hp.hidden_size)

    def forward(self, batch):

        # batch['stroke_wise_split'][:,:,:2] /= 800

        x = pack_padded_sequence(batch['stroke_wise_split'].to(device),
                                 batch['every_stroke_len'],
                                 batch_first=True, enforce_sorted=False)

        # batch['stroke_wise_split'] shape: [number of strokes, maximum number of points per stroke, 3] (padded by zero so all strokes of equal length)
        # batch['every_stroke_len'] [n1, n2, n3, o1, o2, o3, p1, p2, p3, p4]

        # x[0] --> [n1 set of (x,y,p), n2 set of (x,y,p) .... p4 set of (x,y,p)]
        # x[1] --> [strokes at t1, strokes at t2, .. strokes at max (tn)]

        _, x_stroke = self.LSTM_stroke(x.float()) # x_stroke --> last hidden state [2, total number of strokes, 128]

        x_stroke = x_stroke.permute(1, 0, 2).reshape(x_stroke.shape[1], -1)
        x_stroke = self.embedding_1(x_stroke) # convert (n, 256) --> (n, 128)

        # batch['num_stroke'] --> number of strokes in each sketch

        x_sketch = x_stroke.split(batch['num_stroke'])
        x_sketch_h = x_sketch
        x_sketch = pad_sequence(x_sketch, batch_first=True) # to make same number of strokes for each sketch

        x_sketch = pack_padded_sequence(x_sketch, torch.tensor(batch['num_stroke']),
                                        batch_first=True, enforce_sorted=False)
        _, x_sketch_hidden = self.LSTM_global(x_sketch.float())
        x_sketch_hidden = x_sketch_hidden.permute(
            1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
        x_sketch_hidden = self.embedding_2(x_sketch_hidden) # convert (n, 256) --> (n, 128)

        out = []
        for x, y in zip(x_sketch_h, x_sketch_hidden):
            out.append(self.layernorm(x + y))
        out = pad_sequence(out, batch_first=True) # to make same number of strokes for each sketch

        return out, batch['num_stroke']




class Resnet50_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        return F.normalize(x)


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default


    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)
