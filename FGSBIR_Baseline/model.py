from torch.autograd import Variable
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(hp.backbone_name + '_Network(hp)')
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)
        self.hp = hp


    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
        sample_feature = self.sample_embedding_network(batch['sketch_img'].to(device))

        loss = self.loss(sample_feature, positive_feature, negative_feature)
        loss.backward()
        self.optimizer.step()

        return loss.item() 

    def evaluate(self, datloader_Test):
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        start_time = time.time()
        self.eval()
        for i_batch, sanpled_batch in enumerate(datloader_Test):
            sketch_feature, positive_feature= self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Sketch_Name.extend(sanpled_batch['sketch_path'])

            for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
                if positive_name not in Image_Name:
                    Image_Name.append(sanpled_batch['positive_path'][i_num])
                    Image_Feature_ALL.append(positive_feature[i_num])

        rank = torch.zeros(len(Sketch_Name))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            s_name = Sketch_Name[num]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()

        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]

        print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top10

    def test_forward(self, batch):            #  this is being called only during evaluation
        sketch_feature = self.sample_embedding_network(batch['sketch_img'].to(device))
        positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
        return sketch_feature.cpu(), positive_feature.cpu()



