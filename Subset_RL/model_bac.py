import math
import os

import numpy
from torch.autograd import Variable
import pickle
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network, Stroke_Embedding_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical, Binomial, bernoulli
import torchvision.transforms as transforms
from rasterize import mydrawPNG_fromlist, get_stroke_num, select_strokes
import numpy as np
from baseline_model import FGSBIR_Baseline_Model

from dataset import get_dataloader
from utils import random_string



class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()

        self.sample_embedding_network = FGSBIR_Baseline_Model(hp)
        self.sample_embedding_network.to(device)

        self.sample_embedding_network.load_state_dict(torch.load(os.path.join(hp.base_dir, './models/VGG_ShoeV2_model_best_new.pth'), map_location=device))
        self.sample_embedding_network.requires_grad = False

        self.loss = nn.TripletMarginLoss(margin=hp.margin, reduction='none')
        self.sample_train_params = self.sample_embedding_network.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)
        self.hp = hp

        self.stroke_embedding_network = Stroke_Embedding_Network(hp)

        if self.hp.distribution == 'categorical':
            self.stroke_selector_fc = nn.Linear(128, 2)  # categorical
        else:
            self.stroke_selector_fc_to_1 = nn.Linear(128, 1)  # binomial

        if self.hp.distribution == 'categorical':
            self.RL_optimizer = optim.Adam(list(self.stroke_embedding_network.parameters())
                                       + list(self.stroke_selector_fc.parameters()), hp.learning_rate)
        else:
            self.RL_optimizer = optim.Adam(list(self.stroke_embedding_network.parameters())
                                           + list(self.stroke_selector_fc_to_1.parameters()), hp.learning_rate)

        self.sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        # if self.hp.reward == 'rank':
        self.preloading_features()


    def preloading_features(self):
        """
        Preloading features for faster computation
        """

        print("Started preloading image features")
        start_time = time.time()

        self.sample_embedding_network.eval()
        original_batchsize = self.hp.batchsize # storing the original batchsize
        self.hp.batchsize = 1
        dataloader_Train, dataloader_Test = get_dataloader(self.hp)

        Test_Image_Name_ALL = []
        Test_Image_Feature_ALL = []
        with torch.no_grad():
            c = 0
            for batch in dataloader_Test:
                path = batch['positive_path']
                img = batch['positive_img']
                if path not in Test_Image_Name_ALL:
                    c += 1
                    print('Loaded test image feature {}'.format(c))
                    positive_feature = self.sample_embedding_network(img.to(device))
                    Test_Image_Feature_ALL.append(positive_feature)
                    Test_Image_Name_ALL.append(path)
        Test_Image_Feature_ALL = torch.cat(Test_Image_Feature_ALL)
        self.Test_Image_Feature_ALL = Test_Image_Feature_ALL

        if self.hp.eval == False:
            Train_Image_Name_ALL = []
            Train_Image_Feature_ALL = []
            with torch.no_grad():
                c = 0
                for batch in dataloader_Train:
                    path = batch['positive_path']
                    img = batch['positive_img']
                    if path not in Train_Image_Name_ALL:
                        c += 1
                        print('Loaded train image feature {}'.format(c))
                        positive_feature = self.sample_embedding_network(img.to(device))
                        Train_Image_Feature_ALL.append(positive_feature)
                        Train_Image_Name_ALL.append(path)
            Train_Image_Feature_ALL = torch.cat(Train_Image_Feature_ALL)
            self.Train_Image_Feature_ALL = Train_Image_Feature_ALL

        print('Preloaded image features in {} seconds'.format(time.time()-start_time))
        self.hp.batchsize = original_batchsize # reinstated the original batchsize

    def get_rank(self, sample_feature, positive_feature, flag):
        if flag == 'train':
            all_image_feature = self.Train_Image_Feature_ALL
        else:
            all_image_feature = self.Test_Image_Feature_ALL

        target_distance = F.pairwise_distance(sample_feature, positive_feature)
        target_distance = target_distance.view(target_distance.shape[0], 1)

        distance = torch.cdist(sample_feature, all_image_feature)

        rank = distance.le(target_distance)
        # rank = torch.le(distance, target_distance)
        rank = rank.sum(1)
        # torch.le() returns 0 for some cases even when LE value present
        # https://ibb.co/D703bXF
        rank = torch.clamp(rank, min=1)

        top1 = rank.le(1).sum().item() / rank.shape[0]
        top10 = rank.le(10).sum().item() / rank.shape[0]
        avg = rank.sum().item() / rank.shape[0]

        return rank, top1, top10, avg

    def train_RL_selector(self, batch):

        top1_rank, top10_rank, avg_rank = 0, 0, 0
        self.RL_optimizer.zero_grad()

        output_x, num_stroke_x = self.stroke_embedding_network(batch)

        assert self.hp.distribution
        stroke_select_dist = 0 # declaration (to prevent ide warnings)

        if self.hp.distribution == 'categorical':
            stroke_output = self.stroke_selector_fc(output_x)  # (N, L, 128) --> (N, L, 2)
            stroke_output = F.softmax(stroke_output, dim=1)
            stroke_select_dist = Categorical(stroke_output)

        elif self.hp.distribution == 'binomial':
            stroke_output = self.stroke_selector_fc_to_1(output_x)  # (N, L, 128) --> (N, L, 1)
            stroke_output = stroke_output.squeeze()
            stroke_output = F.sigmoid(stroke_output)
            stroke_select_dist = Binomial(total_count=1, probs=stroke_output)

        elif self.hp.distribution == 'bernoulli':
            stroke_output = self.stroke_selector_fc_to_1(output_x)  # (N, L, 128) --> (N, L, 1)
            stroke_output = stroke_output.squeeze()
            stroke_output = F.sigmoid(stroke_output)
            stroke_select_dist = bernoulli.Bernoulli(stroke_output)

        stroke_select_action_list = []
        for i in range(self.hp.sample):
            stroke_select_action = stroke_select_dist.sample()  # (N, L) (batchsize, max_stroke_length_across_batch)
            stroke_select_action_list.append(stroke_select_action)

        stroke_select_action = torch.stack(stroke_select_action_list)
        stroke_select_action = torch.sum(stroke_select_action, dim=0)
        stroke_select_action = stroke_select_action.ge(math.ceil(self.hp.sample/2)).int()

        stroke_select_action = stroke_select_action.view(1, stroke_select_action.shape[-1])

        log_probs = stroke_select_dist.log_prob(stroke_select_action)

        # stroke_select_list = stroke_select_action.split(num_stroke_x, dim=-1)

        sketch_batch = []
        for id_x, (stroke_select_list, num_stroke) in enumerate(zip(stroke_select_action, num_stroke_x)):
            stroke_select = stroke_select_list[:num_stroke] # TO DEAL WITH VARIABLE NUMBER OF STROKES
            stroke_select = torch.nonzero(stroke_select).cpu().numpy()
            sketch_coord = batch['sketch_vector'][id_x].numpy()
            sketch_image = self.sketch_transform(mydrawPNG_fromlist(sketch_coord, stroke_select))
            sketch_batch.append(sketch_image)
        sketch_batch = torch.stack(sketch_batch, dim=0)

        with torch.no_grad():
            positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
            negative_feature = self.sample_embedding_network(batch['negative_img'].to(device))
            sample_feature = self.sample_embedding_network(sketch_batch.to(device))

        rank_sum, top1_rank, top10_rank, avg_rank = self.get_rank(sample_feature, positive_feature, 'train')
        rank_reward = 1 / rank_sum

        if self.hp.reward == 'rank':
            reward = rank_reward
        else:
            triplet_loss_reward = -self.loss(sample_feature, positive_feature, negative_feature)
            # reward is something that needs to be maximized -- so negative.
            reward = triplet_loss_reward

        # numstroke_z == TO HANDLE THE VARIABLE NUMBER OF STROKE for every sketch inside a batch
        policy_loss = 0
        for logprob_x, reward_y, numstroke_z in zip(log_probs, reward, num_stroke_x):
            policy_loss += torch.sum(-logprob_x[:numstroke_z] * reward_y)

        policy_loss = policy_loss/self.hp.batchsize

        policy_loss.backward()
        self.RL_optimizer.step()

        return torch.mean(reward).item(), policy_loss.item(), top1_rank, top10_rank, avg_rank

    def evaluate(self, datloader_Test):
        start_time = time.time()
        top1_rank_sum, top10_rank_sum, avg_rank_sum = 0, 0, 0
        n = 0
        self.eval()

        data_dict = {}

        for batch in datloader_Test:
            output_x, num_stroke_x = self.stroke_embedding_network(batch)

            assert self.hp.distribution
            stroke_select_dist = 0  # declaration (to prevent ide warnings)

            if self.hp.distribution == 'categorical':
                stroke_output = self.stroke_selector_fc(output_x)  # (N, L, 128) --> (N, L, 2)
                stroke_output = F.softmax(stroke_output, dim=1)
                stroke_select_dist = Categorical(stroke_output)

            elif self.hp.distribution == 'binomial':
                stroke_output = self.stroke_selector_fc_to_1(output_x)  # (N, L, 128) --> (N, L, 1)
                stroke_output = stroke_output.squeeze()
                stroke_output = F.sigmoid(stroke_output)
                stroke_select_dist = Binomial(total_count=1, probs=stroke_output)

            elif self.hp.distribution == 'bernoulli':
                stroke_output = self.stroke_selector_fc_to_1(output_x)  # (N, L, 128) --> (N, L, 1)
                stroke_output = stroke_output.squeeze()
                stroke_output = F.sigmoid(stroke_output)
                stroke_select_dist = bernoulli.Bernoulli(stroke_output)

            stroke_select_action_list = []
            for i in range(self.hp.sample):
                stroke_select_action = stroke_select_dist.sample()  # (N, L) (batchsize, max_stroke_length_across_batch)
                stroke_select_action_list.append(stroke_select_action)

            stroke_select_action = torch.stack(stroke_select_action_list)
            stroke_select_action = torch.sum(stroke_select_action, dim=0)
            stroke_select_action = stroke_select_action.ge(math.ceil(self.hp.sample / 2)).int()

            stroke_select_action = stroke_select_action.view(1, stroke_select_action.shape[-1])

            sketch_batch = []
            for id_x, (stroke_select_list, num_stroke) in enumerate(zip(stroke_select_action, num_stroke_x)):
                stroke_select = stroke_select_list[:num_stroke]  # TO DEAL WITH VARIABLE NUMBER OF STROKES
                stroke_select = torch.nonzero(stroke_select).cpu().numpy()
                sketch_coord = batch['sketch_vector'][id_x].numpy()

                sketch_image = self.sketch_transform(mydrawPNG_fromlist(sketch_coord, stroke_select))
                sketch_batch.append(sketch_image)
                if self.hp.aug_save_dir != '':
                    sketch_path = batch['sketch_path'][id_x]
                    if len(stroke_select) > 0:
                        to_save_sketch_image = select_strokes(sketch_coord, numpy.concatenate(stroke_select))
                    else:
                        to_save_sketch_image = []
                    data_dict.update({sketch_path + '#' + random_string(4): to_save_sketch_image})

            sketch_batch = torch.stack(sketch_batch, dim=0)

            positive_feature = self.sample_embedding_network(batch['positive_img'].to(device))
            sample_feature = self.sample_embedding_network(sketch_batch.to(device))

            rank_sum, top1_rank, top10_rank, avg_rank = self.get_rank(sample_feature, positive_feature, 'test')
            n += 1
            top1_rank_sum += top1_rank
            top10_rank_sum += top10_rank
            avg_rank_sum += avg_rank

        if self.hp.aug_save_dir != '':
            save_path = os.path.join(self.hp.base_dir, self.hp.aug_save_dir, str(self.hp.sample))
            with open(save_path, 'wb+') as fp:
                pickle.dump(data_dict, fp)

        top1_rank_sum /= n
        top10_rank_sum /= n
        avg_rank_sum /= n
        print('Time to Evaluate:{}'.format(time.time() - start_time))
        return top1_rank_sum, top10_rank_sum, avg_rank_sum




