from baseline_model import FGSBIR_Baseline_Model
from rasterize import mydrawPNG_fromlist
import torchvision.transforms as transforms
import math
import os

import pickle
import torch.nn as nn
from Networks import Sketch_Value_Network
import torch
import time
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sketch_actor_network import Sketch_Actor_Network
from custom_data_loader import CustomDataLoader


class ActorCritic(nn.Module):

    def __init__(self, hp):
        super(ActorCritic, self).__init__()

        self.hp = hp
        self.load_data = CustomDataLoader(hp)
        self.actor = Sketch_Actor_Network(hp)
        self.critic = Sketch_Value_Network(hp)

        self.actor_old = Sketch_Actor_Network(hp)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_old.requires_grad = False

        self.sample_embedding_network = FGSBIR_Baseline_Model(hp)
        self.sample_embedding_network.to(device)

        self.sample_embedding_network.load_state_dict(torch.load(os.path.join(
            hp.base_dir, './models/VGG_ShoeV2_model_best_new.pth'), map_location=device))
        self.sample_embedding_network.requires_grad = False

        # self.preloading_features()

        self.sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        self.loss = nn.TripletMarginLoss(margin=hp.margin, reduction='none')
        self.eps_clip = 0.2

    def sample_embedding_network_fn(self, feature):
        return self.sample_embedding_network(feature)

    def get_action(self, batch, flag='train'):
        stroke_select_dist, output_x, num_stroke_x = self.actor_old(batch)

        stroke_select_action_list = []
        for i in range(self.hp.sample):
            # (N, L) (batchsize, max_stroke_length_across_batch)
            stroke_select_action = stroke_select_dist.sample()
            stroke_select_action_list.append(stroke_select_action)

        stroke_select_action = torch.stack(stroke_select_action_list)
        stroke_select_action = torch.sum(stroke_select_action, dim=0)
        stroke_select_action = stroke_select_action.ge(
            math.ceil(self.hp.sample / 2)).int()
        # stroke_select_action = stroke_select_action.view(1, stroke_select_action.shape[-1])

        log_probs = stroke_select_dist.log_prob(stroke_select_action)

        sketch_batch = self.batch_stroke2image(batch, stroke_select_action)

        if flag == 'train':
            batch_reward = self.get_reward(batch, sketch_batch)
            return stroke_select_action, batch, log_probs.detach(), batch_reward

        else:
            return stroke_select_action, batch, log_probs.detach(), output_x, num_stroke_x

    def RL_evaluate(self, state_batch, actions):

        stroke_select_dist, output_x, num_stroke_x = self.actor(state_batch)
        action_logprobs = stroke_select_dist.log_prob(actions)
        dist_entropy = stroke_select_dist.entropy()
        state_values = self.critic(state_batch)

        return action_logprobs, state_values, dist_entropy

    def batch_stroke2image(self, batch, stroke_select_action):
        sketch_batch = []
        for id_x, (stroke_select_list, num_stroke) in enumerate(zip(stroke_select_action, batch['num_stroke'])):
            # TO DEAL WITH VARIABLE NUMBER OF STROKES
            stroke_select = stroke_select_list[:num_stroke]
            stroke_select = torch.nonzero(stroke_select).cpu().numpy()
            sketch_coord = batch['sketch_vector'][id_x].numpy()
            sketch_image = self.sketch_transform(
                mydrawPNG_fromlist(sketch_coord, stroke_select))
            sketch_batch.append(sketch_image)
        sketch_batch = torch.stack(sketch_batch, dim=0)

        return sketch_batch

    def get_rank(self, sample_feature, positive_feature, flag):

        if flag == 'train':
            all_image_feature = self.load_data.Train_Image_Feature_ALL.to(device)
        else:
            all_image_feature = self.load_data.Test_Image_Feature_ALL.to(device)

        target_distance = F.pairwise_distance(sample_feature, positive_feature)
        target_distance = target_distance.view(target_distance.shape[0], 1)

        distance = torch.cdist(sample_feature.to(device), all_image_feature)

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

    def get_reward(self, batch, sketch_batch):

        with torch.no_grad():
            positive_feature = self.sample_embedding_network(
                batch['positive_img'].to(device))
            negative_feature = self.sample_embedding_network(
                batch['negative_img'].to(device))
            sample_feature = self.sample_embedding_network(
                sketch_batch.to(device))

        rank_sum, top1_rank, top10_rank, avg_rank = self.get_rank(
            sample_feature, positive_feature, 'train')
        rank_reward = rank_sum

        triplet_loss_reward = self.loss(
            sample_feature, positive_feature, negative_feature)
        # reward is something that needs to be maximized -- so negative.
        # negative done below

        if self.hp.reward == 'rank1':
            reward = 1 / rank_reward
        elif self.hp.reward == 'rank2':
            reward = - triplet_loss_reward
        elif self.hp.reward == 'rank3':
            reward = - triplet_loss_reward + 1 / rank_reward
        elif self.hp.reward == 'rank4':
            reward = - triplet_loss_reward - rank_reward
        elif self.hp.reward == 'rank5':
            reward = 1 / torch.sqrt(rank_reward.float())
        elif self.hp.reward == 'rank6':
            reward = - triplet_loss_reward + 1 / \
                     torch.sqrt(torch.abs(rank_reward).float())
        elif self.hp.reward == 'rank7':
            reward = - torch.sqrt(torch.abs(triplet_loss_reward))
        elif self.hp.reward == 'rank8':
            reward = - torch.sqrt(torch.abs(triplet_loss_reward).float()) + \
                     1 / torch.sqrt(torch.abs(rank_reward).float())

        return reward
