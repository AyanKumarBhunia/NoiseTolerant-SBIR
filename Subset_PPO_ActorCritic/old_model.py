from torchvision.utils import save_image
from utils import random_string
from utils import id_random_string
from dataset import get_dataloader
from baseline_model import FGSBIR_Baseline_Model
import numpy as np
from rasterize import mydrawPNG_fromlist, get_stroke_num, select_strokes, mydraw_redPNG_fromlist
import torchvision.transforms as transforms
from torch.distributions import Categorical, Binomial, bernoulli
import math
import os

import numpy
from torch.autograd import Variable
import pickle
import torch.nn as nn
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network, Stroke_Embedding_Network, Sketch_Value_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
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

        return stroke_select_dist


class ActorCritic(nn.Module):

    def __init__(self, hp):
        super(ActorCritic, self).__init__()

        self.hp = hp
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

        self.preloading_features()

        self.sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        self.loss = nn.TripletMarginLoss(margin=hp.margin, reduction='none')
        self.eps_clip = 0.2

    def preloading_features(self):
        """
        Preloading features for faster computation
        """

        print("Started preloading image features")
        start_time = time.time()

        with open('Test_Image_Feature_ALL.pickle', 'rb') as handle:
            self.Test_Image_Feature_ALL = pickle.load(handle)

        with open('Train_Image_Feature_ALL.pickle', 'rb') as handle:
            self.Train_Image_Feature_ALL = pickle.load(handle)

        print('Preloaded image features in {} seconds'.format(
            time.time()-start_time))

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
            all_image_feature = self.Train_Image_Feature_ALL
        else:
            all_image_feature = self.Test_Image_Feature_ALL

        target_distance = F.pairwise_distance(sample_feature, positive_feature)
        target_distance = target_distance.view(target_distance.shape[0], 1)

        distance = torch.cdist(sample_feature, all_image_feature.to(device))

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

    def get_action(self, batch):
        stroke_select_dist = self.actor_old(batch)

        stroke_select_action_list = []
        for i in range(self.hp.sample):
            # (N, L) (batchsize, max_stroke_length_across_batch)
            stroke_select_action = stroke_select_dist.sample()
            stroke_select_action_list.append(stroke_select_action)

        stroke_select_action = torch.stack(stroke_select_action_list)
        stroke_select_action = torch.sum(stroke_select_action, dim=0)
        stroke_select_action = stroke_select_action.ge(
            math.ceil(self.hp.sample/2)).int()
        # stroke_select_action = stroke_select_action.view(1, stroke_select_action.shape[-1])

        log_probs = stroke_select_dist.log_prob(stroke_select_action)

        sketch_batch = self.batch_stroke2image(batch, stroke_select_action)

        batch_reward = self.get_reward(batch, sketch_batch)

        return stroke_select_action, batch, log_probs.detach(), batch_reward

    def RL_evaluate(self, state_batch, actions):

        stroke_select_dist = self.actor(state_batch)
        action_logprobs = stroke_select_dist.log_prob(actions)
        dist_entropy = stroke_select_dist.entropy()
        state_values = self.critic(state_batch)

        return action_logprobs, state_values, dist_entropy


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()

        self.policy = ActorCritic(hp)

        self.RL_optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.0003},
            {'params': self.policy.critic.parameters(), 'lr': 0.001}
        ])

        self.buffer = RolloutBuffer()

        self.step = 0

        self.Train_Image_Feature_ALL = self.policy.Train_Image_Feature_ALL
        self.Test_Image_Feature_ALL = self.policy.Test_Image_Feature_ALL

        self.eps_clip = 0.2
        self.mseLoss = nn.MSELoss()

    def model_update(self):

        for _ in range(80):
            loss_all = 0
            for (old_states, old_actions, old_logprobs, rewards) in zip(self.buffer.states,
                                                                        self.buffer.actions, self.buffer.logprobs, self.buffer.rewards):
                logprobs, state_values, dist_entropy = self.policy.RL_evaluate(
                    old_states, old_actions)

                ratios = torch.exp(logprobs - old_logprobs.detach())
                rewards = rewards.unsqueeze(-1)
                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages

                # loss = -torch.min(surr1, surr2) + 0.5 * \
                #     self.mseLoss(state_values, rewards) - 0.01*dist_entropy

                policy_loss_matrix = -torch.min(surr1, surr2)
                policy_loss, dist_entropy_loss = 0, 0

                for policy_loss_x, dist_entropy_x, numstroke_z in zip(policy_loss_matrix, dist_entropy, old_states['num_stroke']):
                    policy_loss += torch.sum(policy_loss_x[:numstroke_z])
                    dist_entropy_loss += torch.sum(
                        dist_entropy_x[:numstroke_z])

                policy_loss = policy_loss/logprobs.shape[0]
                dist_entropy_loss = dist_entropy_loss/logprobs.shape[0]

                batch_loss = policy_loss - 0.01*dist_entropy_loss + \
                    0.5 * self.mseLoss(state_values, rewards)

                loss_all = loss_all + batch_loss

            loss_all = loss_all/len(self.buffer.states)

            self.RL_optimizer.zero_grad()
            loss_all.backward()
            self.RL_optimizer.step()

    def train_RL_selector(self, batch):

        top1_rank, top10_rank, avg_rank = 0, 0, 0

        self.step = self.step + 1

        action, batch, log_probs, reward = self.policy.get_action(batch)

        batch_required = {}
        batch_required['stroke_wise_split'] = batch['stroke_wise_split']
        batch_required['every_stroke_len'] = batch['every_stroke_len']
        batch_required['num_stroke'] = batch['num_stroke']

        self.buffer.add_buffer(batch_required, action, log_probs, reward)

        if self.step % 20 == 0:

            self.model_update()

            # Copy new weights into old policy
            self.policy.actor_old.load_state_dict(
                self.policy.actor.state_dict())

            # clear buffer
            self.buffer.clear()

        # return torch.mean(reward).item(), policy_loss.item(), top1_rank, top10_rank, avg_rank

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

    def evaluate(self, datloader_Test):
        start_time = time.time()
        top1_rank_sum, top10_rank_sum, avg_rank_sum = 0, 0, 0
        n = 0
        self.eval()

        data_dict = {}

        for batch in datloader_Test:
            output_x, num_stroke_x = self.policy.actor.stroke_embedding_network(
                batch)

            assert self.hp.distribution
            stroke_select_dist = 0  # declaration (to prevent ide warnings)

            if self.hp.distribution == 'categorical':
                stroke_output = self.stroke_selector_fc(
                    output_x)  # (N, L, 128) --> (N, L, 2)
                stroke_output = F.softmax(stroke_output, dim=1)
                stroke_select_dist = Categorical(stroke_output)

            elif self.hp.distribution == 'binomial':
                stroke_output = self.stroke_selector_fc_to_1(
                    output_x)  # (N, L, 128) --> (N, L, 1)
                stroke_output = stroke_output.squeeze()
                stroke_output = F.sigmoid(stroke_output)
                stroke_select_dist = Binomial(
                    total_count=1, probs=stroke_output)

            elif self.hp.distribution == 'bernoulli':
                stroke_output = self.stroke_selector_fc_to_1(
                    output_x)  # (N, L, 128) --> (N, L, 1)
                stroke_output = stroke_output.squeeze()
                stroke_output = F.sigmoid(stroke_output)
                stroke_select_dist = bernoulli.Bernoulli(stroke_output)

            top1_rank_list = []
            images_list = []
            for i in range(self.hp.sample):
                # (N, L) (batchsize, max_stroke_length_across_batch)
                stroke_select_action = stroke_select_dist.sample()

                sketch_batch = []
                for id_x, (stroke_select_list, num_stroke) in enumerate(zip(stroke_select_action, num_stroke_x)):
                    # TO DEAL WITH VARIABLE NUMBER OF STROKES
                    stroke_select = stroke_select_list[:num_stroke]
                    stroke_select = torch.nonzero(stroke_select).cpu().numpy()
                    sketch_coord = batch['sketch_vector'][id_x].numpy()

                    sketch_image = self.sketch_transform(
                        mydrawPNG_fromlist(sketch_coord, stroke_select))
                    if len(stroke_select) > 0:
                        ss = np.concatenate(stroke_select)
                    else:
                        ss = []

                    tensor_image = mydraw_redPNG_fromlist(sketch_coord, ss)
                    images_list.append(torch.tensor(tensor_image))

                    sketch_batch.append(sketch_image)
                    if self.hp.aug_save_dir != '':
                        sketch_path = batch['sketch_path'][id_x]
                        if len(stroke_select) > 0:
                            to_save_sketch_image = select_strokes(
                                sketch_coord, numpy.concatenate(stroke_select))
                        else:
                            to_save_sketch_image = []
                        data_dict.update(
                            {sketch_path + '#' + random_string(4): to_save_sketch_image})

                sketch_batch = torch.stack(sketch_batch, dim=0)

                positive_feature = self.sample_embedding_network(
                    batch['positive_img'].to(device))
                sample_feature = self.sample_embedding_network(
                    sketch_batch.to(device))

                rank_sum, top1_rank, top10_rank, avg_rank = self.get_rank(
                    sample_feature, positive_feature, 'test')
                top1_rank_list.append(rank_sum[0].item())
            # print(top1_rank_list)
            # save_image(torch.cat((images_list), 2), id_random_string(n))
            # print('Image saved')

            # top1_rank = max(top1_rank_list, key=top1_rank_list.count)
            # print(top1_rank)

            n += 1
            # print(n)
            if 1 in top1_rank_list:
                top1_rank_sum += 1
            top10_rank_sum += 0
            avg_rank_sum += 0

        if self.hp.aug_save_dir != '':
            save_path = os.path.join(
                self.hp.base_dir, self.hp.aug_save_dir, str(self.hp.sample))
            with open(save_path, 'wb+') as fp:
                pickle.dump(data_dict, fp)

        top1_rank_sum /= n
        top10_rank_sum /= n
        avg_rank_sum /= n
        print('Time to Evaluate:{}'.format(time.time() - start_time))
        return top1_rank_sum, top10_rank_sum, avg_rank_sum