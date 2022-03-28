from utils import random_string
import numpy as np
from rasterize import mydrawPNG_fromlist, select_strokes, mydraw_redPNG_fromlist
from torch.distributions import Categorical, Binomial, bernoulli
import os

import numpy
import pickle
import torch.nn as nn
import torch
import time
import torch.nn.functional as F

from rollout_buffer import RolloutBuffer
from actor_critic import ActorCritic
from custom_data_loader import CustomDataLoader
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()

        self.hp = hp

        self.policy = ActorCritic(hp)
        self.load_data = CustomDataLoader(hp)

        self.RL_optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.0003},
            {'params': self.policy.critic.parameters(), 'lr': 0.001}
        ])

        self.buffer = RolloutBuffer()

        self.step = 0

        custom_data_loader = CustomDataLoader(hp)

        self.Train_Image_Feature_ALL = custom_data_loader.load_train_data()
        self.Test_Image_Feature_ALL = custom_data_loader.load_test_data()

        self.eps_clip = 0.2
        self.mseLoss = nn.MSELoss()
        self.loss_all = 0
        self.policy_loss = 0
        self.mse_loss = 0

        self.sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])




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

                mse_loss = self.mseLoss(state_values, rewards)

                batch_loss = policy_loss - 0.01*dist_entropy_loss + \
                    0.5 * mse_loss

                loss_all = loss_all + batch_loss

            loss_all = loss_all/len(self.buffer.states)

            self.RL_optimizer.zero_grad()
            loss_all.backward()
            self.RL_optimizer.step()
            return loss_all, policy_loss, mse_loss

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

            self.loss_all, self.policy_loss, self.mse_loss = self.model_update()

            # Copy new weights into old policy
            self.policy.actor_old.load_state_dict(
                self.policy.actor.state_dict())

            # clear buffer
            self.buffer.clear()

        # return torch.mean(reward).item(), policy_loss.item(), top1_rank, top10_rank, avg_rank
        return self.loss_all, self.policy_loss, self.mse_loss

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

    def evaluate(self, datloader_Test):
        start_time = time.time()
        data_dict = {}
        top1_rank_sum, top10_rank_sum, avg_rank_sum = 0, 0, 0
        n = 0

        for batch in datloader_Test:
            with torch.no_grad():
                action, batch, log_probs, output_x, num_stroke_x = self.policy.get_action(batch, flag='test')

            top1_rank_list = []
            images_list = []
            for i in range(self.hp.sample):
                # (N, L) (batchsize, max_stroke_length_across_batch)
                stroke_select_action = action

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

                positive_feature = self.policy.sample_embedding_network_fn(
                    batch['positive_img'].to(device))
                sample_feature = self.policy.sample_embedding_network_fn(
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
