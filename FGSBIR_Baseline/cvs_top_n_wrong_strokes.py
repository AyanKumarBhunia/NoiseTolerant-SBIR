import time
import torch.nn as nn
from torch import optim
import  torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
import argparse
from cvs_dataset import get_dataloader
from model import FGSBIR_Model
import torchvision.transforms as transforms
from rasterize import mydrawPNG_fromlist, get_stroke_num, mydraw_redPNG_fromlist, convert_to_red, convert_to_blue, \
    convert_to_green, convert_to_black, select_strokes
from itertools import combinations
from utils import random_string
import os

from torchvision.utils import save_image
import numpy as np
import pickle



def Wrong_Strokes_FGSBIR(model, datloader_Test):

    sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    Image_Feature_ALL = []  # positive image feature
    Image_Name = []  # positive image path
    Sketch_Feature_ALL = []  # sketch image feature
    Sketch_Name = []  # sketch path
    start_time = time.time()

    for i_batch, sanpled_batch in enumerate(datloader_Test):
        sketch_feature, positive_feature = model.test_forward(sanpled_batch)  # get feature from sketch and positive image
        # [1x512]
        Sketch_Feature_ALL.extend(sketch_feature)  # add sketch feature from sketch image to list
        Sketch_Name.extend(sanpled_batch['sketch_path'])  # add sketch path to list
        #


        print('Batch index: ', i_batch)

        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)

    new_strokes_dict = {}

    for i_batch, sanpled_batch in enumerate(datloader_Test):


        sketch_coord = sanpled_batch['Coordinate'][0].numpy() # shape = [n,3]

        total_stroke = get_stroke_num(sketch_coord) # gets the number of strokes (0 to 1)

        stroke_idx_list = list(range(total_stroke))
        stroke_combi_all = []
        stroke_combi_all_for_image = []
        half_strokes = []
        for i in stroke_idx_list[:]:
            half_strokes.extend([i])
            stroke_combi_all.append(tuple(half_strokes))
            stroke_combi_all_for_image.append(tuple([i]))

        rank_sketch = []
        print('Batch index: ', i_batch, ' Total strokes', len(stroke_combi_all))

        stroke_combi = stroke_combi_all[:]
        stroke_for_image = stroke_combi_all_for_image[:]

        sketch_image = [sketch_transform(mydrawPNG_fromlist(sketch_coord, x)) for x in stroke_combi]

        sketch_image = torch.stack(sketch_image, dim=0)

        s_name = sanpled_batch['sketch_path'][0]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)

        sketch_feature_combi = model.sample_embedding_network(sketch_image.to(device)).cpu()
        # gives features of all combinations of sketches

        # distances = []

        for sketch_feature in sketch_feature_combi:
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                              Image_Feature_ALL[position_query].unsqueeze(0))
            # distances.append(target_distance)

            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            # print(distance)
            rank_sketch.append(distance.le(target_distance).sum())

        # distances = torch.stack(distances)


        print('rank_sketch', torch.stack(rank_sketch))

        strokes_to_keep = []

        to_keep = torch.stack(rank_sketch).le(int(hp.topn))
        to_keep_index = []
        for i in range(len(to_keep)):
            if to_keep[i]:
                to_keep_index.append(i)


        for i in to_keep_index:
            strokes_to_keep.append(list(range(i+1)))



        print('old_length', get_stroke_num(sketch_coord))

        new_strokes_coordinates = [select_strokes(sketch_coord, x) for x in strokes_to_keep]

        print('new_lengths (many strokes)', [get_stroke_num(x) for x in new_strokes_coordinates])

        # print(sanpled_batch['sketch_path'][0], new_strokes_coordinates)

        print(sanpled_batch['sketch_path'][0])

        cx = 0
        for x in new_strokes_coordinates:
            cx += 1
            new_strokes_dict.update({sanpled_batch['sketch_path'][0] + '+' + str(cx): x})

    save_path = 'cvs_dataset_topn/' + 'top_' + hp.topn + '_' + hp.cvs
    with open(save_path, 'wb+') as fp:
        pickle.dump(new_strokes_dict, fp)


    print('Time to EValuate:{}'.format(time.time() - start_time))







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2_CVS')
    parser.add_argument('--cvs', type=str, default='1')
    parser.add_argument('--topn', type=str, default='5')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--root_dir', type=str, default='./../Dataset/')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=10)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--base_dir', type=str, default=os.getcwd())

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)
    model.load_state_dict(torch.load('./models/' + './VGG_ShoeV2_CVS_' +
                                     hp.cvs + '_model_best.pth', map_location=device))

    with torch.no_grad():
        model.eval()
        Wrong_Strokes_FGSBIR(model, dataloader_Test)



