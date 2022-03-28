import time
import torch.nn as nn
from torch import optim
import  torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Networks import VGG_Network, InceptionV3_Network, Resnet50_Network
import argparse
from dataset import get_dataloader
from model import FGSBIR_Model
import torchvision.transforms as transforms
from rasterize import mydrawPNG_fromlist, get_stroke_num, mydraw_redPNG_fromlist, convert_to_red
from itertools import combinations
from utils import random_string

from torchvision.utils import save_image
import numpy as np



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



    for i_batch, sanpled_batch in enumerate(datloader_Test):


        sketch_coord = sanpled_batch['Coordinate'][0].numpy() # shape = [n,3]

        total_stroke = get_stroke_num(sketch_coord) # gets the number of strokes (0 to 1)

        stroke_idx_list = list(range(total_stroke))
        stroke_combi_all = []
        stroke_combi_all_for_image = []
        half_strokes = stroke_idx_list[:len(stroke_idx_list)//2]
        stroke_combi_all.append(tuple(half_strokes))
        stroke_combi_all_for_image.append(tuple(half_strokes))
        for i in stroke_idx_list[len(stroke_idx_list)//2:]:
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

        distances = []

        for sketch_feature in sketch_feature_combi:
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                              Image_Feature_ALL[position_query].unsqueeze(0))
            distances.append(target_distance)

        distances = torch.stack(distances)
        distances_shift = torch.roll(distances, 1, 0)
        red_strokes = distances_shift > distances # white stroke only if distance decreases, else red stroke
        red_strokes[0] = False
        print('sketch_feature_combi.shape: ', sketch_feature_combi.shape)
        print('stroke_combi_all:', stroke_combi_all)
        print('stroke_combi_all_for_image', stroke_combi_all_for_image)
        print('distances', distances)
        print('red_strokes', red_strokes)

        new_sketch_image = [mydraw_redPNG_fromlist(sketch_coord, x) for x in stroke_for_image]
        for i in range(len(red_strokes)):
            if red_strokes[i]:
                new_sketch_image[i] = convert_to_red(new_sketch_image[i])
        sketch_image = np.array(new_sketch_image)
        sketch_image = np.sum(sketch_image, axis=0).clip(0,255)

        save_image(torch.tensor(sketch_image), random_string(i_batch))
        print('Image saved')

    print('Time to EValuate:{}'.format(time.time() - start_time))







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
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

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    model = FGSBIR_Model(hp)
    model.to(device)
    model.load_state_dict(torch.load('./VGG_ShoeV2_model_best.pth', map_location=device))

    with torch.no_grad():
        model.eval()
        Wrong_Strokes_FGSBIR(model, dataloader_Test)



