import os
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
from rasterize import mydrawPNG_fromlist, get_stroke_num
from itertools import combinations

from torchvision.utils import save_image


def Evaluate_FGSBIR(model, datloader_Test):

    sketch_transform = transforms.Compose([transforms.Resize(299), transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    Image_Feature_ALL = [] # positive image feature
    Image_Name = [] # positive image path
    Sketch_Feature_ALL = [] # sketch image feature
    Sketch_Name = [] # sketch path
    start_time = time.time()

    for i_batch, sanpled_batch in enumerate(datloader_Test):
        sketch_feature, positive_feature= model.test_forward(sanpled_batch) # get feature from sketch and positive image
        # [1x512]
        Sketch_Feature_ALL.extend(sketch_feature) # add sketch feature from sketch image to list
        Sketch_Name.extend(sanpled_batch['sketch_path'])  # add sketch path to list
        print('Batch index: ', i_batch)


        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)
    Sketch_Feature_ALL = []

    for i_batch, sanpled_batch in enumerate(datloader_Test):


        sketch_coord = sanpled_batch['Coordinate'][0].numpy() # shape = [n,3]

        total_stroke = get_stroke_num(sketch_coord) # gets the number of strokes (0 to 1)

        stroke_idx_list = list(range(total_stroke))
        stroke_combi_all = []
        for x in range(1, total_stroke+1):
            stroke_combi_all.extend(list(combinations(stroke_idx_list, x))) # gets all combinations of stroke orderings

        rank_sketch = []
        print('Batch index: ', i_batch, ' Total stroke combinations', len(stroke_combi_all))

        for idx in range(len(stroke_combi_all) // 128 + 1):

            if (idx + 1) * 128 <= len(stroke_combi_all):
                stroke_combi = stroke_combi_all[idx * 128: (idx + 1) * 128]
            else:
                stroke_combi = stroke_combi_all[idx * 128: len(stroke_combi_all)]
            sketch_image = [sketch_transform(mydrawPNG_fromlist(sketch_coord, x)) for x in stroke_combi]

            sketch_image = torch.stack(sketch_image, dim=0)

            s_name = sanpled_batch['sketch_path'][0]
            sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
            position_query = Image_Name.index(sketch_query_name)

            sketch_feature_combi = model.sample_embedding_network(sketch_image.to(device)).cpu()
            # gives features of all combinations of sketches



            for sketch_feature in sketch_feature_combi:
                target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[position_query].unsqueeze(0))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
                rank_sketch.append(distance.le(target_distance).sum())

        rank[i_batch] = torch.stack(rank_sketch).min()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format(time.time() - start_time))
    return top1, top10



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')
    
    parser.add_argument('--base_dir', type=str, default=os.getcwd())
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
    model.load_state_dict(torch.load(os.path.join(hp.base_dir, './models/VGG_ShoeV2_model_best_new.pth'), map_location=device))

    with torch.no_grad():
        model.eval()
        top1_eval, top10_eval = model.evaluate(dataloader_Test)
        print(top1_eval, top10_eval)

    with torch.no_grad():
        model.eval()
        top1_eval, top10_eval = Evaluate_FGSBIR(model, dataloader_Test)
        print(top1_eval, top10_eval)



"""
    $ python evaluation.py --nThreads 1 --batchsize 1 --print_freq_iter 1
    Namespace(backbone_name='VGG', batchsize=1, dataset_name='ShoeV2', eval_freq_iter=100, learning_rate=0.0001, max_epoch=200, nThreads=1, pool_method='AdaptiveAvgPool2d', print_freq_iter=1, root_dir='./../Dataset/')
    Batch Index:  0
    Batch Index:  1
    Batch Index:  0  Stroke Length:  7
    Combined Stroke Iteration 0
    Batch Index:  1  Stroke Length:  255
    Combined Stroke Iteration 0
    Combined Stroke Iteration 1
    Time to EValuate:238.10176372528076
    1.0 1.0
"""
