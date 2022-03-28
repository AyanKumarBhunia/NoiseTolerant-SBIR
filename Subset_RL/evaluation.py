import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import os
import sys


"""
python evaluation.py --base_dir=$(pwd) --model_path=models/VGG_ShoeV2_margin_0.2_batchsize_16_model_best.pth --eval True --sample 1 --aug_save_dir=models
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str, default='./models', help='Saved models directory')
    parser.add_argument('--model_path', type=str, default='./models', help='Model Load')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--nThreads', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)
    parser.add_argument('--print_freq_iter', type=int, default=100)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--reward', type=str, default='rank', help='rank / triplet')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--distribution', type=str, default='categorical', help='categorical / binomial / bernoulli')
    parser.add_argument('--eval', type=bool, default=False, help='False / True')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--aug_save_dir', type=str, default='')


    hp = parser.parse_args()

    if hp.debug == False and not torch.cuda.is_available():
        sys.exit("GPU NOT DETECTED, RE RUN THIS CODE")

    print(hp)

    model = FGSBIR_Model(hp)
    model.load_state_dict(torch.load(os.path.join(hp.base_dir, hp.model_path), map_location=device))
    model.eval()

    dataloader_Train, dataloader_Test = get_dataloader(hp)
    model.to(device)

    with torch.no_grad():
        top1_eval, top10_eval, avg_eval = model.evaluate(dataloader_Test)
        print('evaluation results top1 / top10 / avg ranks : ', top1_eval, ' / ', top10_eval, '/', avg_eval)






