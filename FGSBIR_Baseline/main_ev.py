import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
from evaluation import *
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str, default='./models', help='Saved models directory')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2')
    parser.add_argument('--backbone_name', type=str, default='VGG', help='VGG / InceptionV3/ Resnet50')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d',
                        help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=100)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model_path', type=str)


    hp = parser.parse_args()
    assert hp.model_path

    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)


    model = FGSBIR_Model(hp)
    model.load_state_dict(torch.load(os.path.join(hp.base_dir, hp.model_path), map_location=device))

    model.to(device)
    step_count, top1, top10 = -1, 0, 0

    with torch.no_grad():
        model.eval()
        top1_eval, top10_eval = model.evaluate(dataloader_Test)
        print('results : ', top1_eval, ' / ', top10_eval)
