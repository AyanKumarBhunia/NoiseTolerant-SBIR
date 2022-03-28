import torch
import time
from model import FGSBIR_Model
from cvs_dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
# from evaluation import *
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str, default='./models', help='Saved models directory')

    parser.add_argument('--dataset_name', type=str, default='ShoeV2_CVS')
    parser.add_argument('--cvs', type=str, default='1')
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

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)


    model = FGSBIR_Model(hp)
    model.to(device)
    step_count, top1, top10 = -1, 0, 0

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, loss, top1, top10, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top10_eval)

                if top1_eval > top1:
                    torch.save(model.state_dict(), os.path.join(hp.saved_models, hp.backbone_name +
                                                                '_' + hp.dataset_name +
                                                                '_' + hp.cvs + '_model_best.pth'))
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')