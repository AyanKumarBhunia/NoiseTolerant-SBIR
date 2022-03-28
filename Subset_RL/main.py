import sys
import os
import argparse
import torch
import time
from model import FGSBIR_Model
from dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-Grained SBIR Model')

    parser.add_argument('--base_dir', type=str, default=os.getcwd())
    parser.add_argument('--saved_models', type=str, default='./models', help='Saved models directory')

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
    parser.add_argument('--reward', type=str, default='rank1',
                        help='rank1 / rank2 / rank3 / rank4 / rank5 / rank6 / rank7 / rank8')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--distribution', type=str, default='categorical', help='categorical / binomial / bernoulli')
    parser.add_argument('--eval', type=bool, default=False, help='False / True')
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--aug_save_dir', type=str, default='')

    hp = parser.parse_args()

    folder = os.path.join(hp.base_dir, './models/Tensorboard_logs')

    if not os.path.isdir(folder):
        os.makedirs(folder)

    if hp.debug == False and not torch.cuda.is_available():
        sys.exit("GPU NOT DETECTED, RE RUN THIS CODE")

    print(hp)

    model = FGSBIR_Model(hp)
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    model.to(device)
    step_count, top1, top10 = -1, 0, 0

    for i_epoch in range(hp.max_epoch):
        top1_train_avg, top10_train_avg, avg_train_avg = 0, 0, 0
        reward_avg, policy_loss_avg = 0, 0
        n = 0
        for batch_data in dataloader_Train:
            step_count = step_count + 1

            start = time.time()
            model.train()
            reward, policy_loss, top1_train, top10_train, avg_train = model.train_RL_selector(batch=batch_data)

            n += 1 * len(batch_data['positive_path'])
            reward_avg += reward
            policy_loss_avg += policy_loss

            top1_train_avg += top1_train
            top10_train_avg += top10_train
            avg_train_avg += avg_train

            model_save_path = os.path.join(hp.saved_models,
                                           hp.backbone_name + '_' +
                                           str(hp.dataset_name) +
                                           '_margin_' + str(hp.margin) +
                                           '_batchsize_' + str(hp.batchsize) +
                                           '_reward_' + str(hp.reward)
                                           + '_model_best.pth')

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Policy Loss: {:.5f}, Top1_Accuracy: {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format
                      (i_epoch, step_count, policy_loss, top1_train, top10_train, time.time()-start))

            if step_count % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    top1_eval, top10_eval, avg_eval = model.evaluate(dataloader_Test)
                    print('evaluation results top1 / top10 / avg ranks : ', top1_eval, ' / ', top10_eval, '/', avg_eval)

                if top1_eval > top1:
                    torch.save(model.state_dict(), model_save_path)
                    top1, top10 = top1_eval, top10_eval
                    print('Model Updated')

        """
        After each training epoch
        """

        top1_train_avg /= n
        top10_train_avg /= n
        avg_train_avg /= n
        reward_avg /= n
        policy_loss_avg /= n

        print('training results top1 / top10 / avg ranks : ', top1_train_avg, ' / ', top10_train_avg, '/', avg_train_avg)
