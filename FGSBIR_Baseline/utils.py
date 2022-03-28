import string
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_self_train(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'positive_img': [], 'positive_boxes': [],
                 'negative_img': [], 'negative_boxes': [],
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())
        batch_mod['negative_boxes'].append(torch.tensor(i_batch['negative_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)

    return batch_mod


def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                 'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod



def random_string(name=None):
    N = 7
    res = str(name)
    if name == None:
        res = ''.join(random.choices(string.ascii_lowercase + string.digits, k = N))
    return 'imgs/' + res + '.png'