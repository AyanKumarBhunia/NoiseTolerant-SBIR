import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
'/train/2525993170_1': array([[ 45.43699986,  86.10754888,   0.        ],
       [ 92.34043386, 183.38287694,   0.        ],
       [ 90.655814  , 194.99578935,   0.        ],
       [ 84.80238197, 201.69938653,   1.        ],
       [ 76.42676876,  50.08099993,   0.        ],
       [ 83.38824639,  57.88414238,   0.        ],
       [ 95.69067915,  74.37574457,   1.        ]]),
"""

class FGSBIR_Dataset(data.Dataset):
    def __init__(self, hp, mode):

        self.hp = hp
        self.mode = mode
        coordinate_path = os.path.join(hp.base_dir, './../Dataset', hp.dataset_name , hp.dataset_name + '_Coordinate')
        self.root_dir = os.path.join(hp.base_dir, './../Dataset', hp.dataset_name)
        with open(coordinate_path, 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        self.Train_Sketch = [x for x in self.Coordinate if 'train' in x]
        self.Test_Sketch = [x for x in self.Coordinate if 'test' in x]

        self.train_transform = get_ransform('Train')
        self.test_transform = get_ransform('Test')

    def __getitem__(self, item):
        sample  = {}
        if self.mode == 'Train':
            sketch_path = self.Train_Sketch[item]

            positive_sample = '_'.join(self.Train_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')

            possible_list = list(range(len(self.Train_Sketch)))
            possible_list.remove(item)
            negative_item = possible_list[randint(0, len(possible_list) - 1)] # is it ensured that the positive element is not selected?
            negative_sample = '_'.join(self.Train_Sketch[negative_item].split('/')[-1].split('_')[:-1])
            negative_path = os.path.join(self.root_dir, 'photo', negative_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = Image.fromarray(sketch_img).convert('RGB')

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            n_flip = random.random() # data augmentation or flipping half of the images?
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path,
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negative_img': negative_img, 'negative_path': negative_sample
                      }

        elif self.mode == 'Test':

            sketch_path = self.Test_Sketch[item]
            vector_x = self.Coordinate[sketch_path]
            sketch_img = rasterize_Sketch(vector_x)
            sketch_img = self.test_transform(Image.fromarray(sketch_img).convert('RGB'))

            positive_sample = '_'.join(self.Test_Sketch[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.png')
            positive_img = self.test_transform(Image.open(positive_path).convert('RGB'))

            sample = {'sketch_img': sketch_img, 'sketch_path': sketch_path, 'Coordinate':vector_x,  # vector_x = list_of(x,y,p)
                      'positive_img': positive_img, 'positive_path': positive_sample}

        return sample

    def __len__(self):
        if self.mode == 'Train':
            if self.hp.debug == True:
                return 5
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            if self.hp.debug == True:
                return 5
            return len(self.Test_Sketch)

def get_dataloader(hp):

    dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')

    if torch.cuda.is_available():
        dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
        num_workers=int(hp.nThreads))
    else:
        dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True)


    dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')

    if torch.cuda.is_available():
        dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
                                             num_workers=int(hp.nThreads))
    else:
        dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False)

    return dataloader_Train, dataloader_Test

def get_ransform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(299)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(299)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)