import time
import pickle
import os

class CustomDataLoader():
    def __init__(self, hp):
        """
        Preloading features for faster computation
        """

        print("Started preloading image features")
        start_time = time.time()

        with open(os.path.join(hp.base_dir, 'Test_Image_Feature_ALL.pickle'), 'rb') as handle:
            self.Test_Image_Feature_ALL = pickle.load(handle)

        with open(os.path.join(hp.base_dir, 'Train_Image_Feature_ALL.pickle'), 'rb') as handle:
            self.Train_Image_Feature_ALL = pickle.load(handle)

        print('Preloaded image features in {} seconds'.format(
            time.time()-start_time))

    def load_train_data(self):
        return self.Train_Image_Feature_ALL

    def load_test_data(self):
        return self.Test_Image_Feature_ALL