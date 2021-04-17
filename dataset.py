

# read tarfile and get imgae and label
import io
import tarfile 
import pandas as pd
from PIL import Image

# libarary about pytorch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
#%%

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])])

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.tar_extractor = tarfile.open(data_dir, 'r')
        self.members = self.tar_extractor.getmembers()
        self.labels = get_label(self.members)
        self.transform = transform


    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):
        image = self.tar_extractor.extractfile(self.labels.iloc[idx, 1]).read() 
        img = Image.open(io.BytesIO(image))
        label = torch.tensor(self.labels.iloc[idx, 2])
        
        if self.transform :
            img = self.transform(img)
            
        return img, label


# get file indecies and labels from png file name 
def get_label(members) :
    return pd.DataFrame([[member.name[-11:-6], member.name, int(member.name[-5])] for member in members[1:]], columns=['idx', 'name', 'label'])    


# print 4 outputs of mnist data
def visualization_mnist_data(data) :
    image, label = next(iter(data))
    for idx in range(0,4) :
        if idx == 0 :
            plot_num = 221
        elif idx == 1 :
            plot_num = 222
        elif idx == 2 :
            plot_num = 223
        else :
            plot_num = 224
    
        plt.subplot(plot_num)
        sample_image = np.array((image[idx])).reshape((28,28))
        sample_label = label[idx]
        plt.title('target : {}'.format(sample_label))
        plt.xticks([]); plt.yticks([])
        plt.imshow(sample_image)


#%%
if __name__ == '__main__':
    
    dataset = MNIST(data_dir = '../data/train.tar', transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [40000, 10000, 10000])

    train_data = DataLoader(train_dataset, batch_size=64)
    val_data = DataLoader(val_dataset, batch_size=64)
    test_data = DataLoader(test_dataset)
    
    
    # display 4 outputs 
    visualization_mnist_data(train_data)
    
    # check image's shape
    sample_image = next(iter(train_data))[0]
    sample_image
    # output returns (64, 1, 28, 28) 
    # which means 64 images(batch size) individually have 28x28 size and only have one channel(black & white image )
    
