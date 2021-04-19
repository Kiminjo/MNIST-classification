
import torch.nn as nn

#------------------------------Lenet 5-------------------------------------------------------
class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):   
        super().__init__()

        """
        LeNet 5 has 2 convolutional layers and each filter size is 5x5
        After passing convolutional layer pooling layer is passed, its filter size is 2x2 
        activation function is relu and stride = 1 
        check the architecture of LeNet 5, below
        https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
        """
        
        """
        pytorch implement CNN using a function called Conv2d, check the reference below
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        Modeled by declaring Sequentail and stacking layers like keras
        Reference about relu and max pooling are attached below
        relu : https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        max_pooling : https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        """
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True), # number of parameter = 5*5*6 = 150
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True), # number of parameter = (5*5*6)*16 = 2,400
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fcn = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120), # number of parameter = 16*5*5*120 = 48,000
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84), # number of parameter = 120 * 84 = 10,080
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10), # number of parameter = 84*10 = 840
            nn.Softmax(dim=1)
            )
        

    # number of parameters for forward = 150 + 2,400 + 48,000 + 10,080 + 840 = 61,470
    # total paramters = 61,470 * 2(backpropagation) = 122,940 

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        # read here https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch/42482819#42482819
        out = out.view(-1, 16*5*5)
        output = self.fcn(out)
        return output
    
    
#------------------------------Regularized Lenet 5-------------------------------------------------------
# add dropout in FCN 
# add weight decay in cost function 
class regularized_LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):   
        super().__init__()

        """
        LeNet 5 has 2 convolutional layers and each filter size is 5x5
        After passing convolutional layer pooling layer is passed, its filter size is 2x2 
        activation function is relu and stride = 1 
        check the architecture of LeNet 5, below
        https://www.analyticsvidhya.com/blog/2021/03/the-architecture-of-lenet-5/
        """
        
        """
        pytorch implement CNN using a function called Conv2d, check the reference below
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        Modeled by declaring Sequentail and stacking layers like keras
        Reference about relu and max pooling are attached below
        relu : https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        max_pooling : https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        """
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True), # number of parameter = 5*5*6 = 150
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True), # number of parameter = (5*5*6)*16 = 2,400
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fcn = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120), # number of parameter = 16*5*5*120 = 48,000
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84), # number of parameter = 120 * 84 = 10,080
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10), # number of parameter = 84*10 = 840
            nn.Softmax(dim=1)
            )
        

    # number of parameters for forward = 150 + 2,400 + 48,000 + 10,080 + 840 = 61,470
    # total paramters = 61,470 * 2(backpropagation) = 122,940 

    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        # read here https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch/42482819#42482819
        out = out.view(-1, 16*5*5)
        output = self.fcn(out)
        return output


#------------------------------custom model-------------------------------------------------------
# Use 120 and 150 filters on each convolutional layer
# Add 1x1 filter layer immediately after each layer to reduce to small dimensions

class CustomMLP(nn.Module):
    """""
    Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """""

    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 70), # number of parameters = 28*28*70 = 54,880 
            nn.Dropout(p=0.5),
            nn.ReLU()
            )
        
        self.layer2 = nn.Sequential(
            nn.Linear(70, 64), # number of parameters = 70*64 = 4,480
            nn.Dropout(p=0.5),
            nn.ReLU()
            )
        
        self.layer3 = nn.Sequential(
            nn.Linear(64, 24), # number of paramters = 64*24 = 1,536
            nn.Dropout(p=0.5),
            nn.ReLU()
            )
        
        self.layer4 = nn.Sequential(
            nn.Linear(24, 16), # number of parameters = 24*16 = 384
            nn.Dropout(p=0.5),
            nn.ReLU()
            )
        
        self.layer5 = nn.Sequential(
            nn.Linear(16, 10), # number of parameters = 16*10 = 160
            nn.Softmax(dim=1)
            )
        
       # number of parameters for forward = 54,880 + 4,480 + 1,536 + 384 + 160 = 61,440
       # total parameters = 61,440 * 2(backpropagation) = 122,880
        

    def forward(self, img):
        img = img.view(-1, 28*28)
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        output = self.layer5(out)


        return output
    