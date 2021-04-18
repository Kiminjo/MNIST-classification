# MNIST_for_class
## Artificial neural network and deep learning class task

<hr>
* Requirements
1. (20pts) You should write your own pipeline to provide data to your model. Write your code in the template `dataset.py`. Please read the comments carefully and follow those instructions.
2. (20pts) (Report) Implement LeNet-5 and your custom MLP models in `model.py`. Some instructions are given in the file as comments. Note that your custom MLP model should have about the same number of model parameters with LeNet-5. Describe the number of model parameters of LeNet-5 and your custom MLP and how to compute them in your report.
3. (20pts) Write `main.py` to train your models, LeNet-5 and custom MLP. Here, you should monitor the training process. To do so, you need some statistics such as average loss values and accuracy at the end of each epoch.
4. (10pts) (Report) Plot above statistics, average loss value and accuracy, for training and testing. It is fine to use the test dataset as a validation dataset. Therefore, you will have four plots for each model: loss and accuracy curves for training and test datasets, respectively.
5. (10pts) (Report) Compare the predictive performances of LeNet-5 and your custom MLP. Also, make sure that the accuracy of LeNet-5 (your implementation) is similar to the known accuracy. 
6. (20pts) (Report) Employ at least more than two regularization techniques to improve LeNet-5 model. You can use whatever techniques if you think they may be helpful to improve the performance. Verify that they actually help improve the performance. Keep in mind that when you employ the data augmentation technique, it should be applied only to training data. So, the modification of provided `MNIST` class in `dataset.py` may be needed.

* Due date : 2021.04.20. Tue
* Submission : dataset.py, model.py, main.py files + **written report**