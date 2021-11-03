# MNIST image classification

This project classifies MNIST data that is widely used in computer vision. A basic CNN model was used as the usage model. This project was developed as part of the "Artificial Neural Networks and Deep Learning" class in the Spring Semester of Data Science at Seoul National University of Science and Technology in 2021.

<br></br>

## Image classification using CNN 

MNIST data is widely used in computer vision tasks, especially classification. Also CNN is the main deep learning algorithm for image processing. Although the recent transformer and its derivative models show good performance in image processing, but the CNN model is still widely used in image processing. In the experiment, two lenet5 models under different conditions and an MLP model for performance comparison were used.

This project focuses on using CNN as part of the class and evaluating the performance by changing the parameters of the model. The changed parameters will be described later.

Check out the report on the results of this project [here](https://github.com/Kiminjo/data-mining-lecture/files/7465442/mnist.report.pdf)

<br>

### Parameters changed in the experiment
- `Regulazer` : Of the two lenet5 models, one is normal and one has a regulator. As a regulator, a dropout layer was added in the FCN and a weight decay term was added in the optimizer.
- `MLP model` : For the evaluation of CNN performance, the FCN model was used as a control set. At this time, for accurate evaluation, the model was designed so that the parameters used were similar to the CNN.

<br></br>

## Dataset

MNIST is so famous in computer vision field. So, we will minimize the description of the dataset here. However, if you are not familiar with MNIST data yet, please refer [here](https://velog.io/@tmddn0311/mnist-classification).

<br></br>

## Software Requirements

- python >= 3.5
- pytorch 
- numpy
- matplotlib

<br></br>

## Key Files

- `dataset.py` : Convert the image data into a batch of tensor. Here, since it is black and white data, it takes on a one-dimensional form.
- `model.py` : The model used for image classification was implemented. This model includes three models. These are `lenet5`, `regularized lenet5`, and `custom MLP models`.
- `main.py` : Main file of this project. It train and classify image using models in model.py. Also performance graph (error rate) were visualized using matplotlib.
