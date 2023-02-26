# ML-Projects
Personal projects on some typical datasets (e. g. CIFAR10, MNIST ect.) with [PyTorch](https://pytorch.org/). The propose of these projects is to construct a complete pipeline from data preparation, to model training, then to performance visualization. Different models and different optimization strategies are also to be evaluated. 

## Datasets
### CIFAR10
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) are labeled subsets of the 80 million tiny images dataset. The CIFAR-10 dataset consists of 60000 coloured images in 10 classes, where each image has a dimension of 32x32. There are 50000 training images and 10000 test images. The validation set is taken from training set with probability 20%.

### MNIST
The MNIST database is a large database of handwritten digits that are shown on a grayscale picture with a dimension of 28x28 and labeled by true digits they represent. The MNIST database contains 60,000 training images and 10,000 testing images. The validation set is taken from training set with probability 20%.

## File Description
* `data` : The default folder to save downloaded datasets.
* `data_preparation` : The folder consists of Python files to acquire data and generate transformed dataset.
* `logs` : The default folder to save logs.
* `models` : The folder consists of different models. 
* `outputs` : The default folder to save trained models.
* `training_and_testing` : The folder consists of Python files to define a training/testing process.
* `utilities` : The folder consists of tools to generate logs, to visualize training results etc.

&emsp;
* `main.py` : Specify which datasets you want to learn, then you can start training.
* `cifar10.ipynb` : A IPython Notebook to train models on CIFAR10.
* `mnist.ipynb` : A IPython Notebook to train models on MNIST.

## Result
