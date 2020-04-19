# Semantic Segmentation

This repository includes multiple implementation of semantic segmentation networks which were trained on CityScapes dataset. 

## Installation

Use the [Cityscapes](https://www.cityscapes-dataset.com/) to train and test the model.

System Requirements:

Pytorch 1.3.1
Matplotlib 3.0.3
Torchvision 0.4.2
Python 3.5.2

## Usage

1) Download the dataset and change the DATA_PATH in train_test.py
2) Change the architecture you want to train or test in the main fucntion.
3) USE: "python train_test.py train" to train the model
   USE: "python train_test.py test" to test the model


## Results:
Segnet ~82% \
Modified Segnet ~60% \
Modified Segnet with skip connection ~85%

![Alt text](images/seg.png?raw=true "Segmentation results 1")

![Alt text](images/seg2.png?raw=true "Segmentation results 2")

