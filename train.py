import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import progressbar
from model import ConvNet
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    haveCuda = torch.cuda.is_available()

    # Makes multiple runs comparable
    if haveCuda:
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)

    # path to dataset
    root = "E:/Traffic/trafficSigns" if sys.platform == 'win32' else "./data"

    trRoot = root+"/trainFULL"
    teRoot = root+"/testFULL"

    # Data augmentation


    # Define Datasets

    # Data loaders


    # Create net, convert to cuda

    # Loss, and optimizer

    # Create LR cheduler
    # Epoch counter
    numEpoch = 20

    # train function
    def train(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)

        # Create progress bar

        # Epoch loop

            # Convert inputs and labels to cuda conditionally

            # zero the parameter gradients

            # forward + backward + optimize

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar

        # Finish progress bar

        # print and plot statistics
        tr_loss = running_loss / len(trainLoader)
        tr_corr = correct / total * 100
        print("Train epoch %d lr: %.3f loss: %.3f correct: %.2f" % (epoch + 1, scheduler.get_lr()[0], tr_loss, tr_corr))

    # Validation function
    def val(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to eval (for batchnorm and dropout)

        # Create progress bar

        # Epoch loop

            # Convert inputs and labels to cuda conditionally

            # forward

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar

        # Finish progress bar

        # print and plot statistics
        val_loss = running_loss / len(testLoader)
        val_corr = correct / total * 100
        print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, val_loss, val_corr))

        return val_loss, val_corr

    # Best accuracy
    bestAcc = 0

    trAccs = []
    trLosses = []
    valAccs = []
    valLosses = []
    x = range(numEpoch)

    for epoch in range(numEpoch):

        # Call train and val

        # If val accuracy is better, save the model

        # Step with the scheduler


    # Finished
    plt.figure()
    plt.plot(x,trAccs,x,valAccs)
    plt.figure()
    plt.plot(x,trLosses,x,valLosses)
    plt.show()
    print('Finished Training')