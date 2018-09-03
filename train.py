import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import progressbar
import densenet
from logger import Logger
from model import ConvNet
import sys

if __name__ == "__main__":

    haveCuda = torch.cuda.is_available()

    # Makes multiple runs comparable
    if haveCuda:
        torch.cuda.manual_seed(1)
    else:
        torch.manual_seed(1)

    # path to dataset
    root = 'C:/data/' if sys.platform == 'win32' else './data'

    # Data augmentation

    # Define Datasets

    # Data loaders

    # Create net, convert to cuda

    # Loss, and optimizer

    # Create LR cheduler

    # Logger
    logger = Logger('./logs')

    # train function
    def train(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)

        # Create progress bar

        # Epoch loop

            # get the inputs

            # Convert to cuda conditionally


            # zero the parameter gradients

            # forward + backward + optimize

            # compute statistics

            # Update progress bar

        # Finish progress bar

        # print and plot statistics
        tr_loss = running_loss / len(trainLoader)
        tr_corr = correct / total * 100
        print("Train epoch %d loss: %.3f correct: %.2f" % (epoch + 1, tr_loss, tr_corr))

        # 1. Log scalar values (scalar summary)
        info = {'Training Loss': tr_loss, 'Training Accuracy': tr_corr}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

    # Validation function
    def val(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to eval (for batchnorm and dropout)

        # Create progress bar

        # Epoch loop
            # get the inputs

            # Convert to cuda conditionally

            # forward

            # compute statistics

            # Update progress bar

        # Finish progress bar

        # print and plot statistics
        val_loss = running_loss / len(testLoader)
        val_corr = correct / total * 100
        print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, val_loss, val_corr))

        # 1. Log scalar values (scalar summary)
        info = {'Validation Loss': val_loss, 'Validation Accuracy': val_corr}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        return val_loss, val_corr

    # Accuracy
    bestAcc = 0

    # Epoch counter
    numEpoch = 50

    for epoch in range(numEpoch):

        # Step with the scheduler

        # Call train and val

        # If val accuracy is better, save the model

    # Finished
    print('Finished Training')