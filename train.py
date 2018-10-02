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
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                             (0.24703233, 0.24348505, 0.26158768))
    ])

    # Define Datasets
    trainSet = torchvision.datasets.CIFAR10(root=root, download=True,
                                            train=True, transform=transform)
    testSet = torchvision.datasets.CIFAR10(root=root, download=True,
                                           train=False, transform=transform_val)

    # Data loaders
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=128,  # sampler=sampler,
                                              shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=128,  # sampler=sampler,
                                             shuffle=False, num_workers=2)

    # Create net, convert to cuda
    #net = densenet.DenseNet169()
    net = ConvNet(8)
    if haveCuda:
        net = net.cuda()

    # Loss, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True, weight_decay=1e-4)

    # Create LR cheduler
    scheduler = lr_scheduler.StepLR(optimizer,20)

    # Logger
    logger = Logger('./logs/run1')

    # train function
    def train(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)
        net.train()

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(trainLoader), redirect_stdout=False)

        # Epoch loop
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data

            # Convert to cuda conditionally
            if haveCuda:
                inputs, labels = inputs.cuda(), labels.cuda()


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            bar.update(i)

        # Finish progress bar
        bar.finish()

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
        net.eval()

        # Create progress bar
        bar = progressbar.ProgressBar(0, len(testLoader), redirect_stdout=False)

        # Epoch loop
        for i, data in enumerate(testLoader, 0):
            # get the inputs
            inputs, labels = data

            # Convert to cuda conditionally
            if haveCuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            bar.update(i)

        # Finish progress bar
        bar.finish()

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
    numEpoch = 20

    for epoch in range(numEpoch):

        # Step with the scheduler
        scheduler.step()

        # Call train and val
        train(epoch)
        _,val_corr = val(epoch)

        # If val accuracy is better, save the model
        if bestAcc < val_corr:
            bestAcc = val_corr
            print("Best model, saving")
            torch.save(net,root + '/model.pth')

    # Finished
    print('Finished Training')