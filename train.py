import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import progressbar
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
    root = "D:/Datasets/Traffic/trafficSigns" if sys.platform == 'win32' else "./data"

    trRoot = root+"/trainFULL"
    teRoot = root+"/testFULL"

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    trainSet = torchvision.datasets.ImageFolder(root=trRoot, transform=transform)
    testSet = torchvision.datasets.ImageFolder(root=teRoot, transform=transform_val)

    # Data loaders
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=512,  # sampler=sampler,
                                              shuffle=True, num_workers=4)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=128,  # sampler=sampler,
                                             shuffle=False, num_workers=4)


    # Create net, convert to cuda
    net = ConvNet(8)
    if haveCuda:
        net = net.cuda()

    # Loss, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9,
                          nesterov=True, weight_decay=1e-4)

    # Create LR cheduler
    # Epoch counter
    numEpoch = 20
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,numEpoch,1e-2)

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
        for i, (inputs, labels) in enumerate(trainLoader, 0):

            # Convert inputs and labels to cuda conditionally
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
        print("Train epoch %d lr: %.3f loss: %.3f correct: %.2f" % (epoch + 1, scheduler.get_lr()[0], tr_loss, tr_corr))

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
        for i, (inputs, labels) in enumerate(testLoader, 0):

            # Convert inputs and labels to cuda conditionally
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

        return val_loss, val_corr

    # Accuracy
    bestAcc = 0

    for epoch in range(numEpoch):

        # Call train and val
        train(epoch)
        _,val_corr = val(epoch)

        # If val accuracy is better, save the model
        if bestAcc < val_corr:
            bestAcc = val_corr
            print("Best model, saving")
            torch.save(net,root + '/model.pth')

        # Step with the scheduler
        scheduler.step()


    # Finished
    print('Finished Training')