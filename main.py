import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import transforms as ext_transforms
import utils
from args import get_arguments
from data import CamVid
from models.enet import ENet
from runner import Runner
from utils import display_batch, calc_class_weights, setup_IoU


def load_dataset():
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    """Create datasets and dataloaders"""
    # Get selected dataset
    # Load the training set as tensors
    train_set = CamVid()
    train_loader = None

    # Load the validation set as tensors
    val_set = CamVid()
    val_loader = None

    # Load the test set as tensors
    test_set = CamVid()
    test_loader = None

    # Get encoding between pixel values in label images and RGB colors
    class_encoding = train_set.color_encoding

    # Remove the road_marking class from the CamVid dataset as it's merged
    # with the road class
    if args.dataset.lower() == 'camvid':
        del class_encoding['road_marking']

    # Print information for debugging
    print("Number of classes to predict:", len(class_encoding))
    print("Runner dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    display_batch(args, class_encoding, test_loader, train_loader)

    class_weights = calc_class_weights(args, class_encoding, train_loader)

    return train_loader, val_loader, test_loader, class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    """Create network and deploy to device"""
    # Intialize ENet
    model = None
    # Check if the network architecture is correct
    print(model)

    """Create criterion with weights"""
    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = None

    """Create ADAM optimizer with weight decay"""
    optimizer = None

    """Create learning rate decay scheduler (StepLR)"""
    lr_updater = None

    metric = setup_IoU(args, class_encoding)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print(f"Resuming from model: Start epoch = {start_epoch} | Best mean IoU = {best_miou:.4f}")
    else:
        start_epoch = 0
        best_miou = 0


    """Create Runner objects"""
    print()
    train = None
    val = None

    for epoch in range(start_epoch, args.epochs):
        print(f">>>> [Epoch: {epoch:d}] Training")

        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)

        print(f">>>> [Epoch: {epoch:d}] Avg. loss: {epoch_loss:.4f} | Mean IoU: {miou:.4f}")

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(f">>>> [Epoch: {epoch:d}] Validation")

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(f">>>> [Epoch: {epoch:d}] Avg. loss: {loss:.4f} | Mean IoU: {miou:.4f}")

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print(f"{key}: {class_iou:.4f}")

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, args)

        lr_updater.step()

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequently used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    metric = setup_IoU(args, class_encoding)

    # Test the trained model on the test set
    test = Runner(model, test_loader, criterion, metric, device, is_train=False)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    # Make predictions!
    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    args = get_arguments()

    device = torch.device(args.device)

    train_loader, val_loader, test_loader, w_class, class_encoding = load_dataset()

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Intialize a new ENet model
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previoulsy saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        test(model, test_loader, w_class, class_encoding)
