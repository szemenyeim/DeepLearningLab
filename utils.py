import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os

from torchvision import transforms as transforms

import transforms as ext_transforms
from data_utils import enet_weighing, median_freq_balancing
from metric import IoU


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``

    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``

    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)

    """
    
    B = images.shape[0]

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images, nrow=B//2).numpy()
    labels = torchvision.utils.make_grid(labels, nrow=B//2).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The model to save.
    - optimizer (``torch.optim``): The optimizer state to save.
    - epoch (``int``): The current epoch for the model.
    - miou (``float``): The mean IoU obtained by the model.
    - args (``ArgumentParser``): An instance of ArgumentParser which contains
    the arguments used to train ``model``. The arguments are written to a text
    file in ``args.save_dir`` named "``args.name``_args.txt".

    """
    name = args['name']
    save_dir = args['save_dir']

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    '''summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))'''


def load_checkpoint(model, optimizer, folder_dir, filename):
    """Saves the model in a specified directory with a specified name.save

    Keyword arguments:
    - model (``nn.Module``): The stored model state is copied to this model
    instance.
    - optimizer (``torch.optim``): The stored optimizer state is copied to this
    optimizer instance.
    - folder_dir (``string``): The path to the folder where the saved model
    state is located.
    - filename (``string``): The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from the
    checkpoint.

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou


def display_batch(args, class_encoding, test_loader, train_loader):
    if args['imshow_batch']:
        # Get a batch of samples to display
        if args['mode'].lower() == 'test':
            images, labels = iter(test_loader).next()
        else:
            images, labels = iter(train_loader).next()
        print("Image size:", images.size())
        print("Label size:", labels.size())
        print("Class-color encoding:", class_encoding)

        # Show a batch of samples and labels
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = batch_transform(labels, label_to_rgb)
        imshow_batch(images, color_labels)


def calc_class_weights(args, class_encoding, train_loader):

    num_classes = len(class_encoding)
    device = torch.device(args['device'])

    # Get class weights from the selected weighing technique
    print("\nWeighing technique:", args['weighing'])
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")

    class_weights = 0
    if args['weighing'].lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args['weighing'].lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        if args['ignore_unlabeled']:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0
    print("Class weights:", class_weights)

    return class_weights


def setup_IoU(args, class_encoding):

    num_classes = len(class_encoding)

    # Evaluation metric
    if args['ignore_unlabeled']:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None

    metric = IoU(num_classes, ignore_index=ignore_index)

    return metric