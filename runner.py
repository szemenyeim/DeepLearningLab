import torch

class Runner:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.
    - is_train (```bool```): the model mode (True = train, False = validation OR test)
    - optim (``Optimizer``): The optimization algorithm.

    """

    def __init__(self, model, data_loader, criterion, metric, device, is_train=True, optim=None):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.is_train = is_train

        if self.optim is None:
            self.is_train = False

    def run_epoch(self, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.set_model_mode()


        epoch_loss = 0.0
        self.metric.reset()

        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            if self.is_train:
                loss, outputs = self.train_pass(inputs, labels)
            else:
                loss, outputs = self.test_pass(inputs, labels)


            """Keep track of loss for current epoch"""
            epoch_loss += None

            # Keep track of the evaluation metric
            self.metric.add(outputs.detach(), labels.detach())

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()

    def set_model_mode(self):
        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

    def test_pass(self, inputs, labels):
        loss, outputs = None, None

        """TODO: the below two steps should be within a clause"""
        """TODO: here something is needed that is specific to the test pass"""
            #Forward propagation


            #Loss computation


        return loss, outputs

    def train_pass(self, inputs, labels):
        loss, outputs = None, None
        """TODO: Forward propagation"""


        """TODO: Loss computation"""


        """TODO: Backpropagation"""


        return loss, outputs
