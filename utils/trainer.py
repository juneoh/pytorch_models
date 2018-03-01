#!/usr/bin/env python3
"""Define a trainer class for PyTorch models.
"""
import torch
from torch import Tensor
from torch.nn import DataParallel
from torch.autograd import Variable

from tqdm import tqdm


class Trainer:
    """Wraps a PyTorch model to provide convenience functions.

    Args:
        model (torch.nn.Module): The PyTorch model.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The training optimizer.
        scheduler (torch.optim._LRScheduler): The learning rate scheduler.
        prediction (function): The function to convert the model output into
            prediction labels. Defaults to max index.
    """

    def __init__(self, model, criterion, optimizer, scheduler,
                 prediction=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.prediction = prediction

        if self.prediction is None:
            self.prediction = lambda o: torch.max(o.cpu().data, 1)[1]

        # Send model to GPU if possible.
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self.model = DataParallel(self.model.cuda())
            self.criterion = self.criterion.cuda()

    def train_batch(self, x, y):
        """Train a batch and step the optimizer.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The target label.

        Returns:
            (float): The batch loss.
        """
        # Put the model in training mode.
        self.model.train()

        # Send tensors to GPU if possible.
        if self._use_cuda:
            x = x.cuda()
            y = y.cuda()

        self.optimizer.zero_grad()

        # Forward pass.
        x = Variable(x, requires_grad=True)
        output = self.model(x)

        # Backward pass.
        y = Variable(y)
        loss = self.criterion(output, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().data[0] / len(y)

    def train_epoch(self, loader, no_progress_bar=False):
        """Train an epoch and step the scheduler.

        Args:
            loader (torch.utils.data.DataLoader): The data loader.
            no_progress_bar (bool): Whether to hide the progress bar.

        Returns:
            (float): The mean training loss.
        """
        epoch_loss = []

        if not no_progress_bar:
            progress = tqdm(total=len(loader),
                            unit='batch',
                            desc='[train] batch loss: 0.000',
                            leave=False)

        for x, y in loader:
            batch_loss = self.train_batch(x, y)

            epoch_loss.append(batch_loss)

            if not no_progress_bar:
                progress.update(1)
                progress.set_description(
                    '[train] batch loss: {loss:.3f}'.format(loss=batch_loss))

        self.scheduler.step()

        if not no_progress_bar:
            progress.close()

        return Tensor(epoch_loss).mean()

    def infer_batch(self, x):
        """Infer a batch.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            (torch.Tensor): The inferred labels.
        """
        # Put the model in evaluation mode.
        self.model.eval()

        # Send tensor to GPU if possible.
        if self._use_cuda:
            x = x.cuda()

        # Forward pass.
        x = Variable(x, volatile=True)
        output = self.model(x)
        y_pred = self.prediction(output)

        return y_pred

    def infer_epoch(self, loader, no_progress_bar=False):
        """Infer an epoch.

        Args:
            loader (torch.utils.data.DataLoader): The data loader.
            no_progress_bar (bool): Whether to hide the progress bar.

        Returns:
            (float): The mean training loss.
        """
        epoch_accuracy = []

        if not no_progress_bar:
            progress = tqdm(total=len(loader),
                            unit='batch',
                            desc='[validate] batch accuracy: 0.000',
                            leave=False)

        for x, y in loader:
            y_pred = self.infer_batch(x)

            epoch_accuracy.append((y == y_pred).sum() / len(y))

            if not no_progress_bar:
                progress.update(1)
                progress.set_description(
                    '[test] batch accuracy: {accuracy:.3f}'.format(
                        accuracy=epoch_accuracy[-1]))

        if not no_progress_bar:
            progress.close()

        return Tensor(epoch_accuracy).mean()

    def load(self, path):
        """Load model and optimizer parameters from a saved checkpoint file.

        Args:
            path (str): The path of the saved checkpoint file.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.last_epoch = checkpoint['scheduler_last_epoch']

    def save(self, path):
        """Save model and optimizer parameters to a checkpoint file.

        Args:
            path (str): The path to save the checkpoint file to.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_last_epoch': self.scheduler.last_epoch,
        }

        torch.save(checkpoint, path)
