#!/usr/bin/env python3
"""Provide utility classes and functions.
"""
from argparse import ArgumentParser
import logging
from logging import Formatter, StreamHandler
from logging.handlers import TimedRotatingFileHandler
import os

from torch import randperm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .trainer import Trainer

__all__ = ['get_loaders', 'get_logger', 'get_parser', 'Trainer']


def get_loaders(train_dataset, test_dataset, args):
    """Get splitted data loaders from the training and test datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): The training dataset to be
            splitted into training and validation loaders.
        test_dataset (torch.utils.data.Dataset): The test dataset.
        args (argparse.Namespace): The argument namespace object containing
            the validation ratio, batch size,m and number of workers.

    Returns:
        (tuple): The tuple of dataloaders for training, validation, and test
        datasets.
    """
    # Create shuffled indices, and split into given ratio.
    random_indices = randperm(len(train_dataset))
    val_count = int(len(train_dataset) * args.val_ratio)
    train_indices = random_indices[val_count:]
    val_indices = random_indices[:val_count]

    # Create data loaders.
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=args.num_workers,
                            drop_last=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             drop_last=False)

    return train_loader, val_loader, test_loader


def get_logger():
    """Get a preconfigured logger that streams to STDOUT and file.

    Returns:
        (logging.Logger): The logger object.
    """
    # Prepare log directory.
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    # Create logger and formatter.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = Formatter('%(asctime)s %(message)s')

    # Create and attach stream handler.
    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create and attach file handler.
    file_handler = TimedRotatingFileHandler('logs/log.txt',
                                            when='d',
                                            encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_parser(description):
    """Get a preconfigured argument parser with default arguments.

    Args:
        (argparse.ArgumentParser): The argument parser object.
    """
    parser = ArgumentParser(description=description)

    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='The ratio of the validation set. (default: 0.3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to load the data. (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help=('The number of worker processes to use in '
                              'loading the dataset. (default: 4)'))
    parser.add_argument('--num_epochs', type=int, default=30,
                        help=('The number of training epochs to run. (default:'
                              '30)'))
    parser.add_argument('--checkpoint_file',
                        help='The path of the checkpoint file to load')
    parser.add_argument('--no_progress_bar', action='store_true',
                        default=False,
                        help='Do not display the progress bar.')

    return parser
