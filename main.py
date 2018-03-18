# %% -*- coding: utf-8 -*-
'''
Author: Shreyas Padhy
Driver file for Standard UNet Implementation
'''
from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr

from data import BraTSDatasetUnet, BraTSDatasetLSTM
from losses import DICELossMultiClass
from models import UNet
from tqdm import tqdm
import numpy as np

# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(
    description='UNet + BDCLSTM for BraTS Dataset')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=128, metavar='N',
                    help='imsize')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--data-folder', type=str, default='./Data/', metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--modality', type=str, default='flair', metavar='str',
                    help='Modality to use for training (default: flair)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str',
                    help='Optimizer (default: SGD)')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

DATA_FOLDER = args.data_folder

# %% Loading in the Dataset
dset_train = BraTSDatasetUnet(DATA_FOLDER, train=True,
                              keywords=[args.modality],
                              im_size=[args.size, args.size],
                              transform=tr.ToTensor())

train_loader = DataLoader(dset_train,
                          batch_size=args.batch_size,
                          shuffle=True, num_workers=1)

dset_test = BraTSDatasetUnet(DATA_FOLDER, train=False,
                             keywords=[args.modality],
                             im_size=[args.size, args.size],
                             transform=tr.ToTensor())

test_loader = DataLoader(dset_test,
                         batch_size=args.test_batch_size,
                         shuffle=False, num_workers=1)


print("Training Data : ", len(train_loader.dataset))
print("Test Data :", len(test_loader.dataset))

# %% Loading in the model
model = UNet()

if args.cuda:
    model.cuda()

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.99)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.beta1, args.beta2))


# Defining Loss Function
criterion = DICELossMultiClass()


def train(epoch, loss_lsit):
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, mask)
        loss_list.append(loss.data[0])

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage DICE Loss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(train_accuracy=False, save_output=False):
    test_loss = 0

    if train_accuracy:
        loader = train_loader
    else:
        loader = test_loader

    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image, volatile=True), Variable(
            mask, volatile=True)

        output = model(image)

        # test_loss += criterion(output, mask).data[0]
        maxes, out = torch.max(output, 1, keepdim=True)

        if save_output and (not train_accuracy):
            np.save('./npy-files/out-files/{}-batch-{}-outs.npy'.format(args.save,
                                                                        batch_idx),
                    out.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-batch-{}-masks.npy'.format(args.save,
                                                                         batch_idx),
                    mask.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-batch-{}-images.npy'.format(args.save,
                                                                          batch_idx),
                    image.data.float().cpu().numpy())

        if save_output and train_accuracy:
            np.save('./npy-files/out-files/{}-train-batch-{}-outs.npy'.format(args.save,
                                                                              batch_idx),
                    out.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-train-batch-{}-masks.npy'.format(args.save,
                                                                               batch_idx),
                    mask.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-train-batch-{}-images.npy'.format(args.save,
                                                                                batch_idx),
                    image.data.float().cpu().numpy())

        test_loss += criterion(output, mask).data[0]

    # Average Dice Coefficient
    test_loss /= len(loader)
    if train_accuracy:
        print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))
    else:
        print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))


if args.train:
    loss_list = []
    for i in tqdm(range(args.epochs)):
        train(i, loss_list)
        test()

    plt.plot(loss_list)
    plt.title("UNet bs={}, ep={}, lr={}".format(args.batch_size,
                                                args.epochs, args.lr))
    plt.xlabel("Number of iterations")
    plt.ylabel("Average DICE loss per batch")
    plt.savefig("./plots/{}-UNet_Loss_bs={}_ep={}_lr={}.png".format(args.save,
                                                                    args.batch_size,
                                                                    args.epochs,
                                                                    args.lr))

    np.save('./npy-files/loss-files/{}-UNet_Loss_bs={}_ep={}_lr={}.npy'.format(args.save,
                                                                               args.batch_size,
                                                                               args.epochs,
                                                                               args.lr),
            np.asarray(loss_list))

    torch.save(model.state_dict(), 'unet-final-{}-{}-{}'.format(args.batch_size,
                                                                args.epochs,
                                                                args.lr))
elif args.load is not None:
    model.load_state_dict(torch.load(args.load))
    test(save_output=True)
    test(train_accuracy=True)
