import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from losses import DICELossMultiClass

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from data import BraTSDatasetLSTM
from CLSTM import BDCLSTM
from models import *

# %% import transforms

UNET_MODEL_FILE = 'unetsmall-100-10-0.001'
MODALITY = ["flair"]

# %% Training settings
parser = argparse.ArgumentParser(description='UNet+BDCLSTM for BraTS Dataset')
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
parser.add_argument('--mom', type=float, default=0.99, metavar='MOM',
                    help='SGD momentum (default=0.99)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--test-dataset', action='store_true', default=False,
                    help='test on smaller dataset (default: False)')
parser.add_argument('--size', type=int, default=128, metavar='N',
                    help='imsize')
parser.add_argument('--drop', action='store_true', default=False,
                    help='enables drop')
parser.add_argument('--data-folder', type=str, default='./Data-Nonzero/', metavar='str',
                    help='folder that contains data (default: test dataset)')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print("We are on the GPU!")

DATA_FOLDER = args.data_folder

# %% Loading in the Dataset
dset_test = BraTSDatasetLSTM(
    DATA_FOLDER, keywords=MODALITY, transform=tr.ToTensor())
test_loader = DataLoader(
    dset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

dset_train = BraTSDatasetLSTM(
    DATA_FOLDER, keywords=MODALITY, transform=tr.ToTensor())
train_loader = DataLoader(
    dset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)


# %% Loading in the models
unet = UNetSmall()
unet.load_state_dict(torch.load(UNET_MODEL_FILE))
model = BDCLSTM(input_channels=32, hidden_channels=[32])

if args.cuda:
    unet.cuda()
    model.cuda()

# Setting Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
criterion = DICELossMultiClass()

# Define Training Loop


def train(epoch):
    model.train()
    for batch_idx, (image1, image2, image3, mask) in enumerate(train_loader):
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1), \
            Variable(image2), \
            Variable(image3), \
            Variable(mask)

        optimizer.zero_grad()

        map1 = unet(image1, return_features=True)
        map2 = unet(image2, return_features=True)
        map3 = unet(image3, return_features=True)

        output = model(map1, map2, map3)
        loss = criterion(output, mask)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(train_accuracy=False):
    test_loss = 0

    if train_accuracy == True:
        loader = train_loader
    else:
        loader = test_loader

    for (image1, image2, image3, mask) in loader:
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1, volatile=True), \
            Variable(image2, volatile=True), \
            Variable(image3, volatile=True), \
            Variable(mask, volatile=True)
        map1 = unet(image1, return_features=True)
        map2 = unet(image2, return_features=True)
        map3 = unet(image3, return_features=True)

        # print(image1.type)
        # print(map1.type)

        output = model(map1, map2, map3)
        test_loss += criterion(output, mask).data[0]

    test_loss /= len(loader)
    if train_accuracy:
        print(
            '\nTraining Set: Average Dice Coefficient: {:.4f}\n'.format(test_loss))
    else:
        print(
            '\nTest Set: Average Dice Coefficient: {:.4f}\n'.format(test_loss))


if args.train:
    for i in range(args.epochs):
        train(i)
        test()

    torch.save(model.state_dict(),
               'bdclstm-{}-{}-{}'.format(args.batch_size, args.epochs, args.lr))
else:
    model.load_state_dict(torch.load('bdclstm-{}-{}-{}'.format(args.batch_size,
                                                               args.epochs,
                                                               args.lr)))
    test()
    test(train_accuracy=True)
