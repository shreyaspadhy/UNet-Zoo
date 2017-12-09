import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as sio
from dataParser import getMaskFileName, getImg
import torchvision.transforms as tr


class BraTSDatasetUnet(Dataset):
    __file = []
    __im = []
    __mask = []
    im_ht = 0
    im_wd = 0
    dataset_size = 0

    def __init__(self, dataset_folder, train=True, keywords=["P1", "1", "flair"], im_size=[128, 128], transform=None):

        self.__file = []
        self.__im = []
        self.__mask = []
        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.transform = transform

        folder = dataset_folder
        # # Open and load text file including the whole training data
        if train:
            folder = dataset_folder + "Train/"
        else:
            folder = dataset_folder + "Test/"

        for file in os.listdir(folder):
            if file.endswith(".png"):
                filename = os.path.splitext(file)[0]
                filename_fragments = filename.split("_")
                samekeywords = list(set(filename_fragments) & set(keywords))
                if len(samekeywords) == len(keywords):
                    # 1. read file name
                    self.__file.append(filename)
                    # 2. read raw image
                    # TODO: I think we should open image only in getitem,
                    # otherwise memory explodes

                    # rawImage = getImg(folder + file)
                    self.__im.append(folder + file)
                    # 3. read mask image
                    mask_file = getMaskFileName(file)
                    # maskImage = getImg(folder + mask_file)
                    self.__mask.append(folder + mask_file)
        # self.dataset_size = len(self.__file)

        # print("lengths : ", len(self.__im), len(self.__mask))
        self.dataset_size = len(self.__file)

        if not train:
            sio.savemat('filelist2.mat', {'data': self.__im})

    def __getitem__(self, index):

        img = getImg(self.__im[index])
        mask = getImg(self.__mask[index])

        img = img.resize((self.im_ht, self.im_wd))
        mask = mask.resize((self.im_ht, self.im_wd))
        # mask.show()

        if self.transform is not None:
            # TODO: Not sure why not take full image
            img_tr = self.transform(img)
            mask_tr = self.transform(mask)
            # img_tr = self.transform(img[None, :, :])
            # mask_tr = self.transform(mask[None, :, :])

        return img_tr, mask_tr
        # return img.float(), mask.float()

    def __len__(self):

        return len(self.__im)


class BraTSDatasetLSTM(Dataset):
    __im = []
    __mask = []
    __im1 = []
    __im3 = []
    im_ht = 0
    im_wd = 0
    dataset_size = 0

    def __init__(self, dataset_folder, train=True, keywords=["P1", "1", "flair"], im_size=[128, 128], transform=None):

        self.__file = []
        self.__im = []
        self.__mask = []
        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.transform = transform

        folder = dataset_folder
        # # Open and load text file including the whole training data
        if train:
            folder = dataset_folder + "Train/"
        else:
            folder = dataset_folder + "Test/"

        # print("files : ", os.listdir(folder))
        # print("Folder : ", folder)
        max_file = 0
        min_file = 10000000
        for file in os.listdir(folder):
            if file.endswith(".png"):
                m = re.search('(P[0-9]*[_])([0-9]*)', file)
                pic_num = int(m.group(2))
                if pic_num > max_file:
                    max_file = pic_num
                if pic_num < min_file:
                    min_file = pic_num

        # print('min file number: ', min_file)
        # print('max file number: ', max_file)

        for file in os.listdir(folder):
            if file.endswith(".png"):
                filename = os.path.splitext(file)[0]
                filename_fragments = filename.split("_")
                samekeywords = list(set(filename_fragments) & set(keywords))
                if len(samekeywords) == len(keywords):
                    # 1. read file name
                    # 2. read raw image
                    # TODO: I think we should open image only in getitem,
                    # otherwise memory explodes

                    # rawImage = getImg(folder + file)

                    if (filename_fragments[2] != str(min_file)) and (filename_fragments[2] != str(max_file)):
                        # print("TEST : ", filename_fragments[2])
                        self.__im.append(folder + file)

                        file1 = filename_fragments[0] + '_' + filename_fragments[1] + '_' + str(
                            int(filename_fragments[2]) - 1) + '_' + filename_fragments[3] + '.png'

                        self.__im1.append(folder + file1)

                        file3 = filename_fragments[0] + '_' + filename_fragments[1] + '_' + str(
                            int(filename_fragments[2]) + 1) + '_' + filename_fragments[3] + '.png'

                        self.__im3.append(folder + file3)
                        # 3. read mask image
                        mask_file = getMaskFileName(file)
                        # maskImage = getImg(folder + mask_file)
                        self.__mask.append(folder + mask_file)
        # self.dataset_size = len(self.__file)

        # print("lengths : ", len(self.__im), len(self.__mask))
        self.dataset_size = len(self.__file)

    def __getitem__(self, index):

        img1 = getImg(self.__im1[index])
        img = getImg(self.__im[index])
        img3 = getImg(self.__im3[index])
        mask = getImg(self.__mask[index])

        # img.show()
        # mask.show()

        if self.transform is not None:
            # TODO: Not sure why not take full image
            img_tr1 = self.transform(img1)
            img_tr = self.transform(img)
            img_tr3 = self.transform(img3)
            mask_tr = self.transform(mask)
            # img_tr = self.transform(img[None, :, :])
            # mask_tr = self.transform(mask[None, :, :])

        return img_tr1, img_tr, img_tr3, mask_tr
        # return img.float(), mask.float()

    def __len__(self):

        return len(self.__im)
