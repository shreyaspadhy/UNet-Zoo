
import os
from PIL import Image
import numpy as np
from skimage import color
from skimage import io


def getMaskFileName(file):

    mask_file = file.replace("flair.png", "seg.png")
    mask_file = mask_file.replace("t1.png", "seg.png")
    mask_file = mask_file.replace("t2.png", "seg.png")
    mask_file = mask_file.replace("t1ce.png", "seg.png")

    return mask_file


def getImg(imgpathway):
    # image_file = Image.open(imgpathway)  # open colour image
    # img = image_file.convert('L')
    # IMG = np.asarray(img.getdata())
    # img = io.imread(imgpathway, as_grey=True)
    img = Image.open(imgpathway)
    # img = np.asarray(img)
    # img *= 65536.0 / np.max(img)
    # IMG.astype(np.uint16)
    # plt.imshow(IMG, cmap='gray')
    # plt.show()
    return img


def File2Image(self, index):
    file = self.__file[index]
    filename_fragments = file.split("_")
    if filename_fragments[1] == '0' or filename_fragments[1] == '154':
        # Not sure what to do here
        return 0, 0

    filename1 = filename_fragments[0] + filename_fragments[1] + '_' + \
        str(int(filename_fragments[2]) - 1) + '_' + filename_fragments[3]
    filename3 = filename_fragments[0] + filename_fragments[1] + '_' + \
        str(int(filename_fragments[2]) + 1) + '_' + filename_fragments[3]

    idx1 = self.__file.index(filename1)
    idx3 = self.__file.index(filename3)
    img1 = self.__im[idx1]
    img3 = self.__im[idx3]

    return img1, img3
