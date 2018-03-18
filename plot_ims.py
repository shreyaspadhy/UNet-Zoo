import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
# outs = np.load('Numpy-batch-7-outs.npy')
# masks = np.load('Numpy-batch-7-masks.npy')

final_outs = []
final_masks = []
final_images = []

loss = np.load('./OutMasks-UNet_Loss_bs=4_ep=1_lr=0.001.npy')

for i in range(38):
    outs = np.load(
        './npy-files/out-files/UNetSmall-NonZero-unetsmall-batch-{}-outs.npy'.format(i))
    masks = np.load(
        './npy-files/out-files/UNetSmall-NonZero-unetsmall--batch-{}-masks.npy'.format(i))
    images = np.load(
        './npy-files/out-files/UNetSmall-NonZero-unetsmall--batch-{}-images.npy'.format(i))

    final_outs.append(outs)
    final_masks.append(masks)
    final_images.append(images)
final_outs = np.asarray(final_outs)
final_masks = np.asarray(final_masks)
final_images = np.asarray(final_images)

print(final_images[0].shape)
for i in range(38):
    print(final_images[i].shape)
    plt.imshow(np.squeeze(final_images[i][49, :, :]), cmap='gray')
    plt.show()


# print(final_outs.shape)
# print(final_masks.shape)

# sio.savemat('./mat-files/final_outputs.mat', {'data': final_outs})
# sio.savemat('./mat-files/final_masks.mat', {'data': final_masks})
# sio.savemat('./mat-files/final_images.mat', {'data': final_images})
# for i in range(len(outs)):
#     plt1 = 255 * np.squeeze(outs[i, :, :, :]).astype('uint8')
#     plt2 = 255 * np.squeeze(masks[i, :, :, :]).astype('uint8')
#     print(plt1, plt2)
#     plt.subplot(1, 2, 1)
#     plt.imshow(plt1, cmap='gray')
#     plt.title("UNet Out")
#     plt.subplot(1, 2, 2)
#     plt.imshow(plt2, cmap='gray')
#     plt.title("Mask")
