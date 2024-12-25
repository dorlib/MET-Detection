# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""

import os
import numpy as np
from custom_datagen import image_loader, image_loader_no_mask
# import tensorflow as tf
import nibabel as nib
import keras
from matplotlib import pyplot as plt
import glob
import random
import pandas as pd
import segmentation_models_3D as sm
from simple_3d_unet import simple_unet_model

####################################################
train_img_dir = "../../MET-data/input_data/images/"
train_mask_dir = "../../MET-data/input_data/masks/"

img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
print(img_list)
msk_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])
print(msk_list)


num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0, num_images - 1)
test_img = np.load(train_img_dir + img_list[img_num]).astype('float32')
test_mask = np.load(train_mask_dir + msk_list[img_num]).astype('float32')
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

#############################################################
# Optional step of finding the distribution of each class and calculating appropriate weights
# Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)
train_mask_list = sorted(glob.glob('../../MET-data/input_data/masks/*.npy'))

for img in range(len(train_mask_list)):
    temp_image = np.load(train_mask_list[img]).astype(np.float32)
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)

    # Ensure counts match the expected number of columns
    counts_dict = {str(val[i]): counts[i] for i in range(len(val))}
    conts_dict = {col: counts_dict.get(col, 0) for col in columns}

    # Use pd.concat to add the new row
    new_row = pd.DataFrame([conts_dict])
    df = pd.concat([df, new_row], ignore_index=True)

# Calculate label sums
label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

# Class weights calculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels / (n_classes * label_0)), 2)  # round to 2 decimals
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

# Convert weights to float32
class_weights = np.array([wt0, wt1, wt2, wt3], dtype='float32')

print(wt0, wt1, wt2, wt3)
# Weights are: 0.26, 22.53, 22.53, 26.21
# wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
# These weihts can be used for Dice loss

##############################################################
# Define the image generators for training and validation

train_img_dir = "../../MET-data/input_data/images/"
train_mask_dir = "../../MET-data/input_data/masks/"

val_img_dir = "../../MET-data/input_data/validation_images/"


train_img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
train_mask_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])
val_img_list = os.listdir(val_img_dir)


########################################################################
batch_size = 2

train_img_datagen = image_loader(train_img_dir, train_img_list,
                                 train_mask_dir, train_mask_list, batch_size)

val_img_datagen = image_loader_no_mask(val_img_dir, val_img_list, batch_size)


# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num].astype('float32')
test_mask = msk[img_num].astype('float32')
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')
plt.show()

###########################################################################
# Define loss, metrics and optimizer to be used for training
# wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################
# Fit the model

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size


model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)


# Ensure the generator outputs consistent data types
def convert_generator_to_float32(generator):
    for img_batch, mask_batch in generator:
        yield img_batch.astype('float32'), mask_batch.astype('float32')


train_img_datagen = convert_generator_to_float32(train_img_datagen)

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1)

model.save('brats_3d.hdf5')
# ##################################################################
#
#
# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accura cy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################
from keras.models import load_model

# Load model for prediction or continue training

# For continuing training....
# The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
# This is because the model does not save loss function and metrics. So to compile and
# continue training we need to provide these as custom_objects.
# my_model = load_model('brats_3d.hdf5')

# So let us add the loss as custom object... but the following throws another error...
# Unknown metric function: iou_score
# my_model = load_model('brats_3d.hdf5', custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# Now, let us add the iou_score function we used during our initial training
# my_model = load_model('brats_3d.hdf5',
#                       custom_objects={'dice_loss_plus_1focal_loss': total_loss,
#                                       'iou_score': sm.metrics.IOUScore(threshold=0.5)})
#
# # Now all set to continue the training process.
# history2 = my_model.fit(train_img_datagen,
#                         steps_per_epoch=steps_per_epoch,
#                         epochs=1,
#                         verbose=1,
#                         )
#################################################

# For predictions you do not need to compile the model, so ...
my_model = load_model('brats_3d.hdf5', compile=False)

# Verify IoU on a batch of images from the test dataset
# Using built in keras function for IoU
# Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size = 8  # Check IoU for a batch of images
test_img_datagen = image_loader_no_mask(val_img_dir, val_img_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch = test_img_datagen.__next__()

# test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
# IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#############################################
# Predict on a few test images, one at a time
# Try images:
for i in range(31):
    img_num = i
    numbers = ['00013', '00141', '00143', '00145', '00154', '00157', '00158', '00160', '00161', '00176', '00179', '00180', '00181', '00190', '00194', '00196', '00200', '00201', '00206', '00207', '00208', '00777', '00780', '00783', '00786', '00792', '00802', '00812', '00816', '00818', '00821']

    test_img = np.load("../../MET-data/input_data/validation_images/image_" + str(i) + "_" + str(numbers[img_num]) + ".npy")
    test_t2f_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t2f.nii").get_fdata()
    test_t2w_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t2w.nii").get_fdata()
    test_t1c_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t1c.nii").get_fdata()

    # test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_" + str(img_num) + ".npy")
    # test_mask_argmax = np.argmax(test_mask, axis=3)

    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = my_model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

    print("Unique values in predicted mask:", np.unique(test_prediction_argmax))
    # print("Unique values in training prediction:", np.unique(train_prediction_argmax))

    # print(test_prediction_argmax.shape)
    # print(test_mask_argmax.shape)
    # print(np.unique(test_prediction_argmax))


    # Plot individual slices from test predictions for verification
    from matplotlib import pyplot as plt
    import random

    # n_slice=random.randint(0, test_prediction_argmax.shape[2])
    n_slice = 55
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('T1C')
    plt.imshow(test_t1c_img[:, :, 76])
    plt.subplot(233)
    plt.title('T2W')
    plt.imshow(test_t2w_img[:, :, 76])
    plt.subplot(234)
    plt.title('T2F')
    plt.imshow(test_t2f_img[:, :, 76])
    plt.subplot(235)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:, :, n_slice])
    plt.show()

############################################################
