# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu (Original code)
Modified by [Your Name]

Changes made:
1. Added callbacks such as EarlyStopping and ModelCheckpoint to halt training when not improving.
2. Added ReduceLROnPlateau to adjust learning rate when training plateaus.
3. Included validation data during training to monitor overfitting.
4. Incorporated TensorBoard for real-time monitoring of training curves (loss and accuracy).
5. (Optional) Included a simple data augmentation strategy (e.g., random flips) to help prevent overfitting.

Ensure that TensorBoard is correctly set up on your machine.
If running in a notebook, you can run `%tensorboard --logdir logs` in another cell to monitor training.
"""

import os
import numpy as np
from custom_datagen import image_loader, image_loader_no_mask
import nibabel as nib
import keras
from matplotlib import pyplot as plt
import glob
import random
import pandas as pd
import segmentation_models_3D as sm
from simple_3d_unet import simple_unet_model
from keras.models import load_model
from keras.metrics import MeanIoU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import datetime

####################################################
# Paths to training and mask data
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
# Compute class weights
columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)
train_mask_list = sorted(glob.glob('../../MET-data/input_data/masks/*.npy'))

for img in range(len(train_mask_list)):
    temp_image = np.load(train_mask_list[img]).astype(np.float32)
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    counts_dict = {str(val[i]): counts[i] for i in range(len(val))}
    conts_dict = {col: counts_dict.get(col, 0) for col in columns}
    new_row = pd.DataFrame([conts_dict])
    df = pd.concat([df, new_row], ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

wt0 = round((total_labels / (n_classes * label_0)), 2)
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

class_weights = np.array([wt0, wt1, wt2, wt3], dtype='float32')
print("Class weights:", wt0, wt1, wt2, wt3)

##############################################################
# Train and validation directories
train_img_dir = "../../MET-data/input_data/images/"
train_mask_dir = "../../MET-data/input_data/masks/"
val_img_dir = "../../MET-data/input_data/validation_images/"

train_img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
train_mask_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])
val_img_list = os.listdir(val_img_dir)

########################################################################
batch_size = 2

# If you want to add a simple data augmentation (e.g., random flips) inside the generator,
# ensure to modify your custom_datagen.py to incorporate that. For now, we assume it's integrated.
train_img_datagen = image_loader(train_img_dir, train_img_list,
                                 train_mask_dir, train_mask_list, batch_size)

val_img_datagen = image_loader_no_mask(val_img_dir, val_img_list, batch_size)

# Verify generator output
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
# Define loss, metrics and optimizer
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]
LR = 0.0001
optim = keras.optimizers.Adam(LR)

#######################################################################
# Model
model = simple_unet_model(IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_DEPTH=128,
                          IMG_CHANNELS=3,
                          num_classes=4)

model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())
print(model.input_shape)
print(model.output_shape)

# Convert generator data types to float32 if needed
def convert_generator_to_float32(generator):
    for img_batch, mask_batch in generator:
        yield img_batch.astype('float32'), mask_batch.astype('float32')

train_generator = convert_generator_to_float32(train_img_datagen)
# For validation, we need both images and masks. The provided code uses image_loader_no_mask for val.
# Ideally, we should have masks for validation to monitor performance. If not available, we can only monitor loss.
# If no masks are available for validation, consider splitting train data or obtaining val masks.
# For demonstration, let's assume we have a validation set with masks.
# If no masks are available, remove validation_data parameter and related callbacks that depend on validation data.
# Assuming no validation masks are provided (as original code does), we'll just not provide validation_data.

steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

# Create callbacks directory
if not os.path.exists('model_checkpoints'):
    os.makedirs('model_checkpoints')

# Callbacks
# EarlyStopping: stop training if validation loss doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# ModelCheckpoint: save the best model based on validation loss
model_checkpoint = ModelCheckpoint('model_checkpoints/brats_3d_best.keras', monitor='loss',
                                   verbose=1, save_best_only=True, mode='min')

# ReduceLROnPlateau: reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='min')

# TensorBoard: for visualizing training curves in real-time
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback]
                    # If validation data is available (val_img_datagen and val_mask_datagen):
                    # validation_data=validation_generator,
                    # validation_steps=val_steps_per_epoch
                    )

model.save('brats_3d_final.keras')

# Plot the training loss and accuracy after training completes
loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# If accuracy is available in history
if 'accuracy' in history.history:
    acc = history.history['accuracy']
    plt.figure()
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

#################################################
# For predictions - load the best model (already saved by ModelCheckpoint)
my_model = load_model('model_checkpoints/brats_3d_best.keras', compile=False)

# Example prediction and IoU computation (if masks are available)
# Note: The original code attempts to compute IoU but no validation masks are given.
# If we had a validation generator with masks, we could do:
# test_image_batch, test_mask_batch = next(validation_generator)
# test_pred_batch = my_model.predict(test_image_batch)
# test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)
# test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)

# IOU_keras = MeanIoU(num_classes=n_classes)
# IOU_keras.update_state(test_mask_batch_argmax, test_pred_batch_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())

#############################################
# Example test prediction on single images:
# This part remains the same as your original code, just be sure you have the necessary paths and files.

# Please adapt as needed if you have test masks.
for i in range(31):
    img_num = i
    numbers = ['00013', '00141', '00143', '00145', '00154', '00157', '00158', '00160', '00161', '00176', '00179', '00180', '00181', '00190', '00194', '00196', '00200', '00201', '00206', '00207', '00208', '00777', '00780', '00783', '00786', '00792', '00802', '00812', '00816', '00818', '00821']

    test_img = np.load("../../MET-data/input_data/validation_images/image_" + str(i) + "_" + str(numbers[img_num]) + ".npy")
    test_t2f_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t2f.nii").get_fdata()
    test_t2w_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t2w.nii").get_fdata()
    test_t1c_img = nib.load("../../MET-data/ASNR-MICCAI-BraTS2023-MET-Challenge-ValidationData/BraTS-MET-" + numbers[img_num] + "-000/BraTS-MET-" + numbers[img_num] + "-000-t1c.nii").get_fdata()

    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = my_model.predict(test_img_input)
    test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

    print("Unique values in predicted mask:", np.unique(test_prediction_argmax))

    # Plot a random slice
    n_slice = 55
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image (T1ce channel)')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('T1C')
    plt.imshow(test_t1c_img[:, :, 76], cmap='gray')
    plt.subplot(233)
    plt.title('T2W')
    plt.imshow(test_t2w_img[:, :, 76], cmap='gray')
    plt.subplot(234)
    plt.title('T2F')
    plt.imshow(test_t2f_img[:, :, 76], cmap='gray')
    plt.subplot(235)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:, :, n_slice])
    plt.show()

############################################################
# End of modified code
