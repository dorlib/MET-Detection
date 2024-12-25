# https://youtu.be/PNqnLbzdxwQ
"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators.

No image processing operations are performed here, just load data from local directory
in batches.

"""

# from tifffile import imsave, imread
import os
import numpy as np
from matplotlib import pyplot as plt
import random


def load_img(img_dir, img_list):
    images = []
    print('\n')
    for i, image_name in enumerate(img_list):
        print(f"Loading image/mask: {image_name}")  # Print the filename
        if image_name.split('.')[1] == 'npy':
            image = np.load(img_dir + image_name)
            images.append(image)
    images = np.array(images)
    return images


def image_loader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield X, Y  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size


def image_loader_no_mask(img_dir, img_list, batch_size):
    L = len(img_list)

    # keras needs the generator infinite, so we will use while true
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            x = load_img(img_dir, img_list[batch_start:limit])

            yield x  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size



############################################

# Test the generator

train_img_dir = "../../MET-data/input_data/images/"
train_mask_dir = "../../MET-data/input_data/masks/"
train_img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.npy')])
train_mask_list = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.npy')])

batch_size = 2

train_img_datagen = image_loader(train_img_dir, train_img_list,
                                train_mask_dir, train_mask_list, batch_size)

# Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

# Randomly select an image and mask
img_num = random.randint(0, img.shape[0] - 1)
print(f"Visualizing image {img_num} from batch.")
print(f"Image file: {train_img_list[img_num]}")  # Print the image filename
print(f"Mask file: {train_mask_list[img_num]}")  # Print the mask filename

test_img = img[img_num]
test_mask = msk[img_num]

test_mask = np.argmax(test_mask, axis=3)  # Convert mask to class indices

# Randomly select a slice for visualization
n_slice = random.randint(0, test_mask.shape[2] - 1)
print(f"Visualizing slice {n_slice}.")

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
