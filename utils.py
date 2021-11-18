import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if len(display_list[i].shape)==2:
            img_mode='L'
            plt.imshow(Image.fromarray((display_list[i]).astype(np.uint8), mode=img_mode), cmap='gray')
        else:
            img_mode='RGB'
            plt.imshow(Image.fromarray((display_list[i] * 255).astype(np.uint8), mode=img_mode))

        plt.axis('off')
    plt.show()

def create_binary_mask(pred_mask_b, thre):
    max_val=np.max(pred_mask_b, axis=(1,2), keepdims=True)
    return np.where(pred_mask_b < thre*max_val, np.zeros_like(pred_mask_b), np.ones_like(pred_mask_b))

def create_tf_binary_mask(input, thre):
    y=tf.numpy_function(create_binary_mask, [input, thre], tf.float32)
    return y

def get_label_gt(true_masks):
    labels=np.array([])
    for true_mask in true_masks:
        sum_mask=np.sum(true_mask)
        if sum_mask>0:
            label=1
        else:
            label=0
        labels=np.append(labels, label)
    return labels

def create_binary (input, thre):
    return np.where(input < thre, np.zeros_like(input), np.ones_like(input))

def create_tf_binary(input, thre):
    y=tf.numpy_function(create_binary, [input, thre], tf.float32)
    return y



