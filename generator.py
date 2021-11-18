import tensorflow as tf
import numpy as np
from PIL import Image


class fire_image_generator (tf.keras.utils.Sequence):
    def __init__(self, image_path, mask_path, ids, batch_size, image_size, mask_size, normalization):
        self.indexes=ids
        self.image_path=image_path
        self.mask_path=mask_path
        self.batch_size=batch_size
        self.image_size=image_size
        self.mask_size=mask_size
        self.batch_size=batch_size
        self.normalization=normalization

    def __len__(self):
        return int(len(self.indexes)/float(self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'



    def get_labels(self, masks):
        labels=[]
        for mask in masks:
            if np.sum(mask)>0:
                label=1
            else:
                label=0
            labels.append(label)
        return np.array(labels)


    def __getitem__(self, index):
        indexes= self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        images=[np.asarray(Image.open(self.image_path+k).convert('RGB').resize(self.image_size), dtype=float) for k in indexes]
        masks=[np.asarray(Image.open(self.mask_path+k).convert('L').resize(self.mask_size), dtype=float) for k in indexes]
        if self.normalization:
            images=np.array(images)
            images = tf.keras.applications.resnet50.preprocess_input(images)
            #images=images/255.0
            masks=np.array(masks)/255.0
            masks=masks[..., tf.newaxis]
        labels=self.get_labels(masks)
        return images, [labels,masks]
