import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from generator import fire_image_generator
from utils import display, create_tf_binary_mask, get_label_gt
from models import segmentation_network
from metrics import binary_miou_metric
import timeit
import os




def add_border(img, br_sz):
    img[1:br_sz, :] = 1
    img[:, 1:br_sz] = 1
    img[-1 - br_sz:-1, :] = 1
    img[:, -1 - br_sz:-1] = 1
    return img


def image_slicer(img, slice_num, sq_sz):
    (num_rows, num_cols) = slice_num
    (sq_sz_h, sq_sz_w) = sq_sz

    img_slices = []
    for i in range(num_rows):
        for j in range(num_cols):
            img_part = img[i * sq_sz_h:(i + 1) * sq_sz_h, j * sq_sz_w:(j + 1) * sq_sz_w, :]
            img_slices.append(img_part)
    return img_slices


def image_reconstruct(img_slices, slice_num, sq_sz):
    (num_rows, num_cols) = slice_num
    (sq_sz_h, sq_sz_w) = sq_sz
    recon_img = np.zeros((num_rows * sq_sz_h, num_cols * sq_sz_w, img_slices[0].shape[2]))
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            recon_img[i * sq_sz_h:(i + 1) * sq_sz_h, j * sq_sz_w:(j + 1) * sq_sz_w] = img_slices[k]
            k = k + 1
    return recon_img


def image_slicer_prediction(model, img, tile_sz):

    (sq_size_h, sq_size_w) = tile_sz

    orig_img_size = img.shape
    squares_in_y = int(np.floor(orig_img_size[0] / sq_size_h))
    squares_in_x = int(np.floor(orig_img_size[1] / sq_size_w))

    img_new_size = (squares_in_x * sq_size_w, squares_in_y * sq_size_h)
    img_inp = cv2.resize(img, img_new_size)
    img_tiles = image_slicer(img_inp, (squares_in_y, squares_in_x), (sq_size_h, sq_size_w))

    img_pred_tiles = []
    for tile in img_tiles:
        tile_shape=tile.shape
        if tile_shape[:2]!=(256,256):
            tile=cv2.resize(tile, (256,256))
        pred_class, pred_segment = model.predict(tile[np.newaxis, ...])
        pred_bin = (pred_segment > .5).astype(np.uint8)
        if tile_shape[:2]!=(256,256):
            pred_bin=cv2.resize(np.squeeze(pred_bin, axis=0), (tile_shape[1],tile_shape[0]))
            pred_bin=pred_bin[...,np.newaxis]
        else:
            pred_bin = np.squeeze(pred_bin, axis=0)
        pred_bin=add_border(pred_bin, 3)
        img_pred_tiles.append(pred_bin)

    return image_reconstruct(img_pred_tiles, (squares_in_y, squares_in_x), (sq_size_h, sq_size_w))


def quadtree_pred(model, orig_img, min_sq_sz, input_size, seg_threshold, perc_thre):
    orig_h, orig_w, orig_z = orig_img.shape
    w_img, h_img = int(np.floor(orig_w / 2)), int(np.floor(orig_h / 2))
    img_slices = image_slicer(orig_img, (2, 2), (h_img, w_img))
    pred_bin_slices = []
    for tile in img_slices:
        tile_sz = tile.shape
        tile_inp = cv2.resize(np.copy(tile), (input_size[1], input_size[0]))
        pred_class, pred_img = model.predict(tile_inp[np.newaxis, ...])
        pred_bin = (pred_img > seg_threshold).astype(np.uint8)
        class_pred=(pred_class > 0.01).astype(np.uint8)

        if (np.sum(pred_bin) > perc_thre * input_size[1] * input_size[0] or tile_sz[0] < min_sq_sz[0] or tile_sz[1] <
                min_sq_sz[1] or class_pred == 0):
            pred_bin = cv2.resize(np.squeeze(pred_bin, axis=0), (w_img, h_img))
            pred_bin=add_border(pred_bin, 3)
            pred_bin_slices.append(pred_bin[..., np.newaxis])
        else:
            pred_bin = quadtree_pred(model, tile, min_sq_sz, input_size, seg_threshold, perc_thre)
            pred_bin = cv2.resize(pred_bin, (w_img, h_img))
            pred_bin_slices.append(pred_bin[..., np.newaxis])
    return image_reconstruct(pred_bin_slices, (2, 2), (h_img, w_img))


def segmentation(model, orig_img, seg_threshold):
    orig_img = cv2.resize(orig_img, (256, 256))
    pred_class, pred_img = model.predict(orig_img[np.newaxis, ...])
    pred_bin = (pred_img > seg_threshold).astype(np.uint8)
    return pred_bin


BATCH_SIZE = 32
IMAGE_SIZE=(256,256)
MASK_SIZE=(256,256)
Single_img_evlt=1
pred_method='quad-tree'
#quadtree parameters
MIN_SIZE_SQ=(128,128)
min_perc=.2
seg_thre=.5

test_img_dir='../../dataset/test_weakly/rgb/rgb/'
test_mask_dir='../../dataset/test_weakly/masked/masked/'

TEST_LENGTH = len(os.walk(test_img_dir).__next__()[2])
test_ids = next(os.walk(test_img_dir))[2]


OUTPUT_CHANNELS = 1
#Model define
base_model_name='resnet'
model_name='deep_lab_att_2'
n_classes=OUTPUT_CHANNELS
#Model define
model=segmentation_network(base_model_name,model_name, n_classes, IMAGE_SIZE)


checkpint_folder = model_name+'-'+base_model_name+'-'+str(IMAGE_SIZE[0])
checkpoint_path='check_points/'+checkpint_folder+'/cp-0025.ckpt'
model.load_weights(checkpoint_path)


if Single_img_evlt==1:
    img_id='6'
    img_name=img_id+'.png'
    #img = np.asarray(Image.open(test_img_dir + img_name).convert('RGB'), dtype=float)
    img = np.asarray(Image.open('Sample_images/6.jpg').convert('RGB'), dtype=float)
    orig_img_sz=img.shape
    img_to_sh=np.copy(img)
    img=tf.keras.applications.resnet50.preprocess_input(img)
    if pred_method=='quad-tree':
        start = timeit.default_timer()
        pred_mask_b=quadtree_pred(model, img, MIN_SIZE_SQ, IMAGE_SIZE, seg_thre, min_perc)
        stop = timeit.default_timer()
        pred_mask_b=cv2.resize(pred_mask_b,(orig_img_sz[1], orig_img_sz[0]))
        Image.fromarray((np.squeeze(pred_mask_b)*255).astype(np.uint8)).save('Results/quadtree/quad_pred_mask'+img_id+'.png')
    elif pred_method=='slicer':
        start = timeit.default_timer()
        pred_mask_b=image_slicer_prediction(model, img, MIN_SIZE_SQ)
        stop = timeit.default_timer()
        pred_mask_b=cv2.resize(pred_mask_b,(orig_img_sz[1], orig_img_sz[0]))
        Image.fromarray((np.squeeze(pred_mask_b)*255).astype(np.uint8)).save('Results/quadtree/slicing_pred_mask'+img_id+'.png')
    elif pred_method=='segmentation':
        start = timeit.default_timer()
        pred_mask_b=segmentation(model, img, .5)
        stop = timeit.default_timer()
        pred_mask_b=cv2.resize(np.squeeze(pred_mask_b),(orig_img_sz[1], orig_img_sz[0]))
        Image.fromarray((np.squeeze(pred_mask_b)*255).astype(np.uint8)).save('Results/quadtree/seg_pred_mask'+img_id+'.png')
    display([img_to_sh, np.squeeze(pred_mask_b)])
    print('Computation time:', stop-start)

else:
    iou_v=np.zeros((TEST_LENGTH))
    i=0
    for img_name in test_ids:
        img = np.asarray(Image.open(test_img_dir + img_name).convert('RGB'), dtype=float)
        msk=np.asarray(Image.open(test_mask_dir+img_name).convert('L'), dtype=float)/255.0
        if pred_method=='quad-tree':
            pred_mask_b=quadtree_pred(model, img, MIN_SIZE_SQ, IMAGE_SIZE, .5, min_perc)

        elif pred_method=='slicer':
            pred_method=image_slicer_prediction(model, img, MIN_SIZE_SQ)


        msk=cv2.resize(msk, (pred_mask_b.shape[1], pred_mask_b.shape[0]))
        iou_v[i] = binary_miou_metric(msk[..., np.newaxis], pred_mask_b)
        i=i+1
# consistency = consistv.mean()
    iou = iou_v[:-1].mean()
    print('IOU:', iou)
